"""P3 contact and eclipse timing audit.

For sampled contexts of held-out episodes, the start of the next station pass,
that pass's duration, the next eclipse entry and, in eclipse, the exit are
predicted in several ways and compared with the privileged labels. Censored
events (beyond the recorded episode) are excluded.

- ``recurrence``: onboard history only; the last observed event plus whole
  nominal orbital periods, and the last complete pass's observed duration.
- ``physics``: propagate the onboard navigation fix with the known station
  (``orbital.orekit.forecast_events_from_fix``), a legitimate onboard computation.
- ``latent-readout`` and ``record-readout``: affine readouts fitted on the
  checkpoint's training episodes from the encoded latent, or from the raw record
  as a control.
- ``lewm-rollout``: affine visibility and sunlight readouts, fitted on encoded
  latents, applied to every latent of a rollout under the logged commands. The
  event is the first crossing inside the rollout; later events count as missed.

Times are minutes at the 60 s step resolution; a pass shorter than a step can
fall between samples.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root, expand_coordinate
from autops.core.offline import (
    evaluation_record,
    held_out,
    load_planner_bundle,
    near_contact,
    planner_history,
    sample_contexts,
    write_evidence,
)
from autops.core.provenance import collect_provenance
from autops.core.replay import replay_environments
from autops.core.workflows import _latent_features
from autops.missions.eventsat.orbit import ground_station
from autops.orbital import EclipseInterval, GroundPass
from autops.orbital.orekit import forecast_events_from_fix
from autops.wm.dataset import episode_rows
from autops.wm.probes import affine_readout
from autops.wm.schema import EVENTSAT_OBSERVATIONS, TraceDataset
from autops.wm.scoring import latent_rollout

EVENT_SCHEMA_VERSION = "autops.event-timing-audit/v1"
EVENTS = ("pass_start_min", "pass_duration_s", "eclipse_entry_min", "eclipse_exit_min")
Context = tuple[int, int, bool]


def _next_index(flags: np.ndarray) -> np.ndarray:
    """Index of the first true flag at or after each step, or ``len`` when none."""

    result = np.full(flags.shape, flags.shape[-1], dtype=np.int64)
    upcoming = flags.shape[-1]
    for step in range(flags.shape[-1] - 1, -1, -1):
        if flags[step]:
            upcoming = step
        result[step] = upcoming
    return result


def event_labels(trace: TraceDataset) -> np.ndarray:
    """Privileged event times ``[episode, step, EVENTS]``; NaN where censored or undefined."""

    names = trace.metadata.state_names
    minutes = trace.metadata.timestep_s / 60.0
    steps = np.arange(trace.n_steps)
    labels = np.full((*trace.state.shape[:2], len(EVENTS)), np.nan)
    for episode, state in enumerate(trace.state):
        countdown = state[:, names.index("time_to_next_pass")].astype(np.int64)
        entry = state[:, names.index("time_to_next_eclipse")]
        contact = state[:, names.index("physical_contact_seconds")]
        sunlit = state[:, names.index("in_sunlight")] > 0.5
        valid = countdown >= 0
        labels[episode, valid, 0] = countdown[valid] * minutes
        start = np.minimum(steps + countdown, trace.n_steps - 1)
        run_end = _next_index(contact <= 0.0)[start]
        closed = valid & (steps + countdown < trace.n_steps) & (run_end < trace.n_steps)
        totals = np.concatenate([[0.0], np.cumsum(contact)])
        labels[episode, closed, 1] = (totals[run_end] - totals[start])[closed]
        labels[episode, entry >= 0, 2] = entry[entry >= 0] * minutes
        exit_step = _next_index(sunlit)
        shaded = ~sunlit & (exit_step < trace.n_steps)
        labels[episode, shaded, 3] = (exit_step - steps)[shaded] * minutes
    return labels


def _recurrence(trace: TraceDataset, episode: int, step: int, period: int) -> np.ndarray:
    """Predict from the last observed events and whole nominal orbital periods."""

    obs = trace.obs[episode, : step + 1]
    visible = obs[:, EVENTSAT_OBSERVATIONS.index("station_visible")] > 0.5
    sunlit = obs[:, EVENTSAT_OBSERVATIONS.index("in_sunlight")] > 0.5
    minutes = trace.metadata.timestep_s / 60.0

    def onsets(flags: np.ndarray) -> np.ndarray:
        return np.flatnonzero(flags[1:] & ~flags[:-1]) + 1

    def next_after(events: np.ndarray, earliest: int) -> float:
        if not events.size:
            return np.nan
        last = int(events[-1])
        return (last + max(0, -((last - earliest) // period)) * period - step) * minutes

    rises, falls = onsets(visible), onsets(~visible)
    complete = [(rise, fall) for rise in rises for fall in falls[falls > rise][:1]]
    duration = (complete[-1][1] - complete[-1][0]) * trace.metadata.timestep_s if complete else 0
    return np.asarray(
        [
            next_after(rises - 1, step),
            float(duration) if complete else np.nan,
            next_after(onsets(~sunlit), step + 1),
            np.nan if sunlit[-1] else next_after(onsets(sunlit), step + 1),
        ]
    )


def _nominal_period_steps(trace: TraceDataset) -> int:
    """The declared nominal orbital period in steps, as the environment floors it."""

    orbit = expand_coordinate(trace.metadata.sources[0].coordinate).mission_config["orbit"]
    return max(1, int(float(orbit["orbital_period_s"]) / trace.metadata.timestep_s))


def _interval_events(
    eclipses: tuple[EclipseInterval, ...],
    passes: tuple[GroundPass, ...],
    step_s: float,
    duration_s: float,
) -> np.ndarray:
    """Reduce forecast intervals measured from a fix to the audited events.

    Sampling closes open intervals at ``duration_s`` without marking them as
    censored. Treat endings at that boundary as unknown, preserving any observed
    onset; a real ending there cannot be distinguished from truncation.
    """

    future = [item for item in passes if item.start_s > 0.0]
    entries = [item for item in eclipses if item.start_s > 0.0]
    shaded = [item for item in eclipses if item.start_s == 0.0]
    return np.asarray(
        [
            (future[0].start_s // step_s) * step_s / 60.0 if future else np.nan,
            future[0].duration_s if future and future[0].end_s < duration_s else np.nan,
            entries[0].start_s / 60.0 if entries else np.nan,
            shaded[0].end_s / 60.0 if shaded and shaded[0].end_s < duration_s else np.nan,
        ]
    )


def _physics(trace: TraceDataset, contexts: list[Context], lookahead_steps: int) -> np.ndarray:
    """Propagate the onboard navigation fix of every replayed context."""

    step_s = trace.metadata.timestep_s
    duration_s = lookahead_steps * step_s
    forecasts: dict[tuple[int, int], np.ndarray] = {}
    for episode in sorted({episode for episode, _, _ in contexts}):
        wanted = [step for item, step, _ in contexts if item == episode]
        for step, env in replay_environments(trace, episode, wanted).items():
            fix = env.observe()["satellites"]["eventsat_0"]["metadata"]["navigation"]
            forecasts[episode, step] = np.full(len(EVENTS), np.nan)
            if fix.get("valid", False):
                eclipses, passes = forecast_events_from_fix(
                    fix["position_km"],
                    fix["velocity_km_s"],
                    datetime.fromisoformat(fix["utc"]),
                    ground_station(env.config),
                    duration_s=duration_s,
                    sample_s=step_s,
                )
                forecasts[episode, step] = _interval_events(eclipses, passes, step_s, duration_s)
    return np.stack([forecasts[episode, step] for episode, step, _ in contexts])


def _readouts(
    features: np.ndarray,
    labels: np.ndarray,
    train: tuple[int, ...],
    evaluation: np.ndarray,
) -> np.ndarray:
    """Fit one affine readout per event on uncensored training rows; predict ``evaluation``."""

    X = episode_rows(features, train, np.float64)
    Y = episode_rows(labels, train, np.float64)
    predictions = np.full((evaluation.shape[0], len(EVENTS)), np.nan)
    for index in range(len(EVENTS)):
        rows = np.isfinite(Y[:, index])
        if rows.sum() > X.shape[1]:
            W, b = affine_readout(X[rows], Y[rows, index : index + 1])
            predictions[:, index] = (evaluation @ W.T + b)[:, 0]
    return predictions


def _first_crossing(values: np.ndarray, current: bool, rising: bool) -> float:
    """Steps to the first crossing of 0.5 after ``current``; inf when none."""

    flags = np.concatenate([[current], values >= 0.5])
    change = flags[1:] & ~flags[:-1] if rising else ~flags[1:] & flags[:-1]
    hits = np.flatnonzero(change)
    return float(hits[0] + 1) if hits.size else np.inf


def _rollout_events(
    model: Any,
    artifact: Any,
    trace: TraceDataset,
    evaluation: TraceDataset,
    contexts: list[Context],
    train: tuple[int, ...],
    steps: int,
    device: str,
) -> np.ndarray:
    """Threshold visibility and sunlight readouts along rollouts under logged commands."""

    names = trace.metadata.state_names
    columns = [names.index("station_visible"), names.index("in_sunlight")]
    normalized = artifact.normalization
    latents = _latent_features(
        model,
        (trace.obs - np.asarray(normalized.obs_mean)) / np.asarray(normalized.obs_std),
        device,
    )
    W, b = affine_readout(
        episode_rows(latents, train), episode_rows(trace.state[..., columns], train)
    )
    histories = [planner_history(evaluation, e, t, artifact.model.history) for e, t, _ in contexts]
    episodes = np.asarray([episode for episode, _, _ in contexts])[:, None]
    offsets = np.asarray([step for _, step, _ in contexts])[:, None] + np.arange(steps)
    rollout = latent_rollout(
        model,
        artifact,
        np.stack([history["obs"] for history in histories]).astype(np.float32),
        np.stack([history["action"] for history in histories]).astype(np.float32),
        evaluation.mode[episodes, offsets],
        device=device,
    )
    readouts = rollout @ W.T + b
    current = evaluation.state[episodes[:, 0], offsets[:, 0]][:, columns] > 0.5
    minutes = evaluation.metadata.timestep_s / 60.0
    events = np.full((len(contexts), len(EVENTS)), np.nan)
    for index, (visible, sunlit) in enumerate(current):
        rise = _first_crossing(readouts[index, :, 0], visible, rising=True)
        events[index, 0] = (rise - 1) * minutes
        events[index, 2] = _first_crossing(readouts[index, :, 1], sunlit, rising=False) * minutes
        if not sunlit:
            events[index, 3] = _first_crossing(readouts[index, :, 1], sunlit, rising=True) * minutes
    return events


def _timing_scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Absolute timing error per event over contexts where both values exist."""

    scores: dict[str, Any] = {}
    for index, name in enumerate(EVENTS):
        known = np.isfinite(truth[:, index])
        both = known & np.isfinite(predicted[:, index])
        error = np.abs(predicted[both, index] - truth[both, index])
        scores[name] = {
            "n": int(both.sum()),
            "coverage": float(both.sum() / known.sum()) if known.any() else None,
            "mae": float(error.mean()) if both.any() else None,
            "median_ae": float(np.median(error)) if both.any() else None,
        }
    return scores


def _rollout_scores(predicted: np.ndarray, truth: np.ndarray, horizon_min: float) -> dict[str, Any]:
    """Detection inside the rollout, timing error when detected, and false alarms beyond it."""

    scores: dict[str, Any] = {}
    for index, name in enumerate(EVENTS):
        if np.isnan(predicted[:, index]).all():
            continue
        known = np.isfinite(truth[:, index]) & ~np.isnan(predicted[:, index])
        # Eclipse crossings include the final predicted state. Pass starts name
        # the preceding contact interval, so their upper bound remains strict.
        within = (
            truth[:, index] <= horizon_min
            if name.startswith("eclipse_")
            else truth[:, index] < horizon_min
        )
        inside = known & within
        beyond = known & ~inside
        detected = np.isfinite(predicted[:, index])
        hits = inside & detected
        error = np.abs(predicted[hits, index] - truth[hits, index])
        scores[name] = {
            "n_inside": int(inside.sum()),
            "detection_rate": float(hits.sum() / inside.sum()) if inside.any() else None,
            "mae": float(error.mean()) if hits.any() else None,
            "false_alarm_rate": (
                float((beyond & detected).sum() / beyond.sum()) if beyond.any() else None
            ),
        }
    return scores


def _metrics(
    methods: dict[str, np.ndarray], rollout: np.ndarray, truth: np.ndarray, horizon_min: float
) -> dict[str, Any]:
    return {
        **{name: _timing_scores(values, truth) for name, values in methods.items()},
        "lewm-rollout": _rollout_scores(rollout, truth, horizon_min),
    }


def audit_event_timing(
    trace_path: str | Path,
    artifact_path: str | Path,
    *,
    test_trace_path: str | Path | None = None,
    output: str | Path | None = None,
    contexts: int = 128,
    steps: int = 48,
    lookahead_steps: int = 1440,
    near_contact_fraction: float = 0.5,
    device: str = "cpu",
    seed: int = 3072,
) -> dict[str, Any]:
    trace, artifact, model, contract = load_planner_bundle(trace_path, artifact_path, device)
    evaluation, episodes = held_out(trace, test_trace_path, contract)
    if steps < 1 or steps >= evaluation.n_steps or lookahead_steps < 1:
        raise ValueError("steps must fit inside an episode and lookahead_steps be positive")
    sampled = sample_contexts(
        near_contact(evaluation, artifact.cem.horizon),
        episodes,
        contexts,
        near_contact_fraction,
        np.random.default_rng(seed),
        last_step=evaluation.n_steps - steps - 1,
    )
    at = (
        np.asarray([episode for episode, _, _ in sampled]),
        np.asarray([step for _, step, _ in sampled]),
    )
    train = contract.episodes.train
    normalizer = artifact.normalization
    obs_mean, obs_std = np.asarray(normalizer.obs_mean), np.asarray(normalizer.obs_std)
    labels = event_labels(trace)
    orbital_period = _nominal_period_steps(evaluation)
    methods = {
        "recurrence": np.stack(
            [_recurrence(evaluation, episode, step, orbital_period) for episode, step, _ in sampled]
        ),
        "physics": _physics(evaluation, sampled, lookahead_steps),
        "record-readout": _readouts(trace.obs, labels, train, evaluation.obs[at]),
        "latent-readout": _readouts(
            _latent_features(model, (trace.obs - obs_mean) / obs_std, device),
            labels,
            train,
            _latent_features(model, (evaluation.obs[at] - obs_mean) / obs_std, device),
        ),
    }
    rollout = _rollout_events(model, artifact, trace, evaluation, sampled, train, steps, device)
    truth = event_labels(evaluation)[at]
    settings = {
        "contexts": contexts,
        "steps": steps,
        "lookahead_steps": lookahead_steps,
        "near_contact_fraction": near_contact_fraction,
        "seed": seed,
    }
    minutes = evaluation.metadata.timestep_s / 60.0
    return write_evidence(
        output,
        {
            "schema_version": EVENT_SCHEMA_VERSION,
            "trace_sha256": artifact.model.trace_sha256,
            "checkpoint_sha256": artifact.model.checkpoint_sha256,
            "evaluation": evaluation_record(None if test_trace_path is None else evaluation),
            "config": settings,
            "provenance": collect_provenance(settings, asset_root()),
            "context_count": len(sampled),
            "metrics": _metrics(methods, rollout, truth, steps * minutes),
        },
    )


__all__ = ["EVENTS", "EVENT_SCHEMA_VERSION", "audit_event_timing", "event_labels"]
