"""P5 candidate-selection audit on fixed banks at logged decision points.

At each sampled decision context of held-out episodes, one bank of requested
command sequences is scored by the deployed learned planner, which sees the
onboard view, and by the analytical forecast oracle, which sees the almanac.
Each projects the bank under its own information, exactly as in a planning
event. The measures are top-elite overlap, the oracle regret of the learned
choice, and how often it picks the oracle's best candidate; a uniformly random
scorer gives their chance level. Contexts are stratified by contact: contact
timing is what the learned model must supply and the oracle is given.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root
from autops.core.provenance import collect_provenance
from autops.core.replay import replay_records
from autops.representations.analytical_planner import EventSatAnalyticalCEM
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.artifact import artifact_sha256, load_artifact, resolve_checkpoint
from autops.wm.schema import EVENTSAT_ACTIONS, TraceDataset, load_trace, trace_sha256
from autops.wm.scoring import candidate_selection_metrics
from autops.wm.training import load_checkpoint

SELECTION_SCHEMA_VERSION = "autops.candidate-selection/v1"


def _history(trace: TraceDataset, episode: int, step: int, length: int) -> dict[str, np.ndarray]:
    """Rebuild the deployed planner's observation and command history at a step."""

    first = max(0, step - length + 1)
    padding = length - (step + 1 - first)
    obs = trace.obs[episode, first : step + 1]
    action = trace.action[episode, first : step + 1]
    charging = np.eye(len(EVENTSAT_ACTIONS), dtype=np.float32)[0]
    return {
        "obs": np.concatenate([np.repeat(obs[:1], padding, axis=0), obs]),
        "action": np.concatenate([np.repeat(charging[None], padding, axis=0), action]),
    }


def near_contact(trace: TraceDataset, horizon: int) -> np.ndarray:
    """Label steps where the station is visible or a pass begins within the horizon."""

    names = trace.metadata.state_names
    visible = trace.state[..., names.index("station_visible")] > 0.5
    countdown = trace.state[..., names.index("time_to_next_pass")]
    return visible | ((countdown >= 0.0) & (countdown <= horizon))


def _contexts(
    near: np.ndarray,
    episodes: tuple[int, ...],
    count: int,
    near_fraction: float,
    rng: np.random.Generator,
) -> list[tuple[int, int, bool]]:
    """Sample contexts near and away from contact; a short stratum is filled from the other."""

    pools = {
        flag: [
            (episode, int(step))
            for episode in episodes
            for step in np.flatnonzero(near[episode] == flag)
        ]
        for flag in (True, False)
    }
    count = min(count, len(pools[True]) + len(pools[False]))
    near_count = min(len(pools[True]), round(count * near_fraction))
    near_count = max(near_count, count - len(pools[False]))
    chosen: list[tuple[int, int, bool]] = []
    for flag, size in ((True, near_count), (False, count - near_count)):
        picks = rng.choice(len(pools[flag]), size=size, replace=False)
        chosen.extend((*pools[flag][int(index)], flag) for index in picks)
    return sorted(chosen)


def audit_candidate_selection(
    trace_path: str | Path,
    artifact_path: str | Path,
    *,
    test_trace_path: str | Path | None = None,
    output: str | Path | None = None,
    contexts: int = 64,
    candidates: int = 256,
    near_contact_fraction: float = 0.5,
    mission_mode: str = "science",
    device: str = "cpu",
    seed: int = 3072,
) -> dict[str, Any]:
    if contexts < 1 or not 0.0 <= near_contact_fraction <= 1.0:
        raise ValueError("contexts must be positive and near_contact_fraction in [0, 1]")
    trace = load_trace(trace_path)
    artifact = load_artifact(artifact_path)
    if trace_sha256(trace) != artifact.model.trace_sha256:
        raise ValueError("trace SHA-256 does not match PlannerArtifact")
    _, contract = load_checkpoint(resolve_checkpoint(artifact_path, artifact), device=device)
    config = {"artifact_path": str(artifact_path), "mission_mode": mission_mode}
    learned = EventSatLeWMCEM({**config, "device": device})
    oracle = EventSatAnalyticalCEM(config)
    if candidates < learned.cem.elites:
        raise ValueError("the candidate bank must hold at least one CEM elite set")
    if test_trace_path is None:
        evaluation, episodes = trace, contract.episodes.validation
    else:
        evaluation = load_trace(test_trace_path)
        reused = set(evaluation.episode_seed.tolist()) & set(contract.episode_seeds)
        if reused:
            raise ValueError(f"test seeds also occur in the training trace: {sorted(reused)}")
        episodes = tuple(range(evaluation.n_episodes))
    horizon = learned.cem.horizon
    rng = np.random.default_rng(seed)
    near = near_contact(evaluation, horizon)
    sampled = _contexts(near, episodes, contexts, near_contact_fraction, rng)
    scores: dict[str, list[np.ndarray]] = {"lewm-cem": [], "random": [], "oracle": []}
    for episode in sorted({episode for episode, _, _ in sampled}):
        steps = [step for item, step, _ in sampled if item == episode]
        records = replay_records(evaluation, episode, steps, planning_horizon=horizon)
        for step in steps:
            raw = records[step]
            bank = rng.integers(0, len(EVENTSAT_ACTIONS), size=(candidates, horizon))
            history = _history(evaluation, episode, step, artifact.model.history)
            for name, planner in (("lewm-cem", learned), ("oracle", oracle)):
                state = planner.encode_observation(raw)
                scores[name].append(planner.score_requested(state, history, bank))
            scores["random"].append(rng.random(candidates))
    banks = {name: np.stack(values) for name, values in scores.items()}
    oracle_scores = banks.pop("oracle")
    flags = np.asarray([flag for _, _, flag in sampled])
    metrics = {"all": candidate_selection_metrics(banks, oracle_scores, elites=learned.cem.elites)}
    for label, mask in (("near_contact", flags), ("away_from_contact", ~flags)):
        if mask.any():
            metrics[label] = candidate_selection_metrics(
                {name: values[mask] for name, values in banks.items()},
                oracle_scores[mask],
                elites=learned.cem.elites,
            )
    chosen = banks["lewm-cem"].argmax(axis=1)
    regret = oracle_scores.max(axis=1) - oracle_scores[np.arange(len(chosen)), chosen]
    settings = {
        "contexts": contexts,
        "candidates": candidates,
        "near_contact_fraction": near_contact_fraction,
        "mission_mode": mission_mode,
        "horizon": horizon,
        "elites": learned.cem.elites,
        "seed": seed,
    }
    payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "trace_sha256": artifact.model.trace_sha256,
        "artifact_sha256": artifact_sha256(artifact),
        "checkpoint_sha256": artifact.model.checkpoint_sha256,
        "evaluation": (
            {"episodes": "checkpoint-validation"}
            if test_trace_path is None
            else {"episodes": "test-trace", "test_trace_sha256": trace_sha256(evaluation)}
        ),
        "config": settings,
        "provenance": collect_provenance(settings, asset_root()),
        "metrics": metrics,
        "contexts": [
            {
                "episode": episode,
                "seed": int(evaluation.episode_seed[episode]),
                "step": step,
                "near_contact": flag,
                "lewm_cem_regret": float(value),
            }
            for (episode, step, flag), value in zip(sampled, regret, strict=True)
        ],
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        payload["output"] = str(destination)
    return payload


__all__ = ["SELECTION_SCHEMA_VERSION", "audit_candidate_selection", "near_contact"]
