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

from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root
from autops.core.offline import (
    context_record,
    evaluation_record,
    held_out,
    near_contact,
    planner_history,
    sample_contexts,
    write_evidence,
)
from autops.core.provenance import collect_provenance
from autops.core.replay import replay_environments
from autops.representations.analytical_planner import EventSatAnalyticalCEM
from autops.representations.cem_planner import EventSatCEMBase
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.artifact import artifact_sha256, load_artifact, resolve_checkpoint
from autops.wm.schema import EVENTSAT_ACTIONS, TraceDataset, load_trace, trace_sha256
from autops.wm.scoring import candidate_selection_metrics
from autops.wm.training import load_checkpoint

SELECTION_SCHEMA_VERSION = "autops.candidate-selection/v1"


def _score_banks(
    trace: TraceDataset,
    contexts: list[tuple[int, int, bool]],
    planners: dict[str, EventSatCEMBase],
    candidates: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Score one random requested bank per context with every planner and a random scorer."""

    learned = planners["lewm-cem"]
    scores: dict[str, list[np.ndarray]] = {name: [] for name in (*planners, "random")}
    for episode in sorted({episode for episode, _, _ in contexts}):
        steps = [step for item, step, _ in contexts if item == episode]
        snapshots = replay_environments(trace, episode, steps, planning_horizon=learned.cem.horizon)
        for step in steps:
            size = (candidates, learned.cem.horizon)
            bank = rng.integers(0, len(EVENTSAT_ACTIONS), size=size)
            history = planner_history(trace, episode, step, learned.artifact.model.history)
            for name, planner in planners.items():
                state = planner.encode_observation(snapshots[step].observe())
                scores[name].append(planner.score_requested(state, history, bank))
            scores["random"].append(rng.random(candidates))
    return {name: np.stack(values) for name, values in scores.items()}


def _selection_metrics(
    banks: dict[str, np.ndarray], oracle: np.ndarray, near: np.ndarray, elites: int
) -> dict[str, Any]:
    metrics = {"all": candidate_selection_metrics(banks, oracle, elites=elites)}
    for label, mask in (("near_contact", near), ("away_from_contact", ~near)):
        if mask.any():
            metrics[label] = candidate_selection_metrics(
                {name: values[mask] for name, values in banks.items()}, oracle[mask], elites=elites
            )
    return metrics


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
    trace = load_trace(trace_path)
    artifact = load_artifact(artifact_path)
    if trace_sha256(trace) != artifact.model.trace_sha256:
        raise ValueError("trace SHA-256 does not match PlannerArtifact")
    _, contract = load_checkpoint(resolve_checkpoint(artifact_path, artifact), device=device)
    config = {"artifact_path": str(artifact_path), "mission_mode": mission_mode}
    planners: dict[str, EventSatCEMBase] = {
        "lewm-cem": EventSatLeWMCEM({**config, "device": device}),
        "oracle": EventSatAnalyticalCEM(config),
    }
    cem = planners["lewm-cem"].cem
    if candidates < cem.elites:
        raise ValueError("the candidate bank must hold at least one CEM elite set")
    evaluation, episodes = held_out(trace, test_trace_path, contract)
    rng = np.random.default_rng(seed)
    near = near_contact(evaluation, cem.horizon)
    sampled = sample_contexts(near, episodes, contexts, near_contact_fraction, rng)
    banks = _score_banks(evaluation, sampled, planners, candidates, rng)
    oracle = banks.pop("oracle")
    flags = np.asarray([flag for _, _, flag in sampled])
    chosen = banks["lewm-cem"].argmax(axis=1)
    regret = oracle.max(axis=1) - oracle[np.arange(len(chosen)), chosen]
    settings = {
        "contexts": contexts,
        "candidates": candidates,
        "near_contact_fraction": near_contact_fraction,
        "mission_mode": mission_mode,
        "horizon": cem.horizon,
        "elites": cem.elites,
        "seed": seed,
    }
    return write_evidence(
        output,
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "trace_sha256": artifact.model.trace_sha256,
            "artifact_sha256": artifact_sha256(artifact),
            "checkpoint_sha256": artifact.model.checkpoint_sha256,
            "evaluation": evaluation_record(None if test_trace_path is None else evaluation),
            "config": settings,
            "provenance": collect_provenance(settings, asset_root()),
            "metrics": _selection_metrics(banks, oracle, flags, cem.elites),
            "contexts": [
                {**context_record(evaluation, context), "lewm_cem_regret": float(value)}
                for context, value in zip(sampled, regret, strict=True)
            ],
        },
    )


__all__ = ["SELECTION_SCHEMA_VERSION", "audit_candidate_selection"]
