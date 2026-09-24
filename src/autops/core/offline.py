"""Inputs and evidence output shared by the offline world-model audits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from autops.wm.artifact import (
    PlannerArtifact,
    checkpoint_sha256,
    load_artifact,
    resolve_checkpoint,
)
from autops.wm.schema import EVENTSAT_ACTIONS, TraceDataset, load_trace, trace_sha256
from autops.wm.scoring import validate_planner_checkpoint
from autops.wm.training import CheckpointContract, load_checkpoint


def load_planner_bundle(
    trace_path: str | Path, artifact_path: str | Path, device: str
) -> tuple[TraceDataset, PlannerArtifact, Any, CheckpointContract]:
    """Load a training trace with the planner artifact and checkpoint bound to it."""

    trace = load_trace(trace_path)
    artifact = load_artifact(artifact_path)
    if trace_sha256(trace) != artifact.model.trace_sha256:
        raise ValueError("trace SHA-256 does not match PlannerArtifact")
    checkpoint = resolve_checkpoint(artifact_path, artifact)
    model, contract = load_checkpoint(checkpoint, device=device)
    validate_planner_checkpoint(
        artifact, contract, checkpoint_sha256(checkpoint), checkpoint.stat().st_size
    )
    return trace, artifact, model, contract


def held_out(
    trace: TraceDataset, test_trace_path: str | Path | None, contract: CheckpointContract
) -> tuple[TraceDataset, tuple[int, ...]]:
    """Return the checkpoint's validation episodes, or every episode of an untouched test trace."""

    if test_trace_path is None:
        return trace, contract.episodes.validation
    test = load_trace(test_trace_path)
    reused = set(test.episode_seed.tolist()) & set(contract.episode_seeds)
    if reused:
        raise ValueError(f"test seeds also occur in the training trace: {sorted(reused)}")
    if test.metadata.mission != "eventsat":
        raise ValueError("the test trace must be EventSat")
    return test, tuple(range(test.n_episodes))


def near_contact(trace: TraceDataset, horizon: int) -> np.ndarray:
    """Label steps where the station is visible or a pass begins within the horizon."""

    names = trace.metadata.state_names
    visible = trace.state[..., names.index("station_visible")] > 0.5
    countdown = trace.state[..., names.index("time_to_next_pass")]
    return visible | ((countdown >= 0.0) & (countdown <= horizon))


def sample_contexts(
    near: np.ndarray,
    episodes: tuple[int, ...],
    count: int,
    near_fraction: float,
    rng: np.random.Generator,
    *,
    last_step: int | None = None,
) -> list[tuple[int, int, bool]]:
    """Sample contexts near and away from contact; a short stratum is filled from the other."""

    if count < 1 or not 0.0 <= near_fraction <= 1.0:
        raise ValueError("contexts must be positive and near_contact_fraction in [0, 1]")
    stop = near.shape[1] if last_step is None else last_step + 1
    pools = {
        flag: [
            (episode, int(step))
            for episode in episodes
            for step in np.flatnonzero(near[episode, :stop] == flag)
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


def planner_history(
    trace: TraceDataset, episode: int, step: int, length: int
) -> dict[str, np.ndarray]:
    """Rebuild the deployed planner's observation and command history at a step.

    Like deployment, the first observation is repeated before an episode's
    start and the missing commands are charging.
    """

    first = max(0, step - length + 1)
    padding = length - (step + 1 - first)
    obs = trace.obs[episode, first : step + 1]
    action = trace.action[episode, first : step + 1]
    charging = np.eye(len(EVENTSAT_ACTIONS), dtype=np.float32)[0]
    return {
        "obs": np.concatenate([np.repeat(obs[:1], padding, axis=0), obs]),
        "action": np.concatenate([np.repeat(charging[None], padding, axis=0), action]),
    }


def write_evidence(output: str | Path | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Write an audit payload as JSON when an output is given and return it."""

    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")
        payload["output"] = str(destination)
    return payload


def evaluation_record(test_trace: TraceDataset | None) -> dict[str, Any]:
    """Describe which held-out episodes an audit scored."""

    if test_trace is None:
        return {"episodes": "checkpoint-validation"}
    return {
        "episodes": "test-trace",
        "test_trace_sha256": trace_sha256(test_trace),
        "test_seeds": [int(seed) for seed in test_trace.episode_seed],
    }


__all__ = [
    "evaluation_record",
    "held_out",
    "load_planner_bundle",
    "near_contact",
    "planner_history",
    "sample_contexts",
    "write_evidence",
]
