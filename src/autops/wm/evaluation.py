"""Closed-loop evaluation through the same EventSat runner used in deployment.

The checkpoint's validation split selects seeds, not simulator snapshots.
Every distinct validation seed runs once: the collectors of a policy-diverse
corpus share each seed, and a reset from it yields the same episode.
Each episode starts from reset under the declared mission configuration and
executes the deployed representation, including masking, projection, shaping,
guidance, warm starts, held actions, reflexes, and compute-energy accounting.
These seeds informed checkpoint selection; they are not an untouched test set.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from autops.config import expand_coordinate
from autops.core.provenance import result_document_sha256, scientific_config_sha256
from autops.core.runner import ExperimentRunner
from autops.wm.artifact import (
    PlannerArtifact,
    artifact_sha256,
    checkpoint_sha256,
    load_artifact,
    resolve_checkpoint,
)
from autops.wm.probes import TARGET_DEFINITION_VERSION
from autops.wm.schema import TraceDataset, load_trace, trace_sha256
from autops.wm.scoring import validate_planner_checkpoint
from autops.wm.training import CheckpointContract, load_checkpoint

EVALUATION_SCHEMA_VERSION = "autops.lewm.cem-evaluation/v2"


def _validated_inputs(
    trace_path: Path, artifact_path: Path, destination: Path, device: str
) -> tuple[TraceDataset, PlannerArtifact, CheckpointContract]:
    trace = load_trace(trace_path)
    artifact = load_artifact(artifact_path)
    if artifact.model.mission != "eventsat" or trace.metadata.mission != "eventsat":
        raise ValueError("learned CEM evaluation requires EventSat")
    checkpoint = resolve_checkpoint(artifact_path, artifact)
    if destination.resolve() in {
        trace_path.resolve(),
        artifact_path.resolve(),
        checkpoint.resolve(),
    }:
        raise ValueError("evaluation output must not overwrite an input bundle")
    digest = checkpoint_sha256(checkpoint)
    if digest != artifact.model.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match PlannerArtifact")
    _, contract = load_checkpoint(checkpoint, device=device)
    validate_planner_checkpoint(artifact, contract, digest, checkpoint.stat().st_size)
    contract.validate_trace(trace)
    if trace_sha256(trace) != artifact.model.trace_sha256:
        raise ValueError("trace SHA-256 does not match PlannerArtifact")
    if (
        artifact.probe_evidence.train_episodes != contract.episodes.train
        or artifact.probe_evidence.validation_episodes != contract.episodes.validation
    ):
        raise ValueError("artifact probe split does not match checkpoint episode split")
    return trace, artifact, contract


def _portable_run(result: dict[str, Any]) -> dict[str, Any]:
    """Retain content identities without exposing local input placement."""

    experiment = result["experiment"]
    experiment["representation_config"].pop("artifact_path", None)
    result["provenance"]["config_sha256"] = scientific_config_sha256(experiment)
    result["result_id"] = result_document_sha256(result)
    return result


def _write_payload(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _evaluation_payload(
    trace: TraceDataset,
    artifact: PlannerArtifact,
    contract: CheckpointContract,
    selected: tuple[int, ...],
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "protocol": {
            "execution": "autops.core.runner.ExperimentRunner",
            "split": "checkpoint-selection validation seeds; not an untouched test set",
            "initialization": "environment reset; new closed-loop trajectories",
            "mission_configuration": "canonical YAML plus explicit overrides recorded in run",
            "validation_episode_indices": list(selected),
        },
        "contracts": {
            "trace_schema_version": trace.metadata.schema_version,
            "checkpoint_schema_version": contract.schema_version,
            "artifact_schema_version": artifact.schema_version,
            "probe_target_definition_version": TARGET_DEFINITION_VERSION,
            "cem_function": "autops.wm.cem.categorical_cem",
        },
        "artifact_defaults": {"cem": asdict(artifact.cem), "policy": artifact.planner_controls},
        "hashes": {
            "trace_sha256": artifact.model.trace_sha256,
            "checkpoint_sha256": artifact.model.checkpoint_sha256,
            "artifact_sha256": artifact_sha256(artifact),
        },
        "trace_sources": [source.to_dict() for source in trace.metadata.sources],
        "run": result,
    }


def evaluate_lewm_cem(
    trace_path: str | Path,
    artifact_path: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    mission_mode: str = "science",
    max_episodes: int = 5,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deployed LeWM-CEM on validation seeds with explicit mission settings.

    The trace establishes data identity and the validation split. Mission defaults
    come from the canonical YAML; ``overrides`` uses the ordinary matrix syntax.
    This is a new closed-loop experiment, not a replay of historical trace states.
    """

    if type(max_episodes) is not int or max_episodes <= 0:
        raise ValueError("max_episodes must be a positive integer")
    destination, artifact_file = Path(output), Path(artifact_path)
    if destination.suffix != ".json":
        raise ValueError("learned CEM evaluation output must use the .json suffix")
    trace, artifact, contract = _validated_inputs(
        Path(trace_path), artifact_file, destination, device
    )
    if mission_mode not in artifact.mode_weight_presets:
        raise ValueError(f"unknown mission_mode {mission_mode!r}")
    first_episode: dict[int, int] = {}
    for index in contract.episodes.validation:
        first_episode.setdefault(int(trace.episode_seed[index]), index)
    selected = tuple(first_episode.values())[:max_episodes]
    seeds = list(first_episode)[:max_episodes]
    backends = {source.orbital_backend for source in trace.metadata.sources}
    if len(backends) != 1:
        raise ValueError("evaluation requires one declared orbital backend in the trace")
    backend = backends.pop()
    settings = dict(overrides or {})
    representation = dict(settings.get("representation", {}))
    representation.update(
        artifact_path=str(artifact_file), device=device, mission_mode=mission_mode
    )
    settings["representation"] = representation
    spec = expand_coordinate(
        "eventsat/sas/ao/lewm-cem",
        episodes=len(seeds),
        steps=trace.n_steps,
        seeds=seeds,
        overrides=settings,
    )
    if spec.timestep_s != trace.metadata.timestep_s:
        raise ValueError(
            "evaluation timestep differs from trace; supply explicit mission overrides"
        )
    result = ExperimentRunner(spec, save=False, prefer_orekit=backend == "orekit").run()
    if any(episode["provenance"]["orbital_backend"] != backend for episode in result["episodes"]):
        raise ValueError("evaluation orbital backend differs from the declared trace backend")
    result = _portable_run(result)
    payload = _evaluation_payload(trace, artifact, contract, selected, result)
    _write_payload(destination, payload)
    return {
        "evaluation": str(destination),
        "episodes": len(seeds),
        "seeds": seeds,
        "steps_per_episode": spec.steps,
        "result_id": result["result_id"],
        "metrics": result["metrics"],
    }


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate_lewm_cem"]
