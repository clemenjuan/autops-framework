"""Offline held-out CEM evaluation using the deployed learned scorer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root
from autops.core.provenance import collect_provenance
from autops.wm.artifact import (
    PlannerArtifact,
    artifact_sha256,
    checkpoint_sha256,
    load_artifact,
    resolve_checkpoint,
)
from autops.wm.cem import categorical_cem
from autops.wm.probes import TARGET_DEFINITION_VERSION, build_eventsat_targets
from autops.wm.schema import TraceDataset, load_trace, trace_sha256
from autops.wm.scoring import (
    latent_candidate_attributes,
    scalarization_weights,
    validate_planner_checkpoint,
)
from autops.wm.training import CheckpointContract, load_checkpoint

EVALUATION_SCHEMA_VERSION = "autops.lewm.cem-evaluation/v1"


@dataclass(frozen=True)
class _EvaluationSetup:
    destination: Path
    trace: TraceDataset
    artifact: PlannerArtifact
    model: Any
    checkpoint_contract: CheckpointContract
    checkpoint_digest: str
    trace_digest: str
    device: str
    weights: np.ndarray
    raw_weights: dict[str, float]
    contexts: list[tuple[int, int]]
    targets: np.ndarray


@dataclass(frozen=True)
class _ContextOutcome:
    context_id: int
    episode: int
    timestep: int
    plan: np.ndarray
    recorded: np.ndarray
    planned_attributes: np.ndarray
    recorded_attributes: np.ndarray
    realized_attributes: np.ndarray
    planned_score: float
    recorded_score: float
    realized_score: float
    elite_mean: float
    elite_std: float
    first_action_probabilities: np.ndarray


def _contexts(
    n_steps: int,
    validation_episodes: tuple[int, ...],
    *,
    history: int,
    horizon: int,
    maximum: int,
) -> list[tuple[int, int]]:
    available = [
        (episode, timestep)
        for episode in validation_episodes
        for timestep in range(history - 1, n_steps - horizon)
    ]
    if not available:
        raise ValueError("held-out episodes contain no complete history-plus-horizon context")
    count = min(maximum, len(available))
    indices = np.linspace(0, len(available) - 1, num=count, dtype=np.int64)
    return [available[int(index)] for index in indices]


def _score_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _request_paths(
    trace_path: str | Path,
    artifact_path: str | Path,
    output: str | Path,
    max_contexts: int,
) -> tuple[Path, Path]:
    if isinstance(max_contexts, bool) or not isinstance(max_contexts, int) or max_contexts <= 0:
        raise ValueError("max_contexts must be a positive integer")
    destination = Path(output)
    if destination.suffix != ".json":
        raise ValueError("learned CEM evaluation output must use the .json suffix")
    artifact_file = Path(artifact_path)
    if destination.resolve() in {Path(trace_path).resolve(), artifact_file.resolve()}:
        raise ValueError("evaluation output must not overwrite a trace or artifact")
    return destination, artifact_file


def _validate_data_contracts(
    trace: TraceDataset,
    artifact: PlannerArtifact,
    contract: CheckpointContract,
) -> str:
    contract.validate_trace(trace)
    digest = trace_sha256(trace)
    if digest != artifact.model.trace_sha256:
        raise ValueError("trace SHA-256 does not match PlannerArtifact")
    if (
        artifact.probe_evidence.train_episodes != contract.episodes.train
        or artifact.probe_evidence.validation_episodes != contract.episodes.validation
    ):
        raise ValueError("artifact probe split does not match checkpoint episode split")
    return digest


def _canonical_device(device: str) -> str:
    from autops.wm.jepa import require_torch

    return str(require_torch().device(device))


def _load_setup(
    trace_path: str | Path,
    artifact_path: str | Path,
    output: str | Path,
    *,
    device: str,
    mission_mode: str,
    max_contexts: int,
) -> _EvaluationSetup:
    destination, artifact_file = _request_paths(trace_path, artifact_path, output, max_contexts)
    trace = load_trace(trace_path)
    artifact = load_artifact(artifact_file)
    if artifact.model.mission != "eventsat" or trace.metadata.mission != "eventsat":
        raise ValueError("learned CEM evaluation currently requires EventSat")
    selected_weights = artifact.mode_weight_presets.get(mission_mode)
    if selected_weights is None:
        raise ValueError(f"unknown mission_mode {mission_mode!r}")
    weights, raw_weights = scalarization_weights(artifact, selected_weights)
    checkpoint = resolve_checkpoint(artifact_file, artifact)
    if destination.resolve() == checkpoint.resolve():
        raise ValueError("evaluation output must not overwrite the artifact checkpoint")
    checkpoint_digest = checkpoint_sha256(checkpoint)
    model, contract = load_checkpoint(checkpoint, device=device)
    validate_planner_checkpoint(artifact, contract, checkpoint_digest, checkpoint.stat().st_size)
    digest = _validate_data_contracts(trace, artifact, contract)
    canonical_device = _canonical_device(device)
    contexts = _contexts(
        trace.n_steps,
        contract.episodes.validation,
        history=artifact.model.history,
        horizon=artifact.cem.horizon,
        maximum=max_contexts,
    )
    return _EvaluationSetup(
        destination=destination,
        trace=trace,
        artifact=artifact,
        model=model,
        checkpoint_contract=contract,
        checkpoint_digest=checkpoint_digest,
        trace_digest=digest,
        device=canonical_device,
        weights=weights,
        raw_weights=raw_weights,
        contexts=contexts,
        targets=build_eventsat_targets(trace),
    )


def _evaluate_context(
    setup: _EvaluationSetup,
    rng: np.random.Generator,
    context_id: int,
    episode: int,
    timestep: int,
) -> _ContextOutcome:
    artifact = setup.artifact
    history = artifact.model.history
    horizon = artifact.cem.horizon
    observations = setup.trace.obs[episode, timestep - history + 1 : timestep + 1]
    actions = setup.trace.action[episode, timestep - history + 1 : timestep + 1]

    def attributes(sequences: np.ndarray) -> np.ndarray:
        return latent_candidate_attributes(
            setup.model,
            artifact,
            observations,
            actions,
            sequences,
            device=setup.device,
        )

    def score(sequences: np.ndarray) -> np.ndarray:
        return attributes(sequences).astype(np.float64) @ setup.weights.astype(np.float64)

    result = categorical_cem(score, artifact.cem, rng=rng)
    plan = result.action_sequence.astype(np.int64, copy=False)
    recorded_slice = setup.trace.action[episode, timestep : timestep + horizon]
    recorded = np.argmax(recorded_slice, axis=-1).astype(np.int64)
    planned_attributes = attributes(plan[None])[0]
    recorded_attributes = attributes(recorded[None])[0]
    realized_attributes = setup.targets[episode, timestep + horizon]
    weight_vector = setup.weights.astype(np.float64)
    planned_score = float(planned_attributes.astype(np.float64) @ weight_vector)
    recorded_score = float(recorded_attributes.astype(np.float64) @ weight_vector)
    realized_score = float(realized_attributes.astype(np.float64) @ weight_vector)
    if not math.isclose(planned_score, result.score, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("CEM result score disagrees with the shared learned scorer")
    return _ContextOutcome(
        context_id=context_id,
        episode=episode,
        timestep=timestep,
        plan=plan,
        recorded=recorded,
        planned_attributes=planned_attributes,
        recorded_attributes=recorded_attributes,
        realized_attributes=realized_attributes,
        planned_score=planned_score,
        recorded_score=recorded_score,
        realized_score=realized_score,
        elite_mean=float(result.elite_scores.mean()),
        elite_std=float(result.elite_scores.std()),
        first_action_probabilities=result.probabilities[0],
    )


def _attribute_values(names: tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def _action_values(names: tuple[str, ...], values: np.ndarray) -> list[str]:
    return [names[int(value)] for value in values]


def _context_evidence(setup: _EvaluationSetup, outcome: _ContextOutcome) -> dict[str, Any]:
    names = setup.artifact.probe.attribute_names
    action_names = setup.artifact.model.action_names
    improvement = outcome.planned_score - outcome.recorded_score
    prediction_error = outcome.recorded_score - outcome.realized_score
    return {
        "context_id": outcome.context_id,
        "episode_id": outcome.episode,
        "episode_seed": int(setup.trace.episode_seed[outcome.episode]),
        "timestep": outcome.timestep,
        "planned_action_indices": [int(value) for value in outcome.plan],
        "planned_actions": _action_values(action_names, outcome.plan),
        "recorded_action_indices": [int(value) for value in outcome.recorded],
        "recorded_actions": _action_values(action_names, outcome.recorded),
        "scores": {
            "planned_model": outcome.planned_score,
            "recorded_policy_model": outcome.recorded_score,
            "recorded_policy_realized": outcome.realized_score,
            "model_improvement": improvement,
            "recorded_model_error": prediction_error,
            "last_iteration_elite_mean": outcome.elite_mean,
            "last_iteration_elite_std": outcome.elite_std,
        },
        "planned_attributes": _attribute_values(names, outcome.planned_attributes),
        "recorded_attributes": _attribute_values(names, outcome.recorded_attributes),
        "realized_recorded_attributes": _attribute_values(names, outcome.realized_attributes),
        "first_action_probabilities": [
            float(value) for value in outcome.first_action_probabilities
        ],
    }


def _outcome_array(outcomes: list[_ContextOutcome], name: str) -> np.ndarray:
    return np.asarray([getattr(outcome, name) for outcome in outcomes], dtype=np.float64)


def _aggregate(setup: _EvaluationSetup, outcomes: list[_ContextOutcome]) -> dict[str, Any]:
    planned = _outcome_array(outcomes, "planned_score")
    recorded = _outcome_array(outcomes, "recorded_score")
    realized = _outcome_array(outcomes, "realized_score")
    prediction_errors = recorded - realized
    attribute_errors = np.stack(
        [outcome.recorded_attributes - outcome.realized_attributes for outcome in outcomes]
    ).astype(np.float64)
    improvements = planned - recorded
    action_names = setup.artifact.model.action_names
    first_actions = [int(outcome.plan[0]) for outcome in outcomes]
    action_counts = {name: first_actions.count(index) for index, name in enumerate(action_names)}
    attribute_rmse = np.sqrt(np.mean(attribute_errors**2, axis=0))
    return {
        "context_count": len(outcomes),
        "planned_score": _score_summary(planned),
        "recorded_policy_score": _score_summary(recorded),
        "recorded_policy_realized_score": _score_summary(realized),
        "recorded_model_error": _score_summary(prediction_errors),
        "recorded_attribute_rmse": _attribute_values(
            setup.artifact.probe.attribute_names, attribute_rmse
        ),
        "improvement": _score_summary(improvements),
        "fraction_improved": float(np.mean(improvements > 0.0)),
        "first_action_counts": action_counts,
        "cem_candidate_rollouts": len(outcomes)
        * setup.artifact.cem.samples
        * setup.artifact.cem.iterations,
    }


def _configuration(setup: _EvaluationSetup, mission_mode: str, max_contexts: int) -> dict[str, Any]:
    artifact = setup.artifact
    return {
        "device": setup.device,
        "mission_mode": mission_mode,
        "max_contexts": max_contexts,
        "cem": asdict(artifact.cem),
        "scalarization_weights": setup.raw_weights,
        "effective_scalarization_weights": _attribute_values(
            artifact.probe.attribute_names, setup.weights
        ),
        "normalize_attribute_scale": artifact.normalize_attribute_scale,
        "objective": "learned_terminal_probe_scalarization",
        "context_selection": "evenly_spaced_complete_validation_contexts/v1",
    }


def _payload(
    setup: _EvaluationSetup,
    outcomes: list[_ContextOutcome],
    mission_mode: str,
    max_contexts: int,
) -> dict[str, Any]:
    configuration = _configuration(setup, mission_mode, max_contexts)
    artifact = setup.artifact
    contract = setup.checkpoint_contract
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "config": configuration,
        "contracts": {
            "trace_schema_version": setup.trace.metadata.schema_version,
            "checkpoint_schema_version": contract.schema_version,
            "artifact_schema_version": artifact.schema_version,
            "probe_target_definition_version": TARGET_DEFINITION_VERSION,
            "cem_function": "autops.wm.cem.categorical_cem",
        },
        "hashes": {
            "trace_sha256": setup.trace_digest,
            "checkpoint_sha256": setup.checkpoint_digest,
            "artifact_sha256": artifact_sha256(artifact),
        },
        "trace_sources": [source.to_dict() for source in setup.trace.metadata.sources],
        "runtime_provenance": collect_provenance(configuration, asset_root()),
        "held_out_episodes": list(contract.episodes.validation),
        "contexts": [_context_evidence(setup, outcome) for outcome in outcomes],
        "aggregate": _aggregate(setup, outcomes),
    }


def _write_payload(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _summary(setup: _EvaluationSetup, outcomes: list[_ContextOutcome]) -> dict[str, Any]:
    planned = _outcome_array(outcomes, "planned_score")
    recorded = _outcome_array(outcomes, "recorded_score")
    return {
        "evaluation": str(setup.destination),
        "contexts": len(outcomes),
        "planned_score_mean": float(planned.mean()),
        "recorded_policy_score_mean": float(recorded.mean()),
    }


def evaluate_lewm_cem(
    trace_path: str | Path,
    artifact_path: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    mission_mode: str = "science",
    max_contexts: int = 32,
) -> dict[str, Any]:
    """Evaluate artifact CEM on deterministic contexts from held-out episodes."""

    setup = _load_setup(
        trace_path,
        artifact_path,
        output,
        device=device,
        mission_mode=mission_mode,
        max_contexts=max_contexts,
    )
    rng = np.random.default_rng(setup.artifact.cem.seed)
    outcomes = [
        _evaluate_context(setup, rng, context_id, episode, timestep)
        for context_id, (episode, timestep) in enumerate(setup.contexts)
    ]
    _write_payload(setup.destination, _payload(setup, outcomes, mission_mode, max_contexts))
    return _summary(setup, outcomes)


__all__ = ["EVALUATION_SCHEMA_VERSION", "evaluate_lewm_cem"]
