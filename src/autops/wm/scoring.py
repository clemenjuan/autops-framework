"""Shared learned candidate scoring and planner-checkpoint verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from autops.missions.eventsat.transitions import record_number
from autops.wm.artifact import PlannerArtifact
from autops.wm.cem import one_hot_sequences
from autops.wm.guidance import CandidateProjection
from autops.wm.probes import (
    DEFAULT_ATTRIBUTES,
    FLOW_ATTRIBUTES,
    eventsat_attribute_values,
    scale_attribute_weights,
)


def rollout_attributes(readouts: np.ndarray, attribute_names: tuple[str, ...]) -> np.ndarray:
    """Reduce per-step readouts ``[..., horizon, attribute]`` to candidate attributes.

    Stocks are read at the terminal step; flows are summed over the horizon.
    """

    values = np.asarray(readouts)
    attributes = values[..., -1, :].copy()
    flows = [index for index, name in enumerate(attribute_names) if name in FLOW_ATTRIBUTES]
    attributes[..., flows] = values[..., flows].sum(axis=-2)
    return attributes


def analytical_candidate_attributes(
    state: Mapping[str, Any],
    projection: CandidateProjection,
    attribute_names: tuple[str, ...],
) -> np.ndarray:
    """Decode one projected bank from ``state`` with the canonical target definitions.

    Flows are the exact horizon increments of the projected cumulative counters.
    """

    unknown = set(attribute_names) - set(DEFAULT_ATTRIBUTES)
    if unknown:
        raise ValueError(f"analytical oracle received unknown attributes: {sorted(unknown)}")
    states = projection.terminal_states

    def column(name: str, default: float = 0.0) -> np.ndarray:
        return np.asarray([float(terminal.get(name, default)) for terminal in states])

    def gained(name: str) -> np.ndarray:
        return column(name) - record_number(state, name)

    stored = column("obc_data_mb") + column("jetson_raw_mb") + column("jetson_compressed_mb")
    all_attributes = eventsat_attribute_values(
        battery_soc=column("battery_soc", 0.5),
        stored_mb=stored,
        storage_capacity_mb=column("storage_capacity_mb", 4096.0),
        downlinked_mb=gained("data_downlinked_mb"),
        observation_s=gained("total_observation_s"),
        detections=gained("total_detections"),
        communication_opportunity=column("contact_window_active") > 0.0,
        forced_mode_risk=projection.terminal_forced,
        health_nominal=np.asarray(
            [terminal.get("health_status", "nominal") == "nominal" for terminal in states]
        ),
    )
    indices = [DEFAULT_ATTRIBUTES.index(name) for name in attribute_names]
    return all_attributes[:, indices]


def candidate_selection_metrics(
    scorer_scores: Mapping[str, np.ndarray],
    analytical_scores: np.ndarray,
    *,
    elites: int,
) -> dict[str, dict[str, float]]:
    """Compare scorer choices on one shared bank against an analytical oracle."""

    oracle = np.asarray(analytical_scores, dtype=np.float64)
    if oracle.ndim != 2 or not np.isfinite(oracle).all():
        raise ValueError("analytical_scores must be finite [contexts, candidates]")
    if isinstance(elites, bool) or not isinstance(elites, int) or not 0 < elites <= oracle.shape[1]:
        raise ValueError("elites must be a positive integer no larger than the candidate bank")
    oracle_order = np.argsort(oracle, axis=1, kind="stable")[:, -elites:]
    oracle_best = oracle.max(axis=1)
    evidence: dict[str, dict[str, float]] = {}
    for name, supplied in scorer_scores.items():
        values = np.asarray(supplied, dtype=np.float64)
        if values.shape != oracle.shape or not np.isfinite(values).all():
            raise ValueError(f"{name} scores must match the finite analytical score bank")
        selected = np.argmax(values, axis=1)
        selected_oracle = oracle[np.arange(oracle.shape[0]), selected]
        regret = oracle_best - selected_oracle
        scorer_order = np.argsort(values, axis=1, kind="stable")[:, -elites:]
        overlap = np.asarray(
            [
                len(set(oracle_row) & set(scorer_row)) / elites
                for oracle_row, scorer_row in zip(oracle_order, scorer_order, strict=True)
            ],
            dtype=np.float64,
        )
        evidence[str(name)] = {
            "top_elite_overlap": float(overlap.mean()),
            "analytical_regret_mean": float(regret.mean()),
            "analytical_regret_std": float(regret.std()),
            "analytical_regret_max": float(regret.max()),
            "analytical_best_selection_rate": float(np.mean(regret <= 1e-12)),
        }
    if not evidence:
        raise ValueError("at least one candidate scorer is required")
    return evidence


def scalarization_weights(
    artifact: PlannerArtifact,
    supplied: Mapping[str, Any],
    *,
    normalize_attribute_scale: bool | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Validate and order mission weights exactly as the artifact probe."""

    unknown = set(supplied) - set(artifact.probe.attribute_names)
    if unknown:
        raise ValueError(f"mission weights name unknown probe attributes: {sorted(unknown)}")
    numeric = {name: float(supplied.get(name, 0.0)) for name in artifact.probe.attribute_names}
    weights = np.asarray(list(numeric.values()), dtype=np.float32)
    if not np.isfinite(weights).all() or not np.any(weights):
        raise ValueError("mission weights must contain a finite non-zero value")
    normalize = (
        artifact.normalize_attribute_scale
        if normalize_attribute_scale is None
        else bool(normalize_attribute_scale)
    )
    if normalize:
        weights = scale_attribute_weights(
            weights, np.asarray(artifact.probe.objective_scale, dtype=np.float32)
        )
    return weights, numeric


def validate_planner_checkpoint(
    artifact: PlannerArtifact,
    checkpoint_contract: Any,
    checkpoint_digest: str,
    checkpoint_size_bytes: int,
) -> None:
    """Bind checkpoint bytes, semantics, dimensions, and normalization to an artifact."""

    if checkpoint_digest != artifact.model.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match PlannerArtifact")
    if checkpoint_size_bytes != artifact.probe_evidence.checkpoint_size_bytes:
        raise ValueError("checkpoint size does not match PlannerArtifact evidence")
    semantics = (
        checkpoint_contract.mission,
        checkpoint_contract.observation_names,
        checkpoint_contract.action_names,
        checkpoint_contract.trace_sha256,
    )
    expected_semantics = (
        artifact.model.mission,
        artifact.model.observation_names,
        artifact.model.action_names,
        artifact.model.trace_sha256,
    )
    if semantics != expected_semantics:
        raise ValueError("checkpoint mission/data semantics do not match PlannerArtifact")
    model_config = checkpoint_contract.model_config
    dimensions = (
        model_config.obs_dim,
        model_config.action_dim,
        model_config.embed_dim,
        model_config.history,
    )
    expected_dimensions = (
        artifact.model.obs_dim,
        artifact.model.action_dim,
        artifact.model.embed_dim,
        artifact.model.history,
    )
    if dimensions != expected_dimensions:
        raise ValueError("checkpoint dimensions do not match PlannerArtifact")
    checkpoint_normalizer = checkpoint_contract.normalizer
    for name in ("obs_mean", "obs_std", "action_mean", "action_std"):
        if not np.array_equal(
            np.asarray(getattr(checkpoint_normalizer, name), dtype=np.float32),
            np.asarray(getattr(artifact.normalization, name), dtype=np.float32),
        ):
            raise ValueError("checkpoint normalization does not match PlannerArtifact")


def latent_rollout(
    model: Any,
    artifact: PlannerArtifact,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    sequences: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Roll each history forward under its command sequence.

    Histories are ``[batch, history, dim]`` and sequences ``[batch, horizon]``;
    the result holds the predicted latents as ``[batch, horizon, embed]``.
    """

    from autops.wm.jepa import require_torch

    torch = require_torch()
    observations = np.asarray(observation_history, dtype=np.float32)
    actions = np.asarray(action_history, dtype=np.float32)
    candidates = np.asarray(sequences)
    if candidates.ndim != 2 or not np.issubdtype(candidates.dtype, np.integer):
        raise ValueError("candidate sequences must be a two-dimensional integer array")
    batch = candidates.shape[0]
    expected_obs = (batch, artifact.model.history, artifact.model.obs_dim)
    expected_actions = (batch, artifact.model.history, artifact.model.action_dim)
    if observations.shape != expected_obs:
        raise ValueError(f"observation history must have shape {expected_obs}")
    if actions.shape != expected_actions:
        raise ValueError(f"action history must have shape {expected_actions}")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("candidate history must be finite")
    future = one_hot_sequences(candidates, artifact.model.action_dim)
    normalizer = artifact.normalization
    obs_mean = np.asarray(normalizer.obs_mean, dtype=np.float32)
    obs_std = np.asarray(normalizer.obs_std, dtype=np.float32)
    action_mean = np.asarray(normalizer.action_mean, dtype=np.float32)
    action_std = np.asarray(normalizer.action_std, dtype=np.float32)
    with torch.no_grad():
        latent = model.rollout(
            torch.as_tensor((observations - obs_mean) / obs_std, device=device),
            torch.as_tensor((actions - action_mean) / action_std, device=device),
            torch.as_tensor((future - action_mean) / action_std, device=device),
        )
    predicted = latent.detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(predicted).all():
        raise ValueError("learned rollout contains non-finite values")
    return predicted


def latent_rollout_readouts(
    model: Any,
    artifact: PlannerArtifact,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    sequences: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Apply the frozen affine readouts to every latent of a rollout.

    The result is ``[batch, horizon, attribute]``; see ``latent_rollout``.
    """

    predicted = latent_rollout(
        model, artifact, observation_history, action_history, sequences, device=device
    )
    matrix = np.asarray(artifact.probe.W, dtype=np.float32)
    return predicted @ matrix.T + np.asarray(artifact.probe.b, dtype=np.float32)


def latent_candidate_attributes(
    model: Any,
    artifact: PlannerArtifact,
    observation_history: np.ndarray,
    action_history: np.ndarray,
    sequences: np.ndarray,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """Score a candidate bank from one history: terminal stocks and summed flows."""

    count = np.asarray(sequences).shape[0]
    readouts = latent_rollout_readouts(
        model,
        artifact,
        np.repeat(np.asarray(observation_history, dtype=np.float32)[None], count, axis=0),
        np.repeat(np.asarray(action_history, dtype=np.float32)[None], count, axis=0),
        sequences,
        device=device,
    )
    return rollout_attributes(readouts, artifact.probe.attribute_names)


__all__ = [
    "analytical_candidate_attributes",
    "candidate_selection_metrics",
    "latent_candidate_attributes",
    "latent_rollout",
    "latent_rollout_readouts",
    "rollout_attributes",
    "scalarization_weights",
    "validate_planner_checkpoint",
]
