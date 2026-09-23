"""Affine mission-attribute probes with scale-free validation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from autops.wm.dataset import EpisodeSplit, episode_rows, split_episodes
from autops.wm.schema import TraceDataset

TARGET_DEFINITION_VERSION = "autops.eventsat.probe-targets/v3"

DEFAULT_ATTRIBUTES = (
    "battery_margin",
    "storage_margin",
    "downlink_progress",
    "science_progress",
    "detection_progress",
    "communication_opportunity",
    "forced_mode_risk",
    "anomaly_safe",
)

# Progress attributes are flows: the amount completed in the interval that ends
# at a record, which the onboard record states through its last-interval outcome.
# Cumulative totals are not onboard inputs, so their level cannot be read from a
# latent; a planner sums flow readouts along a rollout, as physics probes decode
# per-step state increments (Li et al. 2026, arXiv:2608.16651) and world-model
# planners sum per-step reward predictions (DreamerV3, arXiv:2301.04104; TD-MPC2,
# arXiv:2310.16828). The remaining attributes are stocks read at a state.
FLOW_ATTRIBUTES = ("downlink_progress", "science_progress", "detection_progress")


def eventsat_attribute_values(
    *,
    battery_soc: np.ndarray,
    stored_mb: np.ndarray,
    storage_capacity_mb: np.ndarray,
    downlinked_mb: np.ndarray,
    observation_s: np.ndarray,
    detections: np.ndarray,
    communication_opportunity: np.ndarray,
    forced_mode_risk: np.ndarray,
    health_nominal: np.ndarray,
) -> np.ndarray:
    """Compute the canonical eight EventSat attributes for any matching axes.

    ``downlinked_mb``, ``observation_s`` and ``detections`` are amounts completed
    over an interval, not cumulative totals.
    """

    capacity = np.maximum(np.asarray(storage_capacity_mb), 1.0)
    values = (
        np.clip((np.asarray(battery_soc) - 0.20) / 0.80, 0.0, 1.0),
        np.clip(1.0 - np.asarray(stored_mb) / capacity, 0.0, 1.0),
        np.asarray(downlinked_mb),
        np.asarray(observation_s) / 3600.0,
        np.asarray(detections),
        np.asarray(communication_opportunity),
        np.asarray(forced_mode_risk),
        np.asarray(health_nominal),
    )
    return np.stack(values, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class ProbeFit:
    """Raw-unit affine readout plus normalization and validation evidence."""

    W: np.ndarray
    b: np.ndarray
    attribute_names: tuple[str, ...]
    target_mean: np.ndarray
    target_std: np.ndarray
    rmse: dict[str, float]
    rmse_over_std: dict[str, float]
    r2: dict[str, float]
    degenerate: tuple[str, ...]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        return (values @ self.W.T + self.b).astype(np.float32)


def build_eventsat_targets(trace: TraceDataset, *, totals: bool = False) -> np.ndarray:
    """Build the eight planner attributes from simulator-native EventSat state.

    ``totals`` reports each flow as the cumulative total it accumulates instead.
    """

    if trace.metadata.mission != "eventsat":
        raise ValueError("EventSat targets require an EventSat trace")
    index = {name: i for i, name in enumerate(trace.metadata.state_names)}
    required = {
        "battery_soc",
        "contact_window_active",
        "obc_data_mb",
        "jetson_raw_mb",
        "jetson_compressed_mb",
        "data_downlinked_mb",
        "total_observation_s",
        "total_detections",
        "storage_capacity_mb",
        "health_nominal",
    }
    missing = required - set(index)
    if missing:
        raise ValueError(f"EventSat state is missing probe fields: {sorted(missing)}")
    state = trace.state
    capacity = state[..., index["storage_capacity_mb"]]
    stored = sum(
        state[..., index[name]] for name in ("obc_data_mb", "jetson_raw_mb", "jetson_compressed_mb")
    )
    # A trace row stores s_t and the outgoing override for a_t. The latent at
    # s_t must be labeled with the incoming transition a_(t-1), not a future
    # command that was unavailable when s_t was predicted. Reset has no
    # incoming interval, so its override and flows are zero.
    incoming_forced = np.zeros_like(trace.forced_mode)
    incoming_forced[:, 1:] = trace.forced_mode[:, :-1]

    def incoming(name: str) -> np.ndarray:
        total = state[..., index[name]]
        if totals:
            return total
        flow = np.zeros_like(total)
        flow[:, 1:] = total[:, 1:] - total[:, :-1]
        return flow

    return eventsat_attribute_values(
        battery_soc=state[..., index["battery_soc"]],
        stored_mb=stored,
        storage_capacity_mb=capacity,
        downlinked_mb=incoming("data_downlinked_mb"),
        observation_s=incoming("total_observation_s"),
        detections=incoming("total_detections"),
        communication_opportunity=state[..., index["contact_window_active"]] > 0.5,
        forced_mode_risk=incoming_forced,
        health_nominal=state[..., index["health_nominal"]],
    )


def objective_scale(trace: TraceDataset, episodes: Sequence[int]) -> np.ndarray:
    """Return the per-attribute scale of preset weights over training episodes.

    Stocks use their own spread. Flows use the spread of the cumulative total
    they accumulate, which keeps the objective's historical balance: scaled by
    the spread of single steps, one rare observation step outweighs the battery
    by an order of magnitude, and even the analytical oracle then observes
    instead of staging data for downlink.
    """

    spread = episode_rows(build_eventsat_targets(trace, totals=True), episodes, np.float64).std(0)
    spread[spread < 1e-8] = 1.0
    return spread.astype(np.float32)


def _validate_probe_inputs(
    features: np.ndarray, targets: np.ndarray, names: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    X = np.asarray(features, dtype=np.float32)
    Y = np.asarray(targets, dtype=np.float32)
    attribute_names = tuple(str(name) for name in names)
    if X.ndim < 3 or Y.ndim < 3 or X.shape[:-1] != Y.shape[:-1]:
        raise ValueError("features and targets must share episode/sample axes")
    if Y.shape[-1] != len(attribute_names):
        raise ValueError("attribute_names must match the target dimension")
    if len(set(attribute_names)) != len(attribute_names):
        raise ValueError("attribute_names must be unique")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("probe inputs must be finite")
    return X, Y, attribute_names


def fit_ridge_probe(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    attribute_names: Sequence[str] = DEFAULT_ATTRIBUTES,
    ridge: float = 1e-3,
    episodes: EpisodeSplit | None = None,
    train_fraction: float = 0.8,
    seed: int = 0,
) -> ProbeFit:
    """Fit and validate an affine probe with complete episodes held out."""

    X, Y, names = _validate_probe_inputs(features, targets, attribute_names)
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    episode_split = episodes or split_episodes(
        range(X.shape[0]), train_fraction=train_fraction, seed=seed
    )
    train, validation = episode_split.train, episode_split.validation
    Xtr, Ytr = (episode_rows(values, train, np.float64) for values in (X, Y))
    Xv, Yv = (episode_rows(values, validation, np.float64) for values in (X, Y))

    x_mean, x_std = Xtr.mean(axis=0), Xtr.std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    target_mean, raw_target_std = Ytr.mean(axis=0), Ytr.std(axis=0)
    degenerate_mask = raw_target_std < 1e-8
    target_std = raw_target_std.copy()
    target_std[degenerate_mask] = 1.0
    degenerate = tuple(name for name, dead in zip(names, degenerate_mask, strict=False) if dead)
    if degenerate:
        warnings.warn(
            "degenerate zero-variance probe targets: " + ", ".join(degenerate),
            RuntimeWarning,
            stacklevel=2,
        )

    Xn = (Xtr - x_mean) / x_std
    Yn = (Ytr - target_mean) / target_std
    design = np.concatenate([Xn, np.ones((Xn.shape[0], 1))], axis=1)
    regularizer = ridge * np.eye(design.shape[1], dtype=np.float64)
    regularizer[-1, -1] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ Yn)
    normalized_W, normalized_b = coefficients[:-1].T, coefficients[-1]
    W = normalized_W / x_std * target_std[:, None]
    b = normalized_b * target_std + target_mean - W @ x_mean

    prediction = Xv @ W.T + b
    residual = prediction - Yv
    error = np.sqrt(np.mean(residual**2, axis=0))
    total = np.sum((Yv - Yv.mean(axis=0, keepdims=True)) ** 2, axis=0)
    unexplained = np.sum(residual**2, axis=0)
    rmse: dict[str, float] = {}
    rmse_over_std: dict[str, float] = {}
    r2: dict[str, float] = {}
    for i, name in enumerate(names):
        rmse[name] = float(error[i])
        if degenerate_mask[i]:
            rmse_over_std[name] = float("nan")
            r2[name] = float("nan")
        else:
            rmse_over_std[name] = float(error[i] / raw_target_std[i])
            r2[name] = float(1.0 - unexplained[i] / total[i]) if total[i] >= 1e-12 else float("nan")
    return ProbeFit(
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        attribute_names=names,
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
        rmse=rmse,
        rmse_over_std=rmse_over_std,
        r2=r2,
        degenerate=degenerate,
        train_episodes=episode_split.train,
        validation_episodes=episode_split.validation,
    )


def scale_attribute_weights(weights: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    """Apply the required raw-unit correction before candidate scalarization."""

    values = np.asarray(weights, dtype=np.float32)
    scale = np.asarray(target_std, dtype=np.float32)
    if values.ndim != 1 or scale.shape != values.shape or np.any(scale <= 0.0):
        raise ValueError("weights and positive target_std must be matching vectors")
    return values / scale


__all__ = [
    "DEFAULT_ATTRIBUTES",
    "FLOW_ATTRIBUTES",
    "TARGET_DEFINITION_VERSION",
    "ProbeFit",
    "build_eventsat_targets",
    "eventsat_attribute_values",
    "fit_ridge_probe",
    "objective_scale",
    "scale_attribute_weights",
]
