"""Affine mission-attribute probes with scale-free validation."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from autops.wm.dataset import EpisodeSplit, split_episodes
from autops.wm.schema import TraceDataset

TARGET_DEFINITION_VERSION = "autops.eventsat.probe-targets/v1"

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


def eventsat_attribute_values(
    *,
    battery_soc: np.ndarray,
    stored_mb: np.ndarray,
    storage_capacity_mb: np.ndarray,
    data_downlinked_mb: np.ndarray,
    total_observation_s: np.ndarray,
    total_detections: np.ndarray,
    communication_opportunity: np.ndarray,
    forced_mode_risk: np.ndarray,
    health_nominal: np.ndarray,
) -> np.ndarray:
    """Compute the canonical eight EventSat attributes for any matching axes."""

    capacity = np.maximum(np.asarray(storage_capacity_mb), 1.0)
    values = (
        np.clip((np.asarray(battery_soc) - 0.20) / 0.80, 0.0, 1.0),
        np.clip(1.0 - np.asarray(stored_mb) / capacity, 0.0, 1.0),
        np.asarray(data_downlinked_mb),
        np.asarray(total_observation_s) / 3600.0,
        np.asarray(total_detections),
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


def build_eventsat_targets(trace: TraceDataset) -> np.ndarray:
    """Build the eight planner attributes from simulator-native EventSat state."""

    if trace.metadata.mission != "eventsat":
        raise ValueError("EventSat targets require an EventSat trace")
    index = {name: i for i, name in enumerate(trace.metadata.state_names)}
    required = {
        "battery_soc",
        "ground_pass_active",
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
    return eventsat_attribute_values(
        battery_soc=state[..., index["battery_soc"]],
        stored_mb=stored,
        storage_capacity_mb=capacity,
        data_downlinked_mb=state[..., index["data_downlinked_mb"]],
        total_observation_s=state[..., index["total_observation_s"]],
        total_detections=state[..., index["total_detections"]],
        communication_opportunity=state[..., index["ground_pass_active"]] > 0.5,
        forced_mode_risk=trace.forced_mode,
        health_nominal=state[..., index["health_nominal"]],
    )


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


def _rows(values: np.ndarray, episodes: Sequence[int]) -> np.ndarray:
    selected = values[np.asarray(episodes, dtype=np.int64)]
    return selected.reshape(-1, selected.shape[-1]).astype(np.float64)


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
    episode_split = episodes or split_episodes(X.shape[0], train_fraction=train_fraction, seed=seed)
    Xtr, Ytr = _rows(X, episode_split.train), _rows(Y, episode_split.train)
    Xv, Yv = _rows(X, episode_split.validation), _rows(Y, episode_split.validation)

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
    "TARGET_DEFINITION_VERSION",
    "ProbeFit",
    "build_eventsat_targets",
    "eventsat_attribute_values",
    "fit_ridge_probe",
    "scale_attribute_weights",
]
