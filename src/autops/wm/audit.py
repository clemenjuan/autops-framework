"""Episode-disjoint linear-versus-MLP decodability audit for frozen features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from autops.wm.dataset import EpisodeSplit, split_episodes
from autops.wm.jepa import require_torch
from autops.wm.probes import fit_ridge_probe


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class ProbeHeadResult:
    linear_r2: float
    mlp_r2: float
    linear_rmse_over_std: float
    mlp_rmse_over_std: float
    mlp_minus_linear_r2: float
    positive_rate: float | None = None
    linear_auc: float | None = None
    mlp_auc: float | None = None
    degenerate: bool = False


@dataclass(frozen=True)
class ProbeAudit:
    attributes: dict[str, ProbeHeadResult]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    feature_window: int
    hidden: tuple[int, ...]
    mlp_epochs: int

    def to_dict(self) -> dict[str, Any]:
        return _json_value(
            {
                "train_episodes": list(self.train_episodes),
                "validation_episodes": list(self.validation_episodes),
                "feature_window": self.feature_window,
                "hidden": list(self.hidden),
                "mlp_epochs": self.mlp_epochs,
                "attributes": {
                    name: {key: value for key, value in vars(result).items() if value is not None}
                    for name, result in self.attributes.items()
                },
            }
        )


def stack_feature_history(features: np.ndarray, window: int) -> np.ndarray:
    """Concatenate K frames with episode-local edge padding."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or window < 1:
        raise ValueError("features must be [episode, time, dim] and window positive")
    if window == 1:
        return values
    frames: list[np.ndarray] = []
    for lag in range(window - 1, -1, -1):
        shifted = np.roll(values, lag, axis=1)
        shifted[:, :lag] = values[:, :1]
        frames.append(shifted)
    return np.concatenate(frames, axis=-1)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney ROC-AUC with average ranks for tied scores."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(labels).reshape(-1) > 0.5
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    sorted_values = values[order]
    start = 0
    while start < len(sorted_values):
        stop = start + 1
        while stop < len(sorted_values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = ranks[order[start:stop]].mean()
        start = stop
    rank_sum = ranks[positive].sum()
    return float((rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative))


def _rows(values: np.ndarray, episodes: Sequence[int]) -> np.ndarray:
    selected = values[np.asarray(episodes, dtype=np.int64)]
    return selected.reshape(-1, selected.shape[-1]).astype(np.float32)


def _standardize(
    train: np.ndarray, validation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        ((train - mean) / std).astype(np.float32),
        ((validation - mean) / std).astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def _fit_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_validation: np.ndarray,
    *,
    hidden: tuple[int, ...],
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> np.ndarray:
    torch = require_torch()
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    layers: list[Any] = []
    previous = X_train.shape[1]
    for width in hidden:
        layers.extend([torch.nn.Linear(previous, width), torch.nn.ReLU(), torch.nn.Dropout(0.1)])
        previous = width
    layers.append(torch.nn.Linear(previous, Y_train.shape[1]))
    network = torch.nn.Sequential(*layers).to(torch_device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = torch.nn.MSELoss()
    inputs = torch.from_numpy(X_train).to(torch_device)
    targets = torch.from_numpy(Y_train).to(torch_device)
    validation = torch.from_numpy(X_validation).to(torch_device)
    order = np.arange(len(inputs))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(order)
        network.train()
        for start in range(0, len(order), 2048):
            batch = order[start : start + 2048]
            optimizer.zero_grad(set_to_none=True)
            loss_function(network(inputs[batch]), targets[batch]).backward()
            optimizer.step()
    network.eval()
    with torch.no_grad():
        return network(validation).cpu().numpy().astype(np.float32)


def _r2_rmse(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = prediction - target
    rmse = np.sqrt(np.mean(error**2, axis=0))
    denominator = np.sum((target - target.mean(axis=0, keepdims=True)) ** 2, axis=0)
    unexplained = np.sum(error**2, axis=0)
    r2 = np.full(denominator.shape, np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    r2[valid] = 1.0 - unexplained[valid] / denominator[valid]
    return r2, rmse


@dataclass(frozen=True)
class _ProbePredictions:
    linear: Any
    validation_targets: np.ndarray
    target_std: np.ndarray
    linear_values: np.ndarray
    mlp_values: np.ndarray
    mlp_r2: np.ndarray
    mlp_rmse: np.ndarray
    binary: tuple[bool, ...]


def _comparison_inputs(
    features: np.ndarray,
    targets: np.ndarray,
    attribute_names: Sequence[str],
    *,
    feature_window: int,
    ridge: float,
    hidden: tuple[int, ...],
    mlp_epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    X = stack_feature_history(features, feature_window)
    Y = np.asarray(targets, dtype=np.float32)
    names = tuple(str(name) for name in attribute_names)
    if Y.ndim != 3 or X.shape[:2] != Y.shape[:2] or Y.shape[-1] != len(names):
        raise ValueError("features and targets must share episode/time axes and named outputs")
    if mlp_epochs < 1 or not hidden or any(width < 1 for width in hidden):
        raise ValueError("the MLP audit requires positive epochs and hidden widths")
    if ridge < 0.0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("invalid probe audit optimizer configuration")
    return X, Y, names


def _fit_probe_predictions(
    X: np.ndarray,
    Y: np.ndarray,
    names: tuple[str, ...],
    split: EpisodeSplit,
    *,
    ridge: float,
    hidden: tuple[int, ...],
    mlp_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    device: str,
) -> _ProbePredictions:
    linear = fit_ridge_probe(
        X,
        Y,
        attribute_names=names,
        ridge=ridge,
        episodes=split,
        seed=seed,
    )
    X_train, X_validation = _rows(X, split.train), _rows(X, split.validation)
    Y_train, Y_validation = _rows(Y, split.train), _rows(Y, split.validation)
    X_train_n, X_validation_n, _, _ = _standardize(X_train, X_validation)
    Y_train_n, _, target_mean, target_std = _standardize(Y_train, Y_validation)
    prediction_mlp_n = _fit_mlp(
        X_train_n,
        Y_train_n,
        X_validation_n,
        hidden=hidden,
        epochs=mlp_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    prediction_mlp = prediction_mlp_n * target_std + target_mean
    prediction_linear = linear.predict(X_validation)
    mlp_r2, mlp_rmse = _r2_rmse(prediction_mlp, Y_validation)
    return _ProbePredictions(
        linear=linear,
        validation_targets=Y_validation,
        target_std=target_std,
        linear_values=prediction_linear,
        mlp_values=prediction_mlp,
        mlp_r2=mlp_r2,
        mlp_rmse=mlp_rmse,
        binary=tuple(np.unique(Y[..., index]).size <= 2 for index in range(len(names))),
    )


def _head_result(predictions: _ProbePredictions, index: int, name: str) -> ProbeHeadResult:
    linear = predictions.linear
    if name in linear.degenerate:
        return ProbeHeadResult(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            degenerate=True,
        )
    positive_rate = linear_auc = mlp_auc = None
    if predictions.binary[index]:
        labels = predictions.validation_targets[:, index]
        positive_rate = float((labels > 0.5).mean())
        linear_auc = rank_auc(predictions.linear_values[:, index], labels)
        mlp_auc = rank_auc(predictions.mlp_values[:, index], labels)
    linear_r2 = float(linear.r2[name])
    mlp_value = float(predictions.mlp_r2[index])
    return ProbeHeadResult(
        linear_r2=linear_r2,
        mlp_r2=mlp_value,
        linear_rmse_over_std=float(linear.rmse_over_std[name]),
        mlp_rmse_over_std=float(predictions.mlp_rmse[index] / predictions.target_std[index]),
        mlp_minus_linear_r2=mlp_value - linear_r2,
        positive_rate=positive_rate,
        linear_auc=linear_auc,
        mlp_auc=mlp_auc,
    )


def compare_probe_heads(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    attribute_names: Sequence[str],
    episodes: EpisodeSplit | None = None,
    feature_window: int = 1,
    ridge: float = 1e-3,
    hidden: tuple[int, ...] = (256, 128),
    mlp_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> ProbeAudit:
    """Measure nonlinearity without changing the affine planner contract."""

    X, Y, names = _comparison_inputs(
        features,
        targets,
        attribute_names,
        feature_window=feature_window,
        ridge=ridge,
        hidden=hidden,
        mlp_epochs=mlp_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    split = episodes or split_episodes(X.shape[0], train_fraction=0.8, seed=seed)
    predictions = _fit_probe_predictions(
        X,
        Y,
        names,
        split,
        ridge=ridge,
        hidden=hidden,
        mlp_epochs=mlp_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    return ProbeAudit(
        attributes={
            name: _head_result(predictions, index, name) for index, name in enumerate(names)
        },
        train_episodes=split.train,
        validation_episodes=split.validation,
        feature_window=feature_window,
        hidden=hidden,
        mlp_epochs=mlp_epochs,
    )


__all__ = [
    "ProbeAudit",
    "ProbeHeadResult",
    "compare_probe_heads",
    "rank_auc",
    "stack_feature_history",
]
