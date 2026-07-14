"""Private validation and decoding helpers for world-model contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from autops.wm.cem import CEMConfig


def exact_fields(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    """Require an exact set of fields with a stable diagnostic."""

    unknown = set(payload) - allowed
    missing = allowed - set(payload)
    if unknown or missing:
        raise ValueError(
            f"invalid {context} fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def finite_vector(values: Sequence[Any], name: str) -> tuple[float, ...]:
    vector = tuple(float(value) for value in values)
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return vector


def relative_bundle_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError("checkpoint must be a relative path inside the artifact bundle")
    if path.name in {"", "."}:
        raise ValueError("checkpoint must name a file")
    return path.as_posix()


def metric_map(
    values: Mapping[str, Any], names: tuple[str, ...], context: str
) -> dict[str, float | None]:
    if set(values) != set(names):
        raise ValueError(f"{context} must have one value per probe attribute")
    result: dict[str, float | None] = {}
    for name in names:
        value = values[name]
        if value is None:
            result[name] = None
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{context}.{name} must be finite or null")
        result[name] = numeric
    return result


def cem_config_from_dict(payload: Mapping[str, Any]) -> CEMConfig:
    allowed = {
        "horizon",
        "action_dim",
        "samples",
        "elites",
        "iterations",
        "alpha",
        "min_probability",
        "plan_hold",
        "seed",
    }
    exact_fields(payload, allowed, "CEM")
    return CEMConfig(**payload)


def canonical_names(values: Any, field_name: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if not names or any(not name for name in names):
        raise ValueError(f"{field_name} must contain non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} must not contain duplicates")
    return names


def integer_array(values: Any, field_name: str) -> np.ndarray:
    """Preserve integer semantics instead of silently truncating floats."""

    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{field_name} must use an integer dtype")
    return raw.astype(np.int64, copy=False)


def _checkpoint_sections(
    payload: Mapping[str, Any], model_config_type: type[Any], training_config_type: type[Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    exact_fields(
        payload,
        {"schema_version", "model_config", "training_config", "data", "evidence"},
        "checkpoint contract",
    )
    model = payload["model_config"]
    training = payload["training_config"]
    data = payload["data"]
    evidence = payload["evidence"]
    if not all(isinstance(value, Mapping) for value in (model, training, data, evidence)):
        raise ValueError("checkpoint recipe, data, and evidence must be mappings")
    exact_fields(
        model, {item.name for item in fields(model_config_type)}, "checkpoint model config"
    )
    exact_fields(
        training,
        {item.name for item in fields(training_config_type)},
        "checkpoint training config",
    )
    exact_fields(
        data,
        {
            "trace_schema_version",
            "mission",
            "observation_names",
            "action_names",
            "trace_sha256",
            "n_episodes",
            "n_steps",
            "train_episodes",
            "validation_episodes",
            "normalizer",
        },
        "checkpoint data",
    )
    exact_fields(
        evidence,
        {
            "train_loss",
            "validation_loss",
            "best_validation_step",
            "best_validation_loss",
            "validation_history",
        },
        "training evidence",
    )
    return model, training, data, evidence


def _validation_history(evidence: Mapping[str, Any]) -> tuple[tuple[int, float], ...]:
    payload = evidence["validation_history"]
    if not isinstance(payload, list) or not payload:
        raise ValueError("checkpoint validation history must be a non-empty list")
    for entry in payload:
        if not isinstance(entry, Mapping):
            raise ValueError("checkpoint validation entries must be mappings")
        exact_fields(entry, {"step", "loss"}, "checkpoint validation entry")
    return tuple((int(entry["step"]), float(entry["loss"])) for entry in payload)


def _checkpoint_normalizer(data: Mapping[str, Any]) -> Any:
    from autops.wm.dataset import FeatureNormalizer

    payload = data["normalizer"]
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint normalizer must be a mapping")
    field_names = {"obs_mean", "obs_std", "action_mean", "action_std"}
    exact_fields(payload, field_names, "checkpoint normalizer")
    return FeatureNormalizer(
        **{name: np.asarray(payload[name], dtype=np.float32) for name in field_names}
    )


def checkpoint_contract_kwargs(
    payload: Mapping[str, Any],
    *,
    model_config_type: type[Any],
    training_config_type: type[Any],
) -> dict[str, Any]:
    """Decode a strict checkpoint payload into constructor keyword arguments."""

    from autops.wm.dataset import EpisodeSplit

    model, training, data, evidence = _checkpoint_sections(
        payload, model_config_type, training_config_type
    )
    return {
        "schema_version": str(payload["schema_version"]),
        "model_config": model_config_type(**dict(model)),
        "training_config": training_config_type(**dict(training)),
        "trace_schema_version": str(data["trace_schema_version"]),
        "mission": str(data["mission"]),
        "observation_names": tuple(data["observation_names"]),
        "action_names": tuple(data["action_names"]),
        "trace_sha256": str(data["trace_sha256"]),
        "n_episodes": int(data["n_episodes"]),
        "n_steps": int(data["n_steps"]),
        "episodes": EpisodeSplit(
            train=tuple(int(value) for value in data["train_episodes"]),
            validation=tuple(int(value) for value in data["validation_episodes"]),
        ),
        "normalizer": _checkpoint_normalizer(data),
        "train_loss": float(evidence["train_loss"]),
        "validation_loss": float(evidence["validation_loss"]),
        "best_validation_step": int(evidence["best_validation_step"]),
        "best_validation_loss": float(evidence["best_validation_loss"]),
        "validation_history": _validation_history(evidence),
    }
