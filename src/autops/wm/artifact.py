"""Strict, relocatable LeWM planner artifact contract."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autops.wm._contract_io import (
    cem_config_from_dict as _cem_from_dict,
)
from autops.wm._contract_io import (
    exact_fields as _only,
)
from autops.wm._contract_io import (
    finite_vector as _finite_vector,
)
from autops.wm._contract_io import (
    metric_map as _metric_map,
)
from autops.wm._contract_io import (
    relative_bundle_path as _relative_checkpoint,
)
from autops.wm.cem import CEMConfig
from autops.wm.schema import (
    EVENTSAT_ACTIONS,
    EVENTSAT_OBSERVATIONS,
    SSA_ACTIONS,
    SSA_OBSERVATIONS,
)

ARTIFACT_SCHEMA_VERSION = "autops.lewm.planner/v3"


def checkpoint_sha256(path_like: str | Path) -> str:
    """Hash checkpoint bytes so probe weights cannot be paired with another model."""

    hasher = hashlib.sha256()
    with Path(path_like).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class ModelContract:
    checkpoint: str
    mission: str
    obs_dim: int
    action_dim: int
    embed_dim: int
    history: int
    observation_names: tuple[str, ...]
    action_names: tuple[str, ...]
    trace_sha256: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", _relative_checkpoint(self.checkpoint))
        object.__setattr__(self, "mission", str(self.mission))
        object.__setattr__(self, "observation_names", tuple(str(v) for v in self.observation_names))
        object.__setattr__(self, "action_names", tuple(str(v) for v in self.action_names))
        object.__setattr__(self, "trace_sha256", str(self.trace_sha256))
        object.__setattr__(self, "checkpoint_sha256", str(self.checkpoint_sha256))
        if self.mission not in {"eventsat", "ssa"}:
            raise ValueError(f"unsupported planner mission {self.mission!r}")
        if min(self.obs_dim, self.action_dim, self.embed_dim, self.history) <= 0:
            raise ValueError("model dimensions and history must be positive")
        if len(self.observation_names) != self.obs_dim or len(set(self.observation_names)) != len(
            self.observation_names
        ):
            raise ValueError(
                "observation_names must uniquely define the model observation dimension"
            )
        expected_observations = {
            "eventsat": EVENTSAT_OBSERVATIONS,
            "ssa": SSA_OBSERVATIONS,
        }[self.mission]
        if self.observation_names != expected_observations:
            raise ValueError("model observation semantics do not match its mission")
        if len(self.action_names) != self.action_dim or len(set(self.action_names)) != len(
            self.action_names
        ):
            raise ValueError("action_names must uniquely define the model action dimension")
        expected_actions = {"eventsat": EVENTSAT_ACTIONS, "ssa": SSA_ACTIONS}[self.mission]
        if self.action_names != expected_actions:
            raise ValueError("model action semantics do not match its mission")
        if len(self.trace_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.trace_sha256
        ):
            raise ValueError("trace_sha256 must be a lowercase SHA-256 digest")
        if len(self.checkpoint_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "mission": self.mission,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "embed_dim": self.embed_dim,
            "history": self.history,
            "observation_names": list(self.observation_names),
            "action_names": list(self.action_names),
            "trace_sha256": self.trace_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ModelContract:
        _only(
            payload,
            {
                "checkpoint",
                "mission",
                "obs_dim",
                "action_dim",
                "embed_dim",
                "history",
                "observation_names",
                "action_names",
                "trace_sha256",
                "checkpoint_sha256",
            },
            "model",
        )
        return cls(
            checkpoint=str(payload["checkpoint"]),
            mission=str(payload["mission"]),
            obs_dim=int(payload["obs_dim"]),
            action_dim=int(payload["action_dim"]),
            embed_dim=int(payload["embed_dim"]),
            history=int(payload["history"]),
            observation_names=tuple(payload["observation_names"]),
            action_names=tuple(payload["action_names"]),
            trace_sha256=str(payload["trace_sha256"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
        )


@dataclass(frozen=True)
class NormalizationContract:
    obs_mean: tuple[float, ...]
    obs_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("obs_mean", "obs_std", "action_mean", "action_std"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), name))
        if len(self.obs_mean) != len(self.obs_std):
            raise ValueError("observation normalization vectors must match")
        if len(self.action_mean) != len(self.action_std):
            raise ValueError("action normalization vectors must match")
        if any(value <= 0.0 for value in (*self.obs_std, *self.action_std)):
            raise ValueError("normalization standard deviations must be positive")

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "obs_mean": list(self.obs_mean),
            "obs_std": list(self.obs_std),
            "action_mean": list(self.action_mean),
            "action_std": list(self.action_std),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> NormalizationContract:
        fields = {"obs_mean", "obs_std", "action_mean", "action_std"}
        _only(payload, fields, "normalization")
        return cls(**{name: tuple(payload[name]) for name in fields})


@dataclass(frozen=True)
class ProbeContract:
    W: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    attribute_names: tuple[str, ...]
    target_mean: tuple[float, ...]
    target_std: tuple[float, ...]
    degenerate: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.W:
            raise ValueError("probe.W must be a non-empty matrix")
        matrix = tuple(_finite_vector(row, "probe.W row") for row in self.W)
        object.__setattr__(self, "W", matrix)
        for name in ("b", "target_mean", "target_std"):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), f"probe.{name}"))
        names = tuple(str(value) for value in self.attribute_names)
        object.__setattr__(self, "attribute_names", names)
        object.__setattr__(self, "degenerate", tuple(str(value) for value in self.degenerate))
        count = len(names)
        if not names or len(set(names)) != count:
            raise ValueError("probe attribute_names must be non-empty and unique")
        if len(matrix) != count or any(len(row) != len(matrix[0]) for row in matrix):
            raise ValueError("probe.W must have one equal-width row per attribute")
        if any(len(getattr(self, field)) != count for field in ("b", "target_mean", "target_std")):
            raise ValueError("probe vectors must have one value per attribute")
        if any(value <= 0.0 for value in self.target_std):
            raise ValueError("probe.target_std must be positive for safe scalarization")
        if not set(self.degenerate) <= set(names):
            raise ValueError("degenerate targets must be named probe attributes")

    @property
    def input_dim(self) -> int:
        return len(self.W[0])

    def to_dict(self) -> dict[str, Any]:
        return {
            "W": [list(row) for row in self.W],
            "b": list(self.b),
            "attribute_names": list(self.attribute_names),
            "target_mean": list(self.target_mean),
            "target_std": list(self.target_std),
            "degenerate": list(self.degenerate),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeContract:
        fields = {"W", "b", "attribute_names", "target_mean", "target_std", "degenerate"}
        _only(payload, fields, "probe")
        return cls(
            W=tuple(tuple(row) for row in payload["W"]),
            b=tuple(payload["b"]),
            attribute_names=tuple(payload["attribute_names"]),
            target_mean=tuple(payload["target_mean"]),
            target_std=tuple(payload["target_std"]),
            degenerate=tuple(payload.get("degenerate", ())),
        )


@dataclass(frozen=True)
class ProbeEvidenceContract:
    """Held-out probe quality and bundle size retained with planner weights."""

    attribute_names: tuple[str, ...]
    rmse: dict[str, float | None]
    rmse_over_std: dict[str, float | None]
    r2: dict[str, float | None]
    train_episodes: tuple[int, ...]
    validation_episodes: tuple[int, ...]
    ridge: float
    checkpoint_size_bytes: int

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.attribute_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("probe evidence attribute_names must be non-empty and unique")
        object.__setattr__(self, "attribute_names", names)
        for field_name in ("rmse", "rmse_over_std", "r2"):
            object.__setattr__(
                self,
                field_name,
                _metric_map(getattr(self, field_name), names, f"probe evidence {field_name}"),
            )
        train = tuple(int(value) for value in self.train_episodes)
        validation = tuple(int(value) for value in self.validation_episodes)
        if (
            not train
            or not validation
            or len(set(train)) != len(train)
            or len(set(validation)) != len(validation)
            or set(train) & set(validation)
        ):
            raise ValueError("probe evidence requires unique, disjoint non-empty episode splits")
        if min(*train, *validation) < 0:
            raise ValueError("probe evidence episode indices must be non-negative")
        object.__setattr__(self, "train_episodes", train)
        object.__setattr__(self, "validation_episodes", validation)
        if not math.isfinite(self.ridge) or self.ridge < 0.0:
            raise ValueError("probe evidence ridge must be finite and non-negative")
        if self.checkpoint_size_bytes <= 0:
            raise ValueError("checkpoint_size_bytes must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute_names": list(self.attribute_names),
            "rmse": self.rmse,
            "rmse_over_std": self.rmse_over_std,
            "r2": self.r2,
            "train_episodes": list(self.train_episodes),
            "validation_episodes": list(self.validation_episodes),
            "ridge": self.ridge,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProbeEvidenceContract:
        fields = {
            "attribute_names",
            "rmse",
            "rmse_over_std",
            "r2",
            "train_episodes",
            "validation_episodes",
            "ridge",
            "checkpoint_size_bytes",
        }
        _only(payload, fields, "probe evidence")
        missing = fields - set(payload)
        if missing:
            raise ValueError(f"missing probe evidence fields: {sorted(missing)}")
        return cls(
            attribute_names=tuple(payload["attribute_names"]),
            rmse=dict(payload["rmse"]),
            rmse_over_std=dict(payload["rmse_over_std"]),
            r2=dict(payload["r2"]),
            train_episodes=tuple(payload["train_episodes"]),
            validation_episodes=tuple(payload["validation_episodes"]),
            ridge=float(payload["ridge"]),
            checkpoint_size_bytes=int(payload["checkpoint_size_bytes"]),
        )


@dataclass(frozen=True)
class PlannerArtifact:
    model: ModelContract
    normalization: NormalizationContract
    probe: ProbeContract
    probe_evidence: ProbeEvidenceContract
    cem: CEMConfig = field(default_factory=CEMConfig)
    mode_weight_presets: dict[str, dict[str, float]] = field(default_factory=dict)
    normalize_attribute_scale: bool = True
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported planner artifact {self.schema_version!r}")
        if len(self.normalization.obs_mean) != self.model.obs_dim:
            raise ValueError("observation normalizer does not match model.obs_dim")
        if len(self.normalization.action_mean) != self.model.action_dim:
            raise ValueError("action normalizer does not match model.action_dim")
        if self.probe.input_dim != self.model.embed_dim:
            raise ValueError("probe input width must equal model.embed_dim")
        if self.probe_evidence.attribute_names != self.probe.attribute_names:
            raise ValueError("probe evidence attributes must match probe weights")
        for name in self.probe.attribute_names:
            if self.probe_evidence.rmse[name] is None:
                raise ValueError("probe RMSE evidence must be finite for every attribute")
            normalized = self.probe_evidence.rmse_over_std[name]
            r2 = self.probe_evidence.r2[name]
            if name in self.probe.degenerate:
                if normalized is not None or r2 is not None:
                    raise ValueError("degenerate probe evidence must use null normalized metrics")
            elif normalized is None:
                raise ValueError("non-degenerate normalized probe error must be finite")
        if self.cem.action_dim != self.model.action_dim:
            raise ValueError("CEM action_dim must equal model.action_dim")
        attributes = set(self.probe.attribute_names)
        normalized: dict[str, dict[str, float]] = {}
        for preset, weights in self.mode_weight_presets.items():
            if set(weights) - attributes:
                raise ValueError(f"preset {preset!r} names unknown probe attributes")
            values = {str(name): float(value) for name, value in weights.items()}
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"preset {preset!r} contains non-finite weights")
            normalized[str(preset)] = values
        object.__setattr__(self, "mode_weight_presets", normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model": self.model.to_dict(),
            "normalization": self.normalization.to_dict(),
            "probe": self.probe.to_dict(),
            "probe_evidence": self.probe_evidence.to_dict(),
            "cem": {
                name: getattr(self.cem, name)
                for name in (
                    "horizon",
                    "action_dim",
                    "samples",
                    "elites",
                    "iterations",
                    "alpha",
                    "min_probability",
                    "plan_hold",
                    "seed",
                )
            },
            "mode_weight_presets": self.mode_weight_presets,
            "normalize_attribute_scale": self.normalize_attribute_scale,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerArtifact:
        allowed = {
            "schema_version",
            "model",
            "normalization",
            "probe",
            "probe_evidence",
            "cem",
            "mode_weight_presets",
            "normalize_attribute_scale",
        }
        _only(payload, allowed, "planner artifact")
        return cls(
            schema_version=str(payload.get("schema_version", "")),
            model=ModelContract.from_dict(payload["model"]),
            normalization=NormalizationContract.from_dict(payload["normalization"]),
            probe=ProbeContract.from_dict(payload["probe"]),
            probe_evidence=ProbeEvidenceContract.from_dict(payload["probe_evidence"]),
            cem=_cem_from_dict(payload["cem"]),
            mode_weight_presets=dict(payload.get("mode_weight_presets", {})),
            normalize_attribute_scale=bool(payload.get("normalize_attribute_scale", True)),
        )


def artifact_sha256(artifact: PlannerArtifact) -> str:
    """Hash canonical artifact semantics independently of JSON formatting."""

    canonical = json.dumps(
        artifact.to_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def resolve_checkpoint(artifact_path: str | Path, artifact: PlannerArtifact) -> Path:
    root = Path(artifact_path).resolve().parent
    checkpoint = (root / artifact.model.checkpoint).resolve()
    if root not in checkpoint.parents:
        raise ValueError("checkpoint escapes the artifact bundle")
    return checkpoint


def save_artifact(
    path_like: str | Path,
    artifact: PlannerArtifact,
    *,
    checkpoint_source: str | Path | None = None,
) -> Path:
    """Write JSON and, when supplied, copy the checkpoint into its relative slot."""

    path = Path(path_like)
    if path.suffix != ".json":
        raise ValueError("planner artifacts must use the .json suffix")
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve_checkpoint(path, artifact)
    if checkpoint_source is not None:
        source = Path(checkpoint_source).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if checkpoint_sha256(source) != artifact.model.checkpoint_sha256:
            raise ValueError("checkpoint source SHA-256 does not match PlannerArtifact")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if source != checkpoint:
            shutil.copy2(source, checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"artifact checkpoint does not exist: {checkpoint}")
    if checkpoint_sha256(checkpoint) != artifact.model.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match PlannerArtifact")
    if checkpoint.stat().st_size != artifact.probe_evidence.checkpoint_size_bytes:
        raise ValueError("checkpoint size does not match PlannerArtifact evidence")
    path.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def load_artifact(path_like: str | Path) -> PlannerArtifact:
    """Load the exact contract and verify that its relocatable checkpoint exists."""

    path = Path(path_like)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("planner artifact root must be an object")
    artifact = PlannerArtifact.from_dict(payload)
    checkpoint = resolve_checkpoint(path, artifact)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"artifact checkpoint does not exist: {checkpoint}")
    if checkpoint_sha256(checkpoint) != artifact.model.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 does not match PlannerArtifact")
    if checkpoint.stat().st_size != artifact.probe_evidence.checkpoint_size_bytes:
        raise ValueError("checkpoint size does not match PlannerArtifact evidence")
    return artifact


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ModelContract",
    "NormalizationContract",
    "PlannerArtifact",
    "ProbeContract",
    "ProbeEvidenceContract",
    "artifact_sha256",
    "checkpoint_sha256",
    "load_artifact",
    "resolve_checkpoint",
    "save_artifact",
]
