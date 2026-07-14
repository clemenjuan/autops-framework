"""Command workflow for Paper-C linear-versus-nonlinear probe audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autops.config import asset_root
from autops.core.provenance import collect_provenance
from autops.core.workflows import _latent_features
from autops.wm.artifact import checkpoint_sha256
from autops.wm.audit import compare_probe_heads
from autops.wm.dataset import split_episodes
from autops.wm.probes import (
    DEFAULT_ATTRIBUTES,
    TARGET_DEFINITION_VERSION,
    build_eventsat_targets,
)
from autops.wm.schema import load_trace, trace_sha256
from autops.wm.training import load_checkpoint

AUDIT_SCHEMA_VERSION = "autops.probe-audit/v1"


@dataclass(frozen=True)
class _AuditSettings:
    features: str
    feature_window: int
    mlp_epochs: int
    hidden: tuple[int, ...]
    device: str
    seed: int
    ridge: float
    learning_rate: float
    weight_decay: float
    validation_episodes: int


def _require_checkpoint(checkpoint_path: str | Path | None) -> str | Path:
    if checkpoint_path is None:
        raise ValueError("probe audit requires a LeWM checkpoint for data identity")
    return checkpoint_path


def _load_probe_features(trace: Any, checkpoint_path: str | Path, settings: _AuditSettings) -> Any:
    model, checkpoint_contract = load_checkpoint(checkpoint_path, device=settings.device)
    checkpoint_contract.validate_trace(trace)
    if settings.features == "obs":
        return trace.obs
    if settings.features == "latents":
        return _latent_features(
            model,
            checkpoint_contract.normalizer.normalize_obs(trace.obs),
            settings.device,
        )
    raise ValueError("features must be 'latents' or 'obs'")


def _audit_split(trace: Any, settings: _AuditSettings) -> tuple[Any, int]:
    if settings.validation_episodes < 1:
        raise ValueError("validation_episodes must be positive")
    validation_count = min(settings.validation_episodes, trace.n_episodes - 1)
    split = split_episodes(
        trace.n_episodes,
        train_fraction=(trace.n_episodes - validation_count) / trace.n_episodes,
        seed=settings.seed,
    )
    return split, validation_count


def _compare_heads(
    trace: Any, probe_features: Any, audit_split: Any, settings: _AuditSettings
) -> Any:
    return compare_probe_heads(
        probe_features,
        build_eventsat_targets(trace),
        attribute_names=DEFAULT_ATTRIBUTES,
        feature_window=settings.feature_window,
        hidden=settings.hidden,
        mlp_epochs=settings.mlp_epochs,
        episodes=audit_split,
        ridge=settings.ridge,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        seed=settings.seed,
        device=settings.device,
    )


def _audit_config(settings: _AuditSettings, validation_count: int) -> dict[str, Any]:
    return {
        "features": settings.features,
        "feature_window": settings.feature_window,
        "hidden": list(settings.hidden),
        "mlp_epochs": settings.mlp_epochs,
        "device": settings.device,
        "seed": settings.seed,
        "ridge": settings.ridge,
        "learning_rate": settings.learning_rate,
        "weight_decay": settings.weight_decay,
        "validation_episodes": validation_count,
    }


def _audit_payload(
    trace: Any,
    checkpoint_path: str | Path,
    audit_config: dict[str, Any],
    audit: Any,
) -> dict[str, Any]:
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "target_definition_version": TARGET_DEFINITION_VERSION,
        "trace_sha256": trace_sha256(trace),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
        "config": audit_config,
        "provenance": collect_provenance(audit_config, asset_root()),
        **audit.to_dict(),
    }


def _write_audit(output: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["output"] = str(destination)


def audit_probe_decodability(
    trace_path: str | Path,
    *,
    checkpoint_path: str | Path | None = None,
    features: str = "latents",
    output: str | Path | None = None,
    feature_window: int = 1,
    mlp_epochs: int = 100,
    hidden: tuple[int, ...] = (256, 128),
    device: str = "cpu",
    seed: int = 3072,
    ridge: float = 1e-3,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_episodes: int = 3,
) -> dict[str, Any]:
    settings = _AuditSettings(
        features,
        feature_window,
        mlp_epochs,
        hidden,
        device,
        seed,
        ridge,
        learning_rate,
        weight_decay,
        validation_episodes,
    )
    trace = load_trace(trace_path)
    if trace.metadata.mission != "eventsat":
        raise ValueError("the certification probe audit currently targets EventSat")
    checkpoint = _require_checkpoint(checkpoint_path)
    probe_features = _load_probe_features(trace, checkpoint, settings)
    audit_split, validation_count = _audit_split(trace, settings)
    audit = _compare_heads(trace, probe_features, audit_split, settings)
    config = _audit_config(settings, validation_count)
    payload = _audit_payload(trace, checkpoint, config, audit)
    if output is not None:
        _write_audit(output, payload)
    return payload


__all__ = ["audit_probe_decodability"]
