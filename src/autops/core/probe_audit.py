"""P1 readout audit: frozen latents against controls on one episode split.

Every feature family is read by the same affine and MLP heads. Heads are fitted
on the checkpoint's training episodes, the episodes its encoder was trained on,
and scored on its validation episodes or, given a test trace, on untouched test
seeds. Scoring latents on encoder-training episodes would favour them over the
controls. The controls are the raw onboard record (stack frames with
``feature_window``), an untrained encoder of the same architecture, and elapsed
time alone, which bounds what an episode clock can explain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root
from autops.core.offline import evaluation_record, held_out, write_evidence
from autops.core.provenance import collect_provenance
from autops.core.workflows import _latent_features
from autops.wm.artifact import checkpoint_sha256
from autops.wm.audit import compare_probe_heads
from autops.wm.dataset import EpisodeSplit
from autops.wm.probes import (
    DEFAULT_ATTRIBUTES,
    TARGET_DEFINITION_VERSION,
    build_eventsat_targets,
)
from autops.wm.schema import TraceDataset, load_trace, trace_sha256
from autops.wm.training import CheckpointContract, load_checkpoint

AUDIT_SCHEMA_VERSION = "autops.probe-audit/v2"
FEATURE_FAMILIES = ("latents", "untrained", "obs", "elapsed")


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


def _features(
    trace: TraceDataset, model: Any, contract: CheckpointContract, settings: _AuditSettings
) -> np.ndarray:
    if settings.features == "obs":
        return trace.obs
    if settings.features == "elapsed":
        elapsed = np.arange(trace.n_steps, dtype=np.float32) / trace.n_steps
        return np.broadcast_to(elapsed[None, :, None], (trace.n_episodes, trace.n_steps, 1))
    if settings.features == "untrained":
        from autops.wm.jepa import build_vector_jepa, require_torch

        require_torch().manual_seed(settings.seed)
        model = build_vector_jepa(contract.model_config).to(settings.device)
    elif settings.features != "latents":
        raise ValueError(f"features must be one of {FEATURE_FAMILIES}")
    return _latent_features(model, contract.normalizer.normalize_obs(trace.obs), settings.device)


def _evaluation_data(
    trace: TraceDataset,
    test_trace: TraceDataset | None,
    model: Any,
    contract: CheckpointContract,
    settings: _AuditSettings,
) -> tuple[np.ndarray, np.ndarray, EpisodeSplit]:
    features = _features(trace, model, contract, settings)
    targets = build_eventsat_targets(trace)
    if test_trace is None:
        return features, targets, contract.episodes
    if test_trace.n_steps != trace.n_steps:
        raise ValueError("the test trace must have the training episode length")
    train = np.asarray(contract.episodes.train, dtype=np.int64)
    count = len(train)
    split = EpisodeSplit(
        train=tuple(range(count)),
        validation=tuple(range(count, count + test_trace.n_episodes)),
    )
    return (
        np.concatenate([features[train], _features(test_trace, model, contract, settings)]),
        np.concatenate([targets[train], build_eventsat_targets(test_trace)]),
        split,
    )


def _audit_config(settings: _AuditSettings) -> dict[str, Any]:
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
    }


def audit_probe_decodability(
    trace_path: str | Path,
    *,
    checkpoint_path: str | Path,
    test_trace_path: str | Path | None = None,
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
) -> dict[str, Any]:
    if features not in FEATURE_FAMILIES:
        raise ValueError(f"features must be one of {FEATURE_FAMILIES}")
    settings = _AuditSettings(
        features=features,
        feature_window=feature_window,
        mlp_epochs=mlp_epochs,
        hidden=hidden,
        device=device,
        seed=seed,
        ridge=ridge,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )
    trace = load_trace(trace_path)
    if trace.metadata.mission != "eventsat":
        raise ValueError("the probe audit currently targets EventSat")
    model, contract = load_checkpoint(checkpoint_path, device=device)
    contract.validate_trace(trace)
    evaluation, _ = held_out(trace, test_trace_path, contract)
    test_trace = None if test_trace_path is None else evaluation
    X, Y, split = _evaluation_data(trace, test_trace, model, contract, settings)
    audit = compare_probe_heads(
        X,
        Y,
        attribute_names=DEFAULT_ATTRIBUTES,
        episodes=split,
        feature_window=feature_window,
        hidden=hidden,
        mlp_epochs=mlp_epochs,
        ridge=ridge,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        device=device,
    )
    config = _audit_config(settings)
    payload = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "target_definition_version": TARGET_DEFINITION_VERSION,
        "trace_sha256": trace_sha256(trace),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "checkpoint_size_bytes": Path(checkpoint_path).stat().st_size,
        "evaluation": evaluation_record(test_trace),
        "config": config,
        "provenance": collect_provenance(config, asset_root()),
        **audit.to_dict(),
    }
    return write_evidence(output, payload)


__all__ = ["FEATURE_FAMILIES", "audit_probe_decodability"]
