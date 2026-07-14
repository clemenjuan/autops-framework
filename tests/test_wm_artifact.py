from __future__ import annotations

import hashlib
import shutil

import pytest

from autops.wm.artifact import (
    ModelContract,
    NormalizationContract,
    PlannerArtifact,
    ProbeContract,
    ProbeEvidenceContract,
    load_artifact,
    resolve_checkpoint,
    save_artifact,
)
from autops.wm.cem import CEMConfig
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS


def _evidence(attributes: tuple[str, ...]) -> ProbeEvidenceContract:
    zeros = {name: 0.0 for name in attributes}
    return ProbeEvidenceContract(
        attribute_names=attributes,
        rmse=zeros,
        rmse_over_std=zeros,
        r2=zeros,
        train_episodes=(0,),
        validation_episodes=(1,),
        ridge=1e-3,
        checkpoint_size_bytes=len(b"checkpoint"),
    )


def _artifact() -> PlannerArtifact:
    attributes = ("battery_margin", "downlink_progress")
    return PlannerArtifact(
        model=ModelContract(
            checkpoint="weights/lewm.ckpt",
            mission="eventsat",
            obs_dim=25,
            action_dim=7,
            embed_dim=192,
            history=3,
            observation_names=EVENTSAT_OBSERVATIONS,
            action_names=EVENTSAT_ACTIONS,
            trace_sha256="0" * 64,
            checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        ),
        normalization=NormalizationContract(
            obs_mean=(0.0,) * 25,
            obs_std=(1.0,) * 25,
            action_mean=(0.0,) * 7,
            action_std=(1.0,) * 7,
        ),
        probe=ProbeContract(
            W=((0.0,) * 192, (0.0,) * 192),
            b=(0.0, 0.0),
            attribute_names=attributes,
            target_mean=(0.8, 12.0),
            target_std=(0.1, 8.0),
        ),
        probe_evidence=_evidence(attributes),
        cem=CEMConfig(),
        mode_weight_presets={"science": {"battery_margin": 0.4, "downlink_progress": 0.6}},
    )


def test_artifact_is_strict_relocatable_and_scale_normalized_by_default(tmp_path):
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    original = tmp_path / "original"
    path = save_artifact(original / "planner.json", _artifact(), checkpoint_source=source)

    relocated = tmp_path / "relocated"
    shutil.move(original, relocated)
    moved_path = relocated / path.name
    loaded = load_artifact(moved_path)

    assert loaded.normalize_attribute_scale is True
    assert loaded.probe.target_std == (0.1, 8.0)
    assert resolve_checkpoint(moved_path, loaded).read_bytes() == b"checkpoint"
    assert loaded.cem == CEMConfig()
    assert loaded.probe_evidence.checkpoint_size_bytes == len(b"checkpoint")
    assert loaded.probe_evidence.train_episodes == (0,)


def test_artifact_rejects_absolute_or_escaping_checkpoint_paths():
    with pytest.raises(ValueError, match="relative path"):
        ModelContract(
            "/tmp/model.ckpt",
            "eventsat",
            25,
            7,
            192,
            3,
            EVENTSAT_OBSERVATIONS,
            EVENTSAT_ACTIONS,
            "0" * 64,
            "0" * 64,
        )
    with pytest.raises(ValueError, match="relative path"):
        ModelContract(
            "../model.ckpt",
            "eventsat",
            25,
            7,
            192,
            3,
            EVENTSAT_OBSERVATIONS,
            EVENTSAT_ACTIONS,
            "0" * 64,
            "0" * 64,
        )


def test_artifact_rejects_replaced_checkpoint_bytes(tmp_path):
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"checkpoint")
    path = save_artifact(tmp_path / "planner.json", _artifact(), checkpoint_source=source)
    resolve_checkpoint(path, _artifact()).write_bytes(b"same-shape-different-weights")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        load_artifact(path)


def test_artifact_rejects_missing_or_invalid_target_scale():
    with pytest.raises(ValueError, match="target_std"):
        ProbeContract(
            W=((0.0,) * 192,),
            b=(0.0,),
            attribute_names=("battery_margin",),
            target_mean=(0.0,),
            target_std=(0.0,),
        )


def test_artifact_rejects_probe_or_cem_dimension_mismatch():
    base = _artifact()
    bad_probe = ProbeContract(
        W=((0.0,) * 16,),
        b=(0.0,),
        attribute_names=("battery_margin",),
        target_mean=(0.0,),
        target_std=(1.0,),
    )
    with pytest.raises(ValueError, match="embed_dim"):
        PlannerArtifact(
            base.model, base.normalization, bad_probe, _evidence(("battery_margin",)), base.cem
        )
    with pytest.raises(ValueError, match="action_dim"):
        PlannerArtifact(
            base.model,
            base.normalization,
            base.probe,
            base.probe_evidence,
            CEMConfig(action_dim=6),
        )
