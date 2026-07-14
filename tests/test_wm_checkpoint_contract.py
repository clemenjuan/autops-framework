from __future__ import annotations

import numpy as np
import pytest

from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.probe_audit import audit_probe_decodability
from autops.core.workflows import fit_planner_artifact
from autops.wm.artifact import checkpoint_sha256, load_artifact
from autops.wm.jepa import LeWMConfig
from autops.wm.schema import load_trace, write_trace
from autops.wm.training import (
    TrainingConfig,
    load_checkpoint,
    save_checkpoint,
    train_lewm,
)


def test_artifact_and_latent_audit_reuse_exact_checkpoint_data_contract(tmp_path) -> None:
    pytest.importorskip("torch")
    trace_path = export_trace(
        expand_coordinate("eventsat/sas/ao/symb", episodes=4, steps=6, seeds=[3, 4, 5, 6]),
        tmp_path / "trace.npz",
        prefer_orekit=False,
    )
    trace = load_trace(trace_path)
    result = train_lewm(
        trace,
        model_config=LeWMConfig(
            obs_dim=25,
            action_dim=7,
            embed_dim=8,
            encoder_hidden_dim=8,
            predictor_depth=1,
            predictor_heads=1,
            predictor_head_dim=8,
            predictor_mlp_dim=16,
            projector_hidden_dim=16,
            dropout=0.0,
            sigreg_knots=3,
            sigreg_projections=4,
        ),
        training_config=TrainingConfig(
            max_steps=1,
            warmup_steps=0,
            batch_size=2,
            train_fraction=0.5,
            seed=17,
            validation_interval=1,
            validation_sample_size=4,
            train_loss_window=1,
        ),
    )
    checkpoint_path = save_checkpoint(tmp_path / "model.pt", result)
    _, contract = load_checkpoint(checkpoint_path)

    fitted = fit_planner_artifact(
        trace_path,
        checkpoint_path,
        tmp_path / "planner.json",
        seed=999,
    )
    artifact = load_artifact(fitted["artifact"])
    np.testing.assert_array_equal(artifact.normalization.obs_mean, contract.normalizer.obs_mean)
    assert fitted["probe"]["train_episodes"] == list(contract.episodes.train)
    assert artifact.cem.seed == 999
    assert artifact.model.checkpoint_sha256 == checkpoint_sha256(checkpoint_path)
    assert artifact.probe_evidence.checkpoint_size_bytes == checkpoint_path.stat().st_size
    assert artifact.probe_evidence.train_episodes == contract.episodes.train
    assert artifact.probe_evidence.validation_episodes == contract.episodes.validation
    assert artifact.probe_evidence.rmse == fitted["probe"]["rmse"]

    audit = audit_probe_decodability(
        trace_path,
        checkpoint_path=checkpoint_path,
        mlp_epochs=1,
        hidden=(4,),
        seed=123,
    )
    assert len(audit["validation_episodes"]) == 3
    assert set(audit["train_episodes"]).isdisjoint(audit["validation_episodes"])
    assert set(audit["train_episodes"] + audit["validation_episodes"]) == set(range(4))
    assert audit["schema_version"] == "autops.probe-audit/v1"
    assert audit["trace_sha256"] == contract.trace_sha256
    assert audit["checkpoint_sha256"] == checkpoint_sha256(checkpoint_path)
    assert audit["config"]["seed"] == 123

    changed = load_trace(trace_path)
    changed.reward[0, 0] += 1.0
    changed_path = write_trace(tmp_path / "changed.npz", changed)
    with pytest.raises(ValueError, match="SHA-256"):
        fit_planner_artifact(changed_path, checkpoint_path, tmp_path / "bad.json")
    with pytest.raises(ValueError, match="SHA-256"):
        audit_probe_decodability(
            changed_path,
            checkpoint_path=checkpoint_path,
            mlp_epochs=1,
            hidden=(4,),
        )
