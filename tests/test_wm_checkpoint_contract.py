from __future__ import annotations

import numpy as np
import pytest

from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.probe_audit import FEATURE_FAMILIES, audit_probe_decodability
from autops.core.workflows import fit_planner_artifact
from autops.wm.artifact import checkpoint_sha256, load_artifact
from autops.wm.schema import load_trace, write_trace
from autops.wm.training import load_checkpoint, save_checkpoint


def test_artifact_and_latent_audit_reuse_exact_checkpoint_data_contract(
    tmp_path, tiny_lewm
) -> None:
    trace_path = export_trace(
        expand_coordinate("eventsat/sas/ao/symb", episodes=4, steps=6, seeds=[3, 4, 5, 6]),
        tmp_path / "trace.npz",
        prefer_orekit=False,
    )
    trace = load_trace(trace_path)
    result = tiny_lewm(trace)
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

    for features in FEATURE_FAMILIES:
        audit = audit_probe_decodability(
            trace_path,
            checkpoint_path=checkpoint_path,
            features=features,
            mlp_epochs=1,
            hidden=(4,),
            seed=123,
        )
        assert audit["train_episodes"] == list(contract.episodes.train)
        assert audit["validation_episodes"] == list(contract.episodes.validation)
        assert audit["evaluation"] == {"episodes": "checkpoint-validation"}
    assert audit["schema_version"] == "autops.probe-audit/v2"
    assert audit["trace_sha256"] == contract.trace_sha256
    assert audit["checkpoint_sha256"] == checkpoint_sha256(checkpoint_path)
    assert audit["config"]["seed"] == 123

    test_path = export_trace(
        expand_coordinate("eventsat/sas/ao/symb", episodes=2, steps=6, seeds=[8, 9]),
        tmp_path / "test.npz",
        prefer_orekit=False,
    )
    held_out = audit_probe_decodability(
        trace_path,
        checkpoint_path=checkpoint_path,
        test_trace_path=test_path,
        mlp_epochs=1,
        hidden=(4,),
    )
    assert held_out["evaluation"]["test_seeds"] == [8, 9]
    assert len(held_out["validation_episodes"]) == 2
    with pytest.raises(ValueError, match="test seeds also occur"):
        audit_probe_decodability(
            trace_path,
            checkpoint_path=checkpoint_path,
            test_trace_path=trace_path,
            mlp_epochs=1,
            hidden=(4,),
        )

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
