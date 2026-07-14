from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from autops.commands import main
from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.workflows import evaluate_lewm_cem, fit_planner_artifact
from autops.wm import cem as cem_module
from autops.wm import evaluation as evaluation_module
from autops.wm.artifact import (
    artifact_sha256,
    load_artifact,
    resolve_checkpoint,
    save_artifact,
)
from autops.wm.cem import CEMConfig
from autops.wm.jepa import LeWMConfig
from autops.wm.schema import load_trace, write_trace
from autops.wm.training import TrainingConfig, save_checkpoint, train_lewm


@pytest.fixture(scope="module")
def evaluation_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    pytest.importorskip("torch")
    root = tmp_path_factory.mktemp("cem-evaluation")
    trace_path = export_trace(
        expand_coordinate("eventsat/sas/ao/symb", episodes=4, steps=7, seeds=[31, 32, 33, 34]),
        root / "trace.npz",
        prefer_orekit=False,
    )
    trace = load_trace(trace_path)
    trained = train_lewm(
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
    checkpoint = save_checkpoint(root / "model.pt", trained)
    fitted = fit_planner_artifact(trace_path, checkpoint, root / "planner.json", seed=23)
    artifact_path = Path(fitted["artifact"])
    artifact = load_artifact(artifact_path)
    smoke_cem = CEMConfig(
        horizon=2,
        action_dim=7,
        samples=8,
        elites=2,
        iterations=2,
        plan_hold=2,
        seed=23,
    )
    save_artifact(
        artifact_path,
        replace(artifact, cem=smoke_cem),
        checkpoint_source=resolve_checkpoint(artifact_path, artifact),
    )
    return trace_path, artifact_path


def test_offline_evaluation_invokes_canonical_cem_and_is_portable(
    evaluation_bundle: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_path, artifact_path = evaluation_bundle
    assert evaluation_module.categorical_cem is cem_module.categorical_cem
    canonical = cem_module.categorical_cem
    calls = 0

    def tracked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return canonical(*args, **kwargs)

    monkeypatch.setattr(evaluation_module, "categorical_cem", tracked)
    output = tmp_path / "evaluation.json"
    summary = evaluate_lewm_cem(
        trace_path,
        artifact_path,
        output,
        max_contexts=2,
        mission_mode="science",
    )

    assert calls == 2
    assert summary["contexts"] == 2
    payload = json.loads(output.read_text(encoding="utf-8"))
    serialized = json.dumps(payload)
    assert str(trace_path) not in serialized
    assert str(artifact_path) not in serialized
    assert payload["schema_version"] == "autops.lewm.cem-evaluation/v1"
    assert payload["config"]["cem"]["samples"] == 8
    assert payload["aggregate"]["cem_candidate_rollouts"] == 2 * 8 * 2
    assert set(payload["aggregate"]["recorded_attribute_rmse"]) == set(
        payload["contexts"][0]["planned_attributes"]
    )
    assert payload["contracts"]["cem_function"] == "autops.wm.cem.categorical_cem"
    assert payload["runtime_provenance"]["source_revision"]
    assert payload["hashes"]["artifact_sha256"] == artifact_sha256(load_artifact(artifact_path))
    assert {item["episode_id"] for item in payload["contexts"]} <= set(payload["held_out_episodes"])
    for context in payload["contexts"]:
        assert len(context["planned_actions"]) == 2
        assert len(context["recorded_actions"]) == 2
        assert set(context["scores"]) == {
            "planned_model",
            "recorded_policy_model",
            "recorded_policy_realized",
            "model_improvement",
            "recorded_model_error",
            "last_iteration_elite_mean",
            "last_iteration_elite_std",
        }


def test_evaluation_cli_writes_durable_evidence(
    evaluation_bundle: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    trace_path, artifact_path = evaluation_bundle
    output = tmp_path / "cli-evaluation.json"
    assert (
        main(
            [
                "train",
                "evaluate",
                str(trace_path),
                "--artifact",
                str(artifact_path),
                "--output",
                str(output),
                "--max-contexts",
                "1",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["evaluation"] == str(output)
    assert summary["contexts"] == 1
    assert json.loads(output.read_text(encoding="utf-8"))["aggregate"]["context_count"] == 1


def test_evaluation_rejects_changed_trace(
    evaluation_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    trace_path, artifact_path = evaluation_bundle
    changed = load_trace(trace_path)
    changed.reward[0, 0] += 1.0
    changed_path = write_trace(tmp_path / "changed.npz", changed)

    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_lewm_cem(changed_path, artifact_path, tmp_path / "bad.json", max_contexts=1)


def test_evaluation_rejects_replaced_checkpoint(
    evaluation_bundle: tuple[Path, Path], tmp_path: Path
) -> None:
    _, artifact_path = evaluation_bundle
    tampered_root = tmp_path / "tampered"
    tampered_root.mkdir()
    copied_artifact = tampered_root / artifact_path.name
    shutil.copy2(artifact_path, copied_artifact)
    artifact = load_artifact(artifact_path)
    copied_checkpoint = tampered_root / artifact.model.checkpoint
    copied_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    copied_checkpoint.write_bytes(b"replaced checkpoint bytes")

    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        evaluate_lewm_cem(evaluation_bundle[0], copied_artifact, tmp_path / "bad-checkpoint.json")
