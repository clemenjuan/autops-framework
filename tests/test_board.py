from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from test_wm_planner import _artifact as planner_fixture_artifact

from autops.board.generator import build_board, load_completed_run
from autops.config import expand_coordinate
from autops.core.provenance import scientific_config_sha256
from autops.core.runner import ExperimentRunner
from autops.memory.fixed import FixedMemory
from autops.missions.eventsat.metrics import experiment_statistics
from autops.paradigms.ao import AutonomousOnboard
from autops.representations.wm_planner import EventSatLeWMCEM


def result_payload(*, steps: int = 3, episode_steps: int = 3) -> dict:
    experiment = {
        "coordinate": "eventsat/sas/ag/symb",
        "organisation": "sas",
        "paradigm": "ag",
        "representation": "symb",
        "episodes": 1,
        "steps": steps,
    }
    digest = hashlib.sha256(
        json.dumps(experiment, sort_keys=True, default=str).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "experiment": experiment,
        "metric_registry": {"M-01": "utility", "M-09": "robustness_cv"},
        "metrics": {"M-01": 1.25, "M-09": 0.0},
        "statistics": {
            "mean": {"utility": 1.25, "robustness_cv": 0.0},
            "std": {"utility": 0.0, "robustness_cv": 0.0},
        },
        "episodes": [
            {
                "episode_id": 0,
                "steps": episode_steps,
                "metrics": {"utility": 1.25, "robustness_cv": 0.0},
            }
        ],
        "provenance": {
            "config_sha256": digest,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "python": "3.11.0",
            "dependencies": {},
        },
    }


def lewm_result_payload() -> dict:
    artifact_sha = "a" * 64
    trace_sha = "b" * 64
    checkpoint_sha = "c" * 64
    experiment = {
        "coordinate": "eventsat/sas/ao/lewm-cem",
        "organisation": "sas",
        "paradigm": "ao",
        "representation": "lewm-cem",
        "episodes": 2,
        "steps": 4,
        "mission_config": {"power": {"onboard_compute_w": 7.0}},
        "representation_config": {"mission_mode": "downlink", "samples": 999},
        "planner_artifact_identity": {
            "schema_version": "autops.lewm.planner/v3",
            "artifact_sha256": artifact_sha,
            "trace_sha256": trace_sha,
            "checkpoint_sha256": checkpoint_sha,
        },
    }

    def diagnostics(latency: float) -> dict:
        return {
            "mission_mode": "downlink",
            "planning_events": 2,
            "held_action_steps": 2,
            "reflex_overrides": 0,
            "cem_latency_total_s": latency,
            "cem_latency_mean_s": latency / 2,
            "evaluated_rollouts": 32,
            "rollouts_per_second": 32 / latency,
            "plan_hold": 2,
            "horizon": 4,
            "samples": 8,
            "elites": 2,
            "iterations": 2,
            "checkpoint_size_bytes": 2_000_000,
            "probe_rmse_over_std_mean": 0.3,
            "probe_rmse_over_std": {"science": 0.2, "downlink": 0.4, "degenerate": None},
            "artifact_identity": {
                "schema_version": "autops.lewm.planner/v3",
                "sha256": artifact_sha,
                "trace_sha256": trace_sha,
            },
            "checkpoint_identity": {
                "relative_path": "weights/lewm.ckpt",
                "sha256": checkpoint_sha,
            },
        }

    episodes = [
        {
            "episode_id": 0,
            "steps": 4,
            "planner_compute_energy_wh": 0.1,
            "metrics": {
                "utility": 1.0,
                "robustness_cv": 0.0,
                "final_battery_soc": 0.8,
                "total_energy_consumed_wh": 10.0,
            },
            "decision_diagnostics": {"onboard": diagnostics(0.4)},
        },
        {
            "episode_id": 1,
            "steps": 4,
            "planner_compute_energy_wh": 0.3,
            "metrics": {
                "utility": 2.0,
                "robustness_cv": 0.0,
                "final_battery_soc": 0.6,
                "total_energy_consumed_wh": 14.0,
            },
            "decision_diagnostics": {"onboard": diagnostics(0.8)},
        },
    ]
    statistics = experiment_statistics([episode["metrics"] for episode in episodes])
    return {
        "schema_version": 1,
        "experiment": experiment,
        "metric_registry": {"M-01": "utility", "M-09": "robustness_cv"},
        "metrics": {
            "M-01": statistics["mean"]["utility"],
            "M-09": statistics["mean"]["robustness_cv"],
        },
        "statistics": statistics,
        "episodes": episodes,
        "provenance": {
            "config_sha256": scientific_config_sha256(experiment),
            "source_revision": "d" * 40,
            "git_dirty": False,
            "python": "3.11.0",
            "dependencies": {},
        },
    }


def write_result(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_board_renders_only_evidenced_numbers(tmp_path: Path) -> None:
    source = write_result(tmp_path / "results" / "cell" / "results.json", result_payload())
    destination = build_board(source.parent.parent, tmp_path / "board" / "index.html")
    rendered = destination.read_text(encoding="utf-8")
    assert "eventsat/sas/ag/symb" in rendered
    assert "M-01" in rendered
    assert ">1<" in rendered
    assert "1.25" in rendered


def test_board_rejects_incomplete_episode(tmp_path: Path) -> None:
    source = write_result(tmp_path / "results.json", result_payload(episode_steps=2))
    with pytest.raises(ValueError, match="incomplete episode"):
        load_completed_run(source)


def test_board_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no completed results"):
        build_board(tmp_path, tmp_path / "index.html")


def test_board_rejects_presentation_only_metrics(tmp_path: Path) -> None:
    payload = result_payload()
    payload.pop("statistics")
    source = write_result(tmp_path / "results.json", payload)
    with pytest.raises(ValueError, match="statistics"):
        load_completed_run(source)


def test_board_rejects_metric_disagreement(tmp_path: Path) -> None:
    payload = result_payload()
    payload["metrics"]["M-01"] = 99.0
    source = write_result(tmp_path / "results.json", payload)
    with pytest.raises(ValueError, match="disagrees"):
        load_completed_run(source)


def test_board_rejects_dirty_source_revision(tmp_path: Path) -> None:
    payload = result_payload()
    payload["provenance"]["git_dirty"] = True
    source = write_result(tmp_path / "results.json", payload)
    with pytest.raises(ValueError, match="clean source"):
        load_completed_run(source)


def test_board_allows_distinct_configurations_of_one_coordinate(tmp_path: Path) -> None:
    first = result_payload()
    second = result_payload()
    second["experiment"]["representation_config"] = {"plan_hold": 12}
    second["provenance"]["config_sha256"] = hashlib.sha256(
        json.dumps(second["experiment"], sort_keys=True, default=str).encode()
    ).hexdigest()
    write_result(tmp_path / "results" / "first" / "results.json", first)
    write_result(tmp_path / "results" / "second" / "results.json", second)
    destination = build_board(tmp_path / "results", tmp_path / "index.html")
    rendered = destination.read_text(encoding="utf-8")
    assert first["provenance"]["config_sha256"][:8] in rendered
    assert second["provenance"]["config_sha256"][:8] in rendered


def test_board_rejects_duplicate_configuration(tmp_path: Path) -> None:
    payload = result_payload()
    write_result(tmp_path / "results" / "first" / "results.json", payload)
    write_result(tmp_path / "results" / "second" / "results.json", payload)
    with pytest.raises(ValueError, match="duplicate experiment configuration"):
        build_board(tmp_path / "results", tmp_path / "index.html")


def test_board_rejects_statistics_detached_from_episode_evidence(tmp_path: Path) -> None:
    payload = result_payload()
    payload["metrics"]["M-01"] = 99.0
    payload["statistics"]["mean"]["utility"] = 99.0
    source = write_result(tmp_path / "results.json", payload)
    with pytest.raises(ValueError, match="episode evidence"):
        load_completed_run(source)


def test_real_ssa_result_is_boardable(tmp_path: Path) -> None:
    spec = expand_coordinate(
        "ssa/sas/ao/symb", episodes=2, steps=2, seeds=[4, 5], constellation_size=1
    )
    payload = ExperimentRunner(spec, save=False).run()
    payload["provenance"].update(
        source_revision="a" * 40,
        source_kind="git",
        git_commit="a" * 40,
        git_dirty=False,
    )
    source = write_result(tmp_path / "ssa" / "results.json", payload)
    destination = build_board(source.parent, tmp_path / "index.html")
    assert "ssa/sas/ao/symb" in destination.read_text(encoding="utf-8")


def test_lewm_board_loads_actual_treatment_and_pooled_evidence(tmp_path: Path) -> None:
    source = write_result(tmp_path / "results.json", lewm_result_payload())
    run = load_completed_run(source)
    assert run.lewm is not None
    assert run.lewm.treatment.samples == 8
    assert run.lewm.treatment.onboard_compute_w == 7.0
    assert run.lewm.mean_final_battery_soc == pytest.approx(0.7)
    assert run.lewm.mean_total_energy_consumed_wh == pytest.approx(12.0)
    assert run.lewm.mean_planner_compute_energy_wh == pytest.approx(0.2)
    assert run.lewm.planning_duty_cycle == pytest.approx(0.5)
    assert run.lewm.mean_planning_events == pytest.approx(2.0)
    assert run.lewm.mean_cem_latency_s == pytest.approx(0.3)
    assert run.lewm.rollouts_per_second == pytest.approx(64 / 1.2)
    assert run.lewm.model_checkpoint_mb == pytest.approx(2.0)
    assert run.lewm.mean_normalized_probe_error == pytest.approx(0.3)

    rendered = build_board(source, tmp_path / "index.html").read_text(encoding="utf-8")
    assert "mode=downlink" in rendered
    assert "hold/H=2/4" in rendered
    assert "CEM=8/2x2" in rendered
    assert "artifact=aaaaaaaa" in rendered
    assert "trace=bbbbbbbb" in rendered
    assert "checkpoint=cccccccc" in rendered
    assert "config SHA-256:" in rendered


def test_board_rejects_missing_or_nonfinite_lewm_diagnostics(tmp_path: Path) -> None:
    missing = lewm_result_payload()
    missing["episodes"][0]["decision_diagnostics"]["onboard"].pop("planning_events")
    source = write_result(tmp_path / "missing.json", missing)
    with pytest.raises(ValueError, match="planning_events"):
        load_completed_run(source)

    nonfinite = lewm_result_payload()
    nonfinite["episodes"][0]["decision_diagnostics"]["onboard"]["cem_latency_total_s"] = float(
        "nan"
    )
    source = write_result(tmp_path / "nonfinite.json", nonfinite)
    with pytest.raises(ValueError, match="non-finite"):
        load_completed_run(source)


def test_board_rejects_inconsistent_lewm_semantics_and_formulae(tmp_path: Path) -> None:
    inconsistent = lewm_result_payload()
    inconsistent["episodes"][1]["decision_diagnostics"]["onboard"]["plan_hold"] = 1
    source = write_result(tmp_path / "identity.json", inconsistent)
    with pytest.raises(ValueError, match="inconsistent semantic identities"):
        load_completed_run(source)

    detached = lewm_result_payload()
    detached["episodes"][0]["decision_diagnostics"]["onboard"]["rollouts_per_second"] = 1.0
    source = write_result(tmp_path / "formula.json", detached)
    with pytest.raises(ValueError, match="throughput"):
        load_completed_run(source)


def test_board_rejects_detached_lewm_auxiliary_statistics(tmp_path: Path) -> None:
    payload = lewm_result_payload()
    payload["statistics"]["mean"]["final_battery_soc"] = 0.1
    source = write_result(tmp_path / "results.json", payload)
    with pytest.raises(ValueError, match="final_battery_soc disagrees"):
        load_completed_run(source)


def test_real_lewm_result_is_boardable(monkeypatch, tmp_path: Path) -> None:
    artifact = planner_fixture_artifact(plan_hold=1)

    def build_paradigm(self: ExperimentRunner) -> AutonomousOnboard:
        planner = EventSatLeWMCEM(
            {
                "artifact": artifact,
                "rollout_scorer": lambda history, sequences: np.zeros(sequences.shape[0]),
            }
        )
        return AutonomousOnboard(planner, FixedMemory())

    monkeypatch.setattr(ExperimentRunner, "_build_paradigm", build_paradigm)
    spec = expand_coordinate(
        "eventsat/sas/ao/lewm-cem",
        episodes=2,
        steps=2,
        seeds=[4, 5],
    )
    payload = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    payload["provenance"].update(
        source_revision="d" * 40,
        source_kind="git",
        git_commit="d" * 40,
        git_dirty=False,
    )
    source = write_result(tmp_path / "lewm" / "results.json", payload)
    run = load_completed_run(source)
    assert run.lewm is not None
    destination = build_board(source.parent, tmp_path / "index.html")
    assert "mode=science" in destination.read_text(encoding="utf-8")
