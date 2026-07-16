from __future__ import annotations

from autops.config import expand_coordinate
from autops.core.provenance import scientific_config_sha256
from autops.core.runner import ExperimentRunner


def _episode(checkpoint_sha: str) -> dict:
    return {
        "decision_diagnostics": {
            "onboard": {
                "artifact_identity": {
                    "schema_version": "autops.lewm.planner/v3",
                    "trace_sha256": "b" * 64,
                },
                "checkpoint_identity": {"relative_path": "model.pt", "sha256": checkpoint_sha},
            }
        }
    }


def test_result_hash_binds_same_path_to_actual_planner_checkpoint() -> None:
    spec = expand_coordinate("eventsat/sas/ao/lewm-cem")
    runner = ExperimentRunner(spec, save=False, prefer_orekit=False)
    statistics = {"mean": {}, "std": {}}

    first = runner._result_document([_episode("a" * 64)], statistics)
    second = runner._result_document([_episode("c" * 64)], statistics)

    assert first["experiment"]["planner_artifact_identity"]["checkpoint_sha256"] == "a" * 64
    assert first["provenance"]["config_sha256"] == scientific_config_sha256(first["experiment"])
    assert first["provenance"]["config_sha256"] != second["provenance"]["config_sha256"]


def test_result_rejects_inconsistent_episode_artifacts() -> None:
    spec = expand_coordinate("eventsat/sas/ao/lewm-cem")
    runner = ExperimentRunner(spec, save=False, prefer_orekit=False)

    try:
        runner._result_document([_episode("a" * 64), _episode("c" * 64)], {"mean": {}, "std": {}})
    except ValueError as exc:
        assert "inconsistent planner artifacts" in str(exc)
    else:
        raise AssertionError("inconsistent LeWM artifact identities were accepted")


def test_analytical_result_binds_scorer_and_planner_artifact() -> None:
    spec = expand_coordinate("eventsat/sas/ao/analytical-cem")
    runner = ExperimentRunner(spec, save=False, prefer_orekit=False)
    episode = _episode("a" * 64)
    diagnostics = episode["decision_diagnostics"]["onboard"]
    diagnostics.update(
        {
            "scorer_kind": "analytical-terminal",
            "propagation_model": "orbit-almanac+eventsat-physics",
            "uses_checkpoint": False,
        }
    )

    result = runner._result_document([episode], {"mean": {}, "std": {}})
    identity = result["experiment"]["planner_artifact_identity"]

    assert identity["scorer_kind"] == "analytical-terminal"
    assert identity["uses_checkpoint"] is False
    assert result["experiment"]["representation"] == "analytical-cem"
