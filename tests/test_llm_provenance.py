from __future__ import annotations

import json

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner


def test_runner_persists_redacted_llm_diagnostics() -> None:
    spec = expand_coordinate(
        "eventsat/sas/ag/llm-s",
        steps=5,
        overrides={"representation": {"llm_mock": True}},
    )
    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    diagnostics = result["episodes"][0]["decision_diagnostics"]["ground"]
    serialized = json.dumps(diagnostics)
    assert diagnostics["llm_provenance"]["mock"] is True
    assert "endpoint" not in serialized
    assert "prompt" not in serialized


def test_ao_llm_runner_records_held_actions_and_planner_compute() -> None:
    replay = json.dumps(
        {
            "mode": "charging",
            "schedule": [["charging", 2]],
            "rationale": "replay plan",
        }
    )
    spec = expand_coordinate(
        "eventsat/sas/ao/llm-s",
        steps=3,
        overrides={
            "mission": {"anomalies": {"probability_per_step": 0.0}},
            "representation": {"llm_replay": [replay], "plan_hold": 3},
        },
    )

    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    episode = result["episodes"][0]
    diagnostics = episode["decision_diagnostics"]["onboard"]

    assert spec.onboard_uses_jetson
    assert diagnostics["planning_events"] == 1
    assert diagnostics["held_action_steps"] == 2
    assert episode["planner_compute_energy_wh"] > 0.0
