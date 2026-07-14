from __future__ import annotations

from typing import Any

from autops.core.plugin import Representation
from autops.core.types import DecisionContext
from autops.memory.fixed import FixedMemory
from autops.paradigms.ah import AutonomousHybrid


class FixedRepresentation(Representation):
    def __init__(self, action: dict[str, Any]) -> None:
        super().__init__()
        self.action = action

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return observation

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        return self.action


def test_ah_preserves_onboard_planner_compute_metadata() -> None:
    onboard = FixedRepresentation(
        {"eventsat_0": {"mode": "payload_observe", "jetson_planned": True}}
    )
    ground = FixedRepresentation({"schedule": []})
    paradigm = AutonomousHybrid(onboard, ground, FixedMemory())
    observation = {"step": 0, "satellites": {"eventsat_0": {}}}
    paradigm.reset(2, observation)
    decision = paradigm.act(observation, physical_contact=False)
    assert decision.actions["eventsat_0"]["jetson_planned"] is True


def test_ah_preserves_compute_metadata_when_ground_plan_wins() -> None:
    onboard = FixedRepresentation(
        {"eventsat_0": {"mode": "payload_observe", "jetson_planned": False}}
    )
    ground = FixedRepresentation(
        {
            "eventsat_0": {"mode": "communication"},
            "schedule": [{"mode": "payload_send", "steps": 1}],
        }
    )
    paradigm = AutonomousHybrid(onboard, ground, FixedMemory())
    observation = {"step": 0, "satellites": {"eventsat_0": {}}}
    paradigm.reset(2, observation)
    paradigm.act(observation, physical_contact=True)
    paradigm.after_step(
        {"resolved_mode": "communication", "contact_seconds": 60.0},
        observation,
    )
    decision = paradigm.act({**observation, "step": 1}, physical_contact=False)
    assert decision.actions["eventsat_0"] == {
        "mode": "payload_send",
        "jetson_planned": False,
    }
