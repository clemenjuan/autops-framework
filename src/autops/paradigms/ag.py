"""Autonomous-ground paradigm with link-gated schedule promotion."""

from __future__ import annotations

from typing import Any

from autops.core.plugin import Representation
from autops.memory.fixed import FixedMemory
from autops.paradigms.base import (
    Paradigm,
    ParadigmDecision,
    expand_schedule,
    mode_action,
    refresh_almanac,
    sat_mode,
)


class AutonomousGround(Paradigm):
    def __init__(self, ground: Representation, memory: FixedMemory) -> None:
        super().__init__(memory)
        self.ground = ground
        self._ground_view: dict[str, Any] = {}
        self._active: list[str] = []
        self._staged: list[str] | None = None
        self._contact_action = "communication"
        self._was_contact = False

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        super().reset(seed, observation)
        self.ground.reset(seed)
        self._ground_view = observation
        self._active = []
        self._staged = None
        self._was_contact = False

    def act(self, observation: dict[str, Any], *, physical_contact: bool) -> ParadigmDecision:
        if physical_contact:
            if not self._was_contact:
                self._ground_view = refresh_almanac(self._ground_view, observation)
                output, latency = self._decide(self.ground, self._ground_view, role="ground")
                self._staged = expand_schedule(output.get("schedule"))
                self._contact_action = sat_mode(output)
                self._was_contact = True
                return ParadigmDecision(
                    mode_action(self._contact_action),
                    latency,
                    inference_allowed=True,
                    rationale=self.ground.last_rationale,
                )
            return ParadigmDecision(
                mode_action(self._contact_action),
                inference_allowed=False,
                rationale=self.ground.last_rationale,
            )
        self._was_contact = False
        mode = self._active.pop(0) if self._active else "charging"
        return ParadigmDecision(mode_action(mode), inference_allowed=False)

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        super().after_step(info, observation)
        if _link_established(info):
            self._ground_view = observation
            if self._staged is not None:
                self._active = self._staged
                self._staged = None


def _link_established(info: dict[str, Any]) -> bool:
    return (
        info.get("resolved_mode") == "communication"
        and float(info.get("contact_seconds", 0.0)) > 0.0
    )
