"""Conventional ground: a link-gated, one-successful-pass-delayed plan."""

from __future__ import annotations

from typing import Any

from autops.core.plugin import Representation
from autops.memory.fixed import FixedMemory
from autops.paradigms.ag import _link_established
from autops.paradigms.base import (
    Paradigm,
    ParadigmDecision,
    expand_schedule,
    mode_action,
    refresh_almanac,
    sat_mode,
)


class ConventionalGround(Paradigm):
    def __init__(self, ground: Representation, memory: FixedMemory) -> None:
        super().__init__(memory)
        self.ground = ground
        self._ground_view: dict[str, Any] = {}
        self._active: list[str] = []
        self._planned: list[str] | None = None
        self._candidate: list[str] | None = None
        self._new_plan: list[str] | None = None
        self._contact_action = "communication"
        self._was_contact = False

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        super().reset(seed, observation)
        self.ground.reset(seed)
        self._ground_view = observation
        self._active = []
        self._planned = self._candidate = self._new_plan = None
        self._was_contact = False

    def act(self, observation: dict[str, Any], *, physical_contact: bool) -> ParadigmDecision:
        if physical_contact:
            if not self._was_contact:
                self._candidate = self._planned
                self._ground_view = refresh_almanac(self._ground_view, observation)
                output, latency = self._decide(self.ground, self._ground_view, role="ground")
                self._new_plan = expand_schedule(output.get("schedule"))
                self._contact_action = sat_mode(output)
                self._was_contact = True
                return ParadigmDecision(
                    mode_action(self._contact_action), latency, rationale=self.ground.last_rationale
                )
            return ParadigmDecision(mode_action(self._contact_action), inference_allowed=False)
        if self._was_contact:
            self._planned = self._new_plan
            self._new_plan = None
        self._was_contact = False
        mode = self._active.pop(0) if self._active else "charging"
        return ParadigmDecision(mode_action(mode), inference_allowed=False)

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        super().after_step(info, observation)
        if _link_established(info):
            self._ground_view = observation
            if self._candidate is not None:
                self._active = self._candidate
                self._candidate = None
