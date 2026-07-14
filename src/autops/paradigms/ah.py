"""Autonomous hybrid: fresh onboard core plus link-gated ground plan."""

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


class AutonomousHybrid(Paradigm):
    def __init__(
        self, onboard: Representation, ground: Representation, memory: FixedMemory
    ) -> None:
        super().__init__(memory)
        self.onboard = onboard
        self.ground = ground
        self._ground_view: dict[str, Any] = {}
        self._active: list[str] = []
        self._staged: list[str] | None = None
        self._was_contact = False

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        super().reset(seed, observation)
        self.onboard.reset(seed)
        self.ground.reset(seed + 1)
        self._ground_view = observation
        self._active = []
        self._staged = None
        self._was_contact = False

    def act(self, observation: dict[str, Any], *, physical_contact: bool) -> ParadigmDecision:
        onboard_actions, latency = self._decide(self.onboard, observation, role="onboard")
        onboard_mode = sat_mode(onboard_actions)
        ground_latency = 0.0
        if physical_contact and not self._was_contact:
            self._ground_view = refresh_almanac(self._ground_view, observation)
            ground_output, ground_latency = self._decide(
                self.ground, self._ground_view, role="ground"
            )
            self._staged = expand_schedule(ground_output.get("schedule"))
        self._was_contact = physical_contact
        if physical_contact or not self._active:
            selected = onboard_mode
        else:
            planned = self._active.pop(0)
            selected = onboard_mode if onboard_mode in {"charging", "safe"} else planned
        return ParadigmDecision(
            mode_action(selected),
            latency,
            ground_latency_s=ground_latency,
            rationale=self.onboard.last_rationale,
        )

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        super().after_step(info, observation)
        if _link_established(info):
            self._ground_view = observation
            if self._staged is not None:
                self._active = self._staged
                self._staged = None
