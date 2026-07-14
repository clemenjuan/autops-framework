"""Autonomous-onboard paradigm: fresh closed-loop decisions every step."""

from autops.core.plugin import Representation
from autops.memory.fixed import FixedMemory
from autops.paradigms.base import Paradigm, ParadigmDecision


class AutonomousOnboard(Paradigm):
    def __init__(self, onboard: Representation, memory: FixedMemory) -> None:
        super().__init__(memory)
        self.onboard = onboard

    def reset(self, seed: int, observation: dict) -> None:
        super().reset(seed, observation)
        self.onboard.reset(seed)

    def act(self, observation: dict, *, physical_contact: bool) -> ParadigmDecision:
        actions, latency = self._decide(self.onboard, observation, role="onboard")
        return ParadigmDecision(actions, latency, rationale=self.onboard.last_rationale)
