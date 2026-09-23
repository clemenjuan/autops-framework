"""Per-satellite SSA reward for multi-agent reinforcement learning.

The agentic framework's collective-negative reward: each satellite's local term
(its failed-action and safe-mode penalties) is blended with a team term, and the
shared custody term is added exactly once per satellite (agentic fe3dd35). An
agent's reward is the sum over the satellites it commands. The environment's
scalar reward and every metric are unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from autops.missions.ssa.dynamics import custody_mission_term

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment

TEAM_REDUCERS = frozenset({"mean", "sum", "min"})


class SSARewardFunction:
    """``local_weight * local_i + team_weight * team(local) + custody term``."""

    def __init__(self, blend: Mapping[str, Any] | None = None) -> None:
        values = dict(blend or {})
        self.local_weight = float(values.get("local_weight", 1.0))
        self.team_weight = float(values.get("team_weight", 0.0))
        self.team_reducer = str(values.get("team_reducer", "mean"))
        if self.team_reducer not in TEAM_REDUCERS:
            raise ValueError(f"team_reducer must be one of {sorted(TEAM_REDUCERS)}")

    def satellite_rewards(
        self,
        env: SSAEnvironment,
        modes: Mapping[str, str],
        per_satellite: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, float]:
        config = env.config["ssa"]
        local = {
            satellite_id: -float(config["failed_action_penalty"])
            * float(bool(per_satellite[satellite_id].get("failure_reason")))
            - float(config["safe_penalty"]) * float(modes[satellite_id] == "safe")
            for satellite_id in modes
        }
        values = list(local.values())
        team = {"sum": sum, "min": min}.get(self.team_reducer, _mean)(values) if values else 0.0
        mission = custody_mission_term(env)
        return {
            satellite_id: self.local_weight * value + self.team_weight * team + mission
            for satellite_id, value in local.items()
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


__all__ = ["TEAM_REDUCERS", "SSARewardFunction"]
