"""Policy sharing across organisation agents during multi-agent RLlib training.

Sharing is not a matrix axis: it only decides which agents train one set of
weights. Agents sharing a policy must have identical spaces.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

SHARING_MODES = frozenset({"shared_all", "shared_by_role", "independent_per_agent"})


@dataclass(frozen=True)
class PolicySharingConfig:
    mode: str = "shared_all"

    def __post_init__(self) -> None:
        if self.mode not in SHARING_MODES:
            raise ValueError(f"policy_sharing must be one of {sorted(SHARING_MODES)}")

    def policy_id_for(self, agent_id: str) -> str:
        if self.mode == "shared_all":
            return "shared_policy"
        if self.mode == "independent_per_agent":
            return f"policy_{agent_id}"
        if agent_id == "mission_manager":
            return "manager_policy"
        if agent_id.startswith("sat_agent_"):
            return "satellite_policy"
        if agent_id.startswith("cluster_agent_"):
            return "cluster_policy"
        if agent_id == "central_agent":
            return "central_policy"
        return "shared_policy"

    def policy_ids(self, agent_ids: Iterable[str]) -> list[str]:
        return sorted({self.policy_id_for(agent_id) for agent_id in agent_ids})

    def mapping_fn(self) -> Callable[..., str]:
        def mapping(agent_id: str, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return self.policy_id_for(agent_id)

        return mapping


def build_policy_specs(
    agent_ids: Iterable[str],
    observation_spaces: Mapping[str, Any],
    action_spaces: Mapping[str, Any],
    sharing: PolicySharingConfig,
) -> dict[str, Any]:
    """RLlib PolicySpecs; agents sharing a policy must have compatible spaces."""

    from ray.rllib.policy.policy import PolicySpec

    agents = list(agent_ids)
    policies: dict[str, Any] = {}
    for policy_id in sharing.policy_ids(agents):
        members = [agent for agent in agents if sharing.policy_id_for(agent) == policy_id]
        first = members[0]
        for other in members[1:]:
            if not _compatible(observation_spaces[first], observation_spaces[other]) or not (
                _compatible(action_spaces[first], action_spaces[other])
            ):
                raise ValueError(
                    f"agents {first} and {other} cannot share policy {policy_id!r}: their "
                    "RL spaces differ; use policy_sharing independent_per_agent"
                )
        policies[policy_id] = PolicySpec(
            observation_space=observation_spaces[first],
            action_space=action_spaces[first],
        )
    return policies


def _compatible(left: Any, right: Any) -> bool:
    if type(left) is not type(right) or getattr(left, "shape", None) != getattr(
        right, "shape", None
    ):
        return False
    if hasattr(left, "nvec"):
        return bool(np.array_equal(left.nvec, right.nvec))
    return bool(
        np.array_equal(getattr(left, "low", None), getattr(right, "low", None))
        and np.array_equal(getattr(left, "high", None), getattr(right, "high", None))
    )


__all__ = ["SHARING_MODES", "PolicySharingConfig", "build_policy_specs"]
