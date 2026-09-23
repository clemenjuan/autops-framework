"""Decision loops: one representation plugin and fixed memory per organisation agent.

The runner and the RLlib bridge share the organisation's ``distribute_observation``
and ``collect_actions``; here each agent's decision comes from the coordinate's
representation plugin instead of a learner.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autops.core.plugin import Representation, create_representation
from autops.core.types import DecisionContext
from autops.memory.fixed import FixedMemory
from autops.organisations.base import (
    AgentAction,
    AgentObservation,
    Organisation,
    validate_agent_satellite_mapping,
)
from autops.organisations.topologies import ORGANISATIONS

_LOOP_KEYS = frozenset({"representation", "policy"})


class DecisionLoops:
    """Organisation-scoped decision loops with the controller lifecycle of a runner."""

    def __init__(
        self,
        organisation: Organisation,
        *,
        mission: str,
        representation: str,
        policy_config: dict[str, Any] | None = None,
    ) -> None:
        self.organisation = organisation
        self.mission = mission
        self.representation = representation
        self.policy_config = dict(policy_config or {})
        self.policies: dict[str, Representation] = {}
        self.memories: dict[str, FixedMemory] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._seed = 0

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        self.organisation.initialize(list(observation.get("satellites", {})))
        validate_agent_satellite_mapping(self.organisation)
        self.policies, self.memories, self._pending = {}, {}, {}
        self._seed = seed

    def act(self, observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
        channel = self.organisation.channel(observation)
        views = self.organisation.distribute_observation(observation, channel)
        plans = {agent_id: self._plan(agent_id, view) for agent_id, view in views.items()}
        return self.organisation.collect_actions(plans, channel)

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        # Memories keep exactly what each loop saw; next-step truth is no free channel.
        del info, observation
        for agent_id, record in self._pending.items():
            self.memories[agent_id].record(record)
        self._pending.clear()

    def metrics(self) -> dict[str, float]:
        return self.organisation.metrics()

    def _plan(self, agent_id: str, view: AgentObservation) -> AgentAction:
        policy = self._policy(agent_id)
        scoped = view.local_state["full_observation"]
        action = policy.select_action(
            DecisionContext(
                state=policy.encode_observation(scoped),
                observation=scoped,
                memory=self.memories[agent_id],
                step=int(scoped.get("step", 0)),
                role="onboard",
                metadata={"organisation_agent": agent_id},
            )
        )
        self._pending[agent_id] = deepcopy({"observation": scoped, "action": action})
        return AgentAction(agent_id, action)

    def _policy(self, agent_id: str) -> Representation:
        if agent_id not in self.policies:
            config = {
                **self.policy_config,
                "act_ids": self.organisation.satellites_for_agent(agent_id),
                "observe_ids": self.organisation.observed_satellites_for_agent(agent_id),
            }
            policy = create_representation(self.mission, self.representation, "onboard", config)
            policy.reset(self._seed)
            self.policies[agent_id] = policy
            self.memories[agent_id] = FixedMemory()
        return self.policies[agent_id]


def create_organisation(
    token: str, config: dict[str, Any] | None = None, *, mission: str = "ssa"
) -> DecisionLoops:
    """Decision loops over the named organisation; unknown tokens fail closed."""

    try:
        organisation_type = ORGANISATIONS[token]
    except KeyError as exc:
        raise ValueError(f"unknown organisation {token!r}") from exc
    values = dict(config or {})
    organisation = organisation_type({k: v for k, v in values.items() if k not in _LOOP_KEYS})
    return DecisionLoops(
        organisation,
        mission=mission,
        representation=str(values.get("representation", "symb")),
        policy_config=dict(values.get("policy", {})),
    )


__all__ = ["DecisionLoops", "create_organisation"]
