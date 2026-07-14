"""SSA organisation controllers with physical-channel-scoped knowledge and commands."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from autops.core.types import DecisionContext
from autops.memory.fixed import FixedMemory
from autops.missions.ssa.policy import RuleBasedSSA
from autops.missions.ssa.topology import build_leader_hierarchy


def _pairs(observation: dict[str, Any]) -> set[tuple[str, str]]:
    raw = observation.get("global", {}).get("isl_feasible_pairs", [])
    return {
        tuple(sorted((str(pair[0]), str(pair[1]))))
        for pair in raw
        if isinstance(pair, (list, tuple)) and len(pair) == 2 and pair[0] != pair[1]
    }


def _linked(left: str, right: str, pairs: set[tuple[str, str]]) -> bool:
    return left == right or tuple(sorted((left, right))) in pairs


def scope_observation(observation: dict[str, Any], members: list[str]) -> dict[str, Any]:
    """Expose local satellite truth and channel facts, never metric-only global truth."""

    member_set = set(members)
    satellites = observation.get("satellites", {})
    global_state = observation.get("global", {})
    scoped_pairs = [
        list(pair)
        for pair in sorted(_pairs(observation))
        if pair[0] in member_set and pair[1] in member_set
    ]
    scoped_global: dict[str, Any] = {"isl_feasible_pairs": scoped_pairs}
    if "max_steps" in global_state:
        scoped_global["max_steps"] = global_state["max_steps"]
    passes = global_state.get("ground_pass_active", {})
    if isinstance(passes, dict):
        scoped_global["ground_pass_active"] = {
            satellite_id: bool(passes.get(satellite_id, False))
            for satellite_id in members
            if satellite_id in satellites
        }
    return {
        "step": observation.get("step", 0),
        "epoch_s": observation.get("epoch_s", 0.0),
        "satellites": {
            satellite_id: satellites[satellite_id]
            for satellite_id in members
            if satellite_id in satellites
        },
        "global": scoped_global,
        "tasks": [
            task for task in observation.get("tasks", []) if task.get("satellite_id") in member_set
        ],
    }


@dataclass
class OrganisationController:
    """Common lifecycle for organisation-specific symbolic AO allocation."""

    config: dict[str, Any] = field(default_factory=dict)
    policies: dict[str, RuleBasedSSA] = field(default_factory=dict, init=False)
    memories: dict[str, FixedMemory] = field(default_factory=dict, init=False)
    coordination_messages: int = field(default=0, init=False)
    command_staleness: dict[str, int] = field(default_factory=dict, init=False)

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        self.policies = {}
        self.memories = {}
        self.coordination_messages = 0
        self.command_staleness = {
            satellite_id: 0 for satellite_id in observation.get("satellites", {})
        }

    def _policy(self, agent_id: str, seed: int = 0) -> RuleBasedSSA:
        if agent_id not in self.policies:
            policy_config = dict(self.config.get("policy", {}))
            policy = RuleBasedSSA(policy_config)
            policy.reset(seed)
            self.policies[agent_id] = policy
            self.memories[agent_id] = FixedMemory()
        return self.policies[agent_id]

    def _plan(self, agent_id: str, observation: dict[str, Any]) -> dict[str, Any]:
        policy = self._policy(agent_id)
        encoded = policy.encode_observation(observation)
        return policy.select_action(
            DecisionContext(
                state=encoded,
                observation=observation,
                memory=self.memories[agent_id],
                step=int(observation.get("step", 0)),
                role="onboard",
                metadata={"organisation_agent": agent_id},
            )
        )

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        record = {"info": info, "observation": observation}
        for memory in self.memories.values():
            memory.record(record)

    def metrics(self) -> dict[str, float]:
        staleness = list(self.command_staleness.values())
        return {
            "coordination_messages": float(self.coordination_messages),
            "mean_command_staleness": sum(staleness) / len(staleness) if staleness else 0.0,
        }


class SingleAgent(OrganisationController):
    """One decision loop coordinates the complete constellation."""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self._plan("single_agent", observation)


class IndependentAgents(OrganisationController):
    """One strictly local decision loop per satellite, with no message channel."""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        actions: dict[str, Any] = {}
        for satellite_id in sorted(observation.get("satellites", {})):
            plan = self._plan(satellite_id, scope_observation(observation, [satellite_id]))
            actions[satellite_id] = plan.get(satellite_id, {"mode": "charging"})
        return actions


class DecentralisedAgents(OrganisationController):
    """Physical-neighbour peer views; each peer commands only its own satellite."""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        satellite_ids = sorted(observation.get("satellites", {}))
        pairs = _pairs(observation)
        actions: dict[str, Any] = {}
        directed_messages = 0
        for satellite_id in satellite_ids:
            neighbours = [
                other
                for other in satellite_ids
                if other != satellite_id and _linked(satellite_id, other, pairs)
            ]
            directed_messages += len(neighbours)
            view = scope_observation(observation, [satellite_id, *neighbours])
            plan = self._plan(satellite_id, view)
            actions[satellite_id] = plan.get(satellite_id, {"mode": "charging"})
        self.coordination_messages += directed_messages
        return actions


class _HeadControlled(OrganisationController):
    held_commands: dict[str, dict[str, Any]]

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        super().reset(seed, observation)
        self.held_commands = {
            satellite_id: {"mode": "charging"} for satellite_id in observation.get("satellites", {})
        }

    def _deliver_plan(
        self,
        head: str,
        members: list[str],
        plan: dict[str, Any],
        pairs: set[tuple[str, str]],
    ) -> dict[str, Any]:
        actions: dict[str, Any] = {}
        for satellite_id in members:
            deliverable = _linked(head, satellite_id, pairs)
            command = plan.get(satellite_id)
            if deliverable and isinstance(command, dict):
                self.held_commands[satellite_id] = deepcopy(command)
                self.command_staleness[satellite_id] = 0
                self.coordination_messages += int(satellite_id != head)
            else:
                self.command_staleness[satellite_id] += 1
            actions[satellite_id] = deepcopy(self.held_commands[satellite_id])
        return actions


class CentralisedAgents(_HeadControlled):
    """A sat-0-hosted manager plans and commands over its current physical star."""

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        satellite_ids = sorted(observation.get("satellites", {}))
        if not satellite_ids:
            return {}
        head = satellite_ids[0]
        pairs = _pairs(observation)
        visible = [
            satellite_id for satellite_id in satellite_ids if _linked(head, satellite_id, pairs)
        ]
        plan = self._plan("mission_manager", scope_observation(observation, visible))
        return self._deliver_plan(head, satellite_ids, plan, pairs)


class HierarchicalAgents(_HeadControlled):
    """Cluster heads coordinate within physical links; clusters stay independent."""

    hierarchy: list[list[list[str]]]
    clusters: list[list[str]]

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        super().reset(seed, observation)
        members = sorted(observation.get("satellites", {}))
        branching = max(1, int(self.config.get("branching_factor", 10)))
        self.hierarchy = build_leader_hierarchy(members, branching)
        if not self.hierarchy:
            self.clusters = []
        elif branching == 1 or len(self.hierarchy) == 1:
            self.clusters = self.hierarchy[0]
        else:
            self.clusters = self.hierarchy[1]

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        pairs = _pairs(observation)
        actions: dict[str, Any] = {}
        for index, members in enumerate(self.clusters):
            if not members:
                continue
            head = members[0]
            visible = [
                satellite_id for satellite_id in members if _linked(head, satellite_id, pairs)
            ]
            plan = self._plan(f"cluster_{index}", scope_observation(observation, visible))
            actions.update(self._deliver_plan(head, members, plan, pairs))
        return actions

    def metrics(self) -> dict[str, float]:
        return {
            **super().metrics(),
            "num_clusters": float(len(self.clusters)),
            "hierarchy_depth": float(max(0, len(self.hierarchy) - 1)),
        }


CONTROLLERS: dict[str, type[OrganisationController]] = {
    "sas": SingleAgent,
    "cmas": CentralisedAgents,
    "dmas": DecentralisedAgents,
    "hmas": HierarchicalAgents,
    "imas": IndependentAgents,
}


def create_organisation(token: str, config: dict[str, Any] | None = None) -> OrganisationController:
    try:
        controller = CONTROLLERS[token]
    except KeyError as exc:
        raise ValueError(f"unknown organisation {token!r}") from exc
    return controller(config or {})


__all__ = [
    "CentralisedAgents",
    "DecentralisedAgents",
    "HierarchicalAgents",
    "IndependentAgents",
    "OrganisationController",
    "SingleAgent",
    "create_organisation",
    "scope_observation",
]
