"""Organisation contract: who decides, who sees what, and how commands reach satellites.

The contract is the agentic framework's organisation layer (Kim et al. 2025
taxonomy, as used there): each organisation declares its agents, each agent's
actuation and observation scope over satellites, and optionally the logical
agent graph that authorises inter-satellite links. Organisations own allocation
and information routing, never sensing, link, or target truth. Agents hosted on
another satellite reach it only over a live link (``autops.organisations.channel``);
the canonical physical gating can be relaxed to the agentic logical channel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from autops.organisations.channel import LINK_GATINGS, Channel, CommandDelivery


@dataclass
class AgentObservation:
    """One agent's view; ``local_state["full_observation"]`` is its scoped record."""

    agent_id: str
    local_state: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentAction:
    """One agent's commands, keyed by satellite id."""

    agent_id: str
    action: Any = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def satellite_index(satellite_id: str) -> int:
    """Constellation index of ``<prefix>_<index>`` satellite ids."""

    return int(satellite_id.rsplit("_", 1)[1])


def scope_observation(observation: Mapping[str, Any], members: Sequence[str]) -> dict[str, Any]:
    """Expose member satellites and channel facts, never metric-only global truth."""

    member_set = set(members)
    satellites = observation.get("satellites", {})
    global_state = observation.get("global", {})
    scoped_global: dict[str, Any] = {
        "isl_feasible_pairs": [
            pair
            for pair in global_state.get("isl_feasible_pairs", [])
            if pair[0] in member_set and pair[1] in member_set
        ]
    }
    if "max_steps" in global_state:
        scoped_global["max_steps"] = global_state["max_steps"]
    passes = global_state.get("ground_pass_active")
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


class Organisation(ABC):
    """Allocation of decisions and information over one constellation."""

    token = ""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.link_gating = str(self.config.get("link_gating", "physical"))
        if self.link_gating not in LINK_GATINGS:
            raise ValueError(f"link_gating must be one of {sorted(LINK_GATINGS)}")
        self.satellite_ids: list[str] = []
        self.delivery = CommandDelivery()
        self._authorized: set[tuple[str, str]] | None = None

    def initialize(self, satellite_ids: Sequence[str]) -> None:
        """Build agents over the constellation in index order; call at every reset."""

        self.satellite_ids = sorted(satellite_ids, key=satellite_index)
        self._build_agents()
        self.delivery.reset(self.satellite_ids)
        self._authorized = derive_authorized_satellite_links(self)

    @abstractmethod
    def _build_agents(self) -> None: ...

    @abstractmethod
    def get_agents(self) -> list[str]: ...

    @abstractmethod
    def satellites_for_agent(self, agent_id: str) -> list[str]:
        """Actuation scope; scopes form a disjoint cover of the constellation."""

    def observed_satellites_for_agent(self, agent_id: str) -> list[str]:
        """Largest observation scope; the channel may hide members at a given step."""

        return self.satellites_for_agent(agent_id)

    def host_satellite(self, agent_id: str) -> str | None:
        """Satellite that runs the agent, or None for an agent not bound to one."""

        return None

    def logical_communication_edges(self) -> set[tuple[str, str]] | None:
        """Authoritative directed agent links; None keeps every physical link usable."""

        return None

    def channel(self, observation: Mapping[str, Any]) -> Channel:
        return Channel.from_observation(observation, self.link_gating)

    def visible_satellites(self, agent_id: str, channel: Channel) -> list[str]:
        host = self.host_satellite(agent_id)
        return [
            satellite_id
            for satellite_id in self.observed_satellites_for_agent(agent_id)
            if host is None or channel.linked(host, satellite_id)
        ]

    def distribute_observation(
        self, observation: Mapping[str, Any], channel: Channel
    ) -> dict[str, AgentObservation]:
        """Give each agent the members it can currently see from its host."""

        result: dict[str, AgentObservation] = {}
        for agent_id in self.get_agents():
            visible = self.visible_satellites(agent_id, channel)
            view = scope_observation(observation, visible)
            self._annotate_peers(view, channel)
            self.delivery.count_telemetry(self._telemetry_messages(agent_id, visible))
            result[agent_id] = AgentObservation(
                agent_id=agent_id,
                local_state={"full_observation": view},
                metadata={"host_satellite": self.host_satellite(agent_id)},
            )
        return result

    def collect_actions(
        self, agent_actions: Mapping[str, AgentAction], channel: Channel
    ) -> dict[str, dict[str, Any]]:
        """Deliver each owner's command, or hold the last one when its link is down."""

        unknown = sorted(set(agent_actions) - set(self.get_agents()))
        if unknown:
            raise ValueError(f"{self.token}: unknown agents {unknown}")
        commands: dict[str, dict[str, Any]] = {}
        for agent_id in self.get_agents():
            proposal = agent_actions.get(agent_id)
            plan = proposal.action if proposal is not None else {}
            plan = plan if isinstance(plan, Mapping) else {}
            misrouted = sorted(set(plan) - set(self.satellite_ids))
            if misrouted:
                raise ValueError(f"{agent_id} commanded unknown satellites {misrouted}")
            host = self.host_satellite(agent_id)
            for satellite_id in self.satellites_for_agent(agent_id):
                commands[satellite_id] = self.delivery.deliver(
                    host, satellite_id, plan.get(satellite_id), channel
                )
        return commands

    def metrics(self) -> dict[str, float]:
        return self.delivery.metrics()

    def _telemetry_messages(self, agent_id: str, visible: Sequence[str]) -> int:
        """Counted peer telemetry; only peer views exchange records as messages."""

        return 0

    def _annotate_peers(self, view: dict[str, Any], channel: Channel) -> None:
        """Mark whether an authorised ISL peer is reachable now, when the mission has ISLs."""

        if "isl_feasible_pairs" not in view["global"] or not view["satellites"]:
            return
        satellites = view["satellites"]
        for satellite_id, record in list(satellites.items()):
            peers = self.authorized_destinations(satellite_id)
            satellites[satellite_id] = {
                **record,
                "has_isl_peer": any(channel.linked(satellite_id, peer) for peer in peers),
            }

    def authorized_destinations(self, satellite_id: str) -> list[str]:
        if self._authorized is None:
            return [other for other in self.satellite_ids if other != satellite_id]
        return [
            destination
            for source, destination in sorted(self._authorized)
            if source == satellite_id
        ]


def validate_agent_satellite_mapping(organisation: Organisation) -> None:
    """Actuation scopes must cover the constellation exactly once; views stay inside it."""

    known = set(organisation.satellite_ids)
    owners: dict[str, str] = {}
    for agent_id in organisation.get_agents():
        observed = set(organisation.observed_satellites_for_agent(agent_id))
        if not observed <= known:
            raise ValueError(f"{agent_id} observes unknown satellites {sorted(observed - known)}")
        for satellite_id in organisation.satellites_for_agent(agent_id):
            if satellite_id not in known:
                raise ValueError(f"{agent_id} commands unknown satellite {satellite_id!r}")
            if satellite_id in owners:
                raise ValueError(
                    f"{satellite_id!r} is commanded by {owners[satellite_id]} and {agent_id}"
                )
            owners[satellite_id] = agent_id
    missing = known - set(owners)
    if missing:
        raise ValueError(f"{organisation.token}: no agent commands {sorted(missing)}")


def derive_authorized_satellite_links(
    organisation: Organisation,
) -> set[tuple[str, str]] | None:
    """Map the logical agent graph to directed satellite links; None authorises all."""

    edges = organisation.logical_communication_edges()
    if edges is None:
        return None
    agents = set(organisation.get_agents())
    invalid = sorted(edge for edge in edges if not set(edge) <= agents or edge[0] == edge[1])
    if invalid:
        raise ValueError(f"{organisation.token}: invalid communication edges {invalid}")
    return {
        (source, destination)
        for left, right in edges
        for source in organisation.satellites_for_agent(left)
        for destination in organisation.satellites_for_agent(right)
        if source != destination
    }


def bind_communication_topology(organisation: Organisation, environment: Any) -> None:
    """Hand the organisation's authorised links to an environment that routes ISLs."""

    configure = getattr(environment, "configure_communication_links", None)
    if callable(configure):
        configure(deepcopy(derive_authorized_satellite_links(organisation)))


__all__ = [
    "AgentAction",
    "AgentObservation",
    "Organisation",
    "bind_communication_topology",
    "derive_authorized_satellite_links",
    "satellite_index",
    "scope_observation",
    "validate_agent_satellite_mapping",
]
