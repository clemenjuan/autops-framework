"""The five organisation tokens over one constellation.

Agent ids follow the agentic framework (``central_agent``, ``sat_agent_i``,
``mission_manager``, ``cluster_agent_i``) so policy sharing by role applies.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from autops.organisations.base import Organisation


class SingleAgent(Organisation):
    """SAS: one agent, not bound to a satellite, sees and commands every satellite."""

    token = "sas"
    agent_id = "central_agent"

    def _build_agents(self) -> None:
        return None

    def get_agents(self) -> list[str]:
        return [self.agent_id]

    def satellites_for_agent(self, agent_id: str) -> list[str]:
        _require(agent_id == self.agent_id, self.token, agent_id)
        return list(self.satellite_ids)


class _PerSatellite(Organisation):
    """One agent per satellite, hosted on the satellite it commands."""

    def _build_agents(self) -> None:
        self._owners = {f"sat_agent_{index}": sat for index, sat in enumerate(self.satellite_ids)}

    def get_agents(self) -> list[str]:
        return list(self._owners)

    def satellites_for_agent(self, agent_id: str) -> list[str]:
        _require(agent_id in self._owners, self.token, agent_id)
        return [self._owners[agent_id]]

    def host_satellite(self, agent_id: str) -> str | None:
        return self.satellites_for_agent(agent_id)[0]


class IndependentAgents(_PerSatellite):
    """IMAS: strictly local agents with no message channel."""

    token = "imas"


class DecentralisedAgents(_PerSatellite):
    """DMAS: equal peers; knowledge moves only through physical ISL sharing.

    ``peer_view: linked`` additionally shows linked neighbours' records, the earlier
    idealised telemetry channel, counted as one message per neighbour and step.
    """

    token = "dmas"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.peer_view = str(self.config.get("peer_view", "local"))
        if self.peer_view not in {"local", "linked"}:
            raise ValueError("dmas peer_view must be local or linked")

    def observed_satellites_for_agent(self, agent_id: str) -> list[str]:
        own = self.satellites_for_agent(agent_id)
        if self.peer_view == "local":
            return own
        return own + [satellite for satellite in self.satellite_ids if satellite not in own]

    def logical_communication_edges(self) -> set[tuple[str, str]]:
        agents = self.get_agents()
        return {(left, right) for left in agents for right in agents if left != right}

    def _telemetry_messages(self, agent_id: str, visible: Sequence[str]) -> int:
        return max(0, len(visible) - 1)


class CentralisedAgents(Organisation):
    """CMAS: a manager hosted on the first satellite plans and commands over its links."""

    token = "cmas"
    agent_id = "mission_manager"

    def _build_agents(self) -> None:
        return None

    def get_agents(self) -> list[str]:
        return [self.agent_id]

    def satellites_for_agent(self, agent_id: str) -> list[str]:
        _require(agent_id == self.agent_id, self.token, agent_id)
        return list(self.satellite_ids)

    def host_satellite(self, agent_id: str) -> str | None:
        return self.satellite_ids[0] if self.satellite_ids else None


class HierarchicalAgents(Organisation):
    """HMAS: cluster heads coordinate their own cluster over links; clusters are independent.

    Clusters are contiguous in satellite index: an explicit ``clusters`` partition,
    ``num_clusters`` near-equal groups, or groups of ``branching_factor``.
    """

    token = "hmas"

    def _build_agents(self) -> None:
        members = list(self.satellite_ids)
        if "clusters" in self.config:
            self.clusters = [[members[int(i)] for i in group] for group in self.config["clusters"]]
            self.depth = 1
        elif "num_clusters" in self.config:
            count = max(1, min(int(self.config["num_clusters"]), len(members)))
            base, extra = divmod(len(members), count)
            bounds = [index * base + min(index, extra) for index in range(count + 1)]
            self.clusters = [members[bounds[i] : bounds[i + 1]] for i in range(count)]
            self.depth = 1
        else:
            branching = max(1, int(self.config.get("branching_factor", 10)))
            hierarchy = build_leader_hierarchy(members, branching)
            if not hierarchy:
                self.clusters = []
            elif branching == 1 or len(hierarchy) == 1:
                self.clusters = hierarchy[0]
            else:
                self.clusters = hierarchy[1]
            self.depth = max(0, len(hierarchy) - 1)
        self.clusters = [cluster for cluster in self.clusters if cluster]

    def get_agents(self) -> list[str]:
        return [f"cluster_agent_{index}" for index in range(len(self.clusters))]

    def satellites_for_agent(self, agent_id: str) -> list[str]:
        prefix, _, index = agent_id.rpartition("_")
        _require(
            prefix == "cluster_agent" and index.isdigit() and int(index) < len(self.clusters),
            self.token,
            agent_id,
        )
        return list(self.clusters[int(index)])

    def host_satellite(self, agent_id: str) -> str | None:
        return self.satellites_for_agent(agent_id)[0]

    def metrics(self) -> dict[str, float]:
        return {
            **super().metrics(),
            "num_clusters": float(len(self.clusters)),
            "hierarchy_depth": float(self.depth),
        }


ORGANISATIONS: dict[str, type[Organisation]] = {
    organisation.token: organisation
    for organisation in (
        SingleAgent,
        CentralisedAgents,
        DecentralisedAgents,
        HierarchicalAgents,
        IndependentAgents,
    )
}


def build_leader_hierarchy(
    members: list[str] | tuple[str, ...],
    branching_factor: int,
) -> list[list[list[str]]]:
    """Group members bottom-up, terminating even for unary branching."""

    if branching_factor < 1:
        raise ValueError("branching_factor must be at least one")
    current = list(members)
    if not current:
        return []
    leaves = [[member] for member in current]
    if branching_factor == 1:
        return [leaves]
    levels = [leaves]
    while len(current) > 1:
        groups = [
            current[index : index + branching_factor]
            for index in range(0, len(current), branching_factor)
        ]
        levels.append(groups)
        current = [group[0] for group in groups]
    return levels


def _require(condition: bool, token: str, agent_id: str) -> None:
    if not condition:
        raise ValueError(f"{token} has no agent {agent_id!r}")


__all__ = [
    "ORGANISATIONS",
    "CentralisedAgents",
    "DecentralisedAgents",
    "HierarchicalAgents",
    "IndependentAgents",
    "SingleAgent",
    "build_leader_hierarchy",
]
