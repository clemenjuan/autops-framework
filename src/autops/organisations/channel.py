"""Inter-satellite channel availability and link-gated command delivery.

``physical`` gating uses the ISL pairs the environment publishes for the current
step: an agent hosted on one satellite sees and commands another only over a live
link, and an unreachable satellite keeps executing its last received command.
``logical`` gating treats every link as available, as the agentic organisations do.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

LINK_GATINGS = frozenset({"physical", "logical"})
DEFAULT_COMMAND = {"mode": "charging"}


@dataclass(frozen=True)
class Channel:
    """Undirected links available this step; None means every link is available."""

    pairs: frozenset[tuple[str, str]] | None

    @classmethod
    def from_observation(cls, observation: Mapping[str, Any], link_gating: str) -> Channel:
        if link_gating == "logical":
            return cls(None)
        raw = observation.get("global", {}).get("isl_feasible_pairs", [])
        return cls(
            frozenset(
                (min(str(left), str(right)), max(str(left), str(right)))
                for left, right in raw
                if left != right
            )
        )

    def linked(self, left: str, right: str) -> bool:
        return (
            self.pairs is None
            or left == right
            or (min(left, right), max(left, right)) in (self.pairs)
        )


class CommandDelivery:
    """Held commands, their staleness, and coordination-message counts."""

    def __init__(self) -> None:
        self.held: dict[str, dict[str, Any]] = {}
        self.staleness: dict[str, int] = {}
        self.messages = 0

    def reset(self, satellite_ids: Sequence[str]) -> None:
        self.held = {satellite_id: dict(DEFAULT_COMMAND) for satellite_id in satellite_ids}
        self.staleness = {satellite_id: 0 for satellite_id in satellite_ids}
        self.messages = 0

    def deliver(
        self, host: str | None, satellite_id: str, command: Any, channel: Channel
    ) -> dict[str, Any]:
        reachable = host is None or channel.linked(host, satellite_id)
        if reachable and isinstance(command, Mapping):
            self.held[satellite_id] = deepcopy(dict(command))
            self.staleness[satellite_id] = 0
            self.messages += int(host is not None and host != satellite_id)
        else:
            self.staleness[satellite_id] += 1
        return deepcopy(self.held[satellite_id])

    def count_telemetry(self, messages: int) -> None:
        self.messages += messages

    def metrics(self) -> dict[str, float]:
        values = list(self.staleness.values())
        return {
            "coordination_messages": float(self.messages),
            "mean_command_staleness": sum(values) / len(values) if values else 0.0,
        }


__all__ = ["DEFAULT_COMMAND", "LINK_GATINGS", "Channel", "CommandDelivery"]
