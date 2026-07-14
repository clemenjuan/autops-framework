"""Operational-paradigm interface and shared decision helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from autops.core.plugin import Representation
from autops.core.types import DecisionContext
from autops.memory.fixed import FixedMemory

_ALMANAC_KEYS = (
    "orbital_phase",
    "time_to_next_eclipse",
    "time_to_next_pass",
    "remaining_pass_duration",
    "remaining_pass_duration_s",
    "contact_window_seconds",
    "contact_window_active",
    "in_sunlight",
    "next_gap_steps",
    "following_gap_steps",
    "planning_gap_steps",
    "future_pass_capacity_mb",
    "achievable_downlink_mb",
    "remaining_achievable_downlink_mb",
)


@dataclass(frozen=True)
class ParadigmDecision:
    actions: dict[str, Any]
    latency_s: float = 0.0
    ground_latency_s: float = 0.0
    inference_allowed: bool = True
    rationale: str | None = None


class Paradigm:
    def __init__(self, memory: FixedMemory) -> None:
        self.memory = memory

    def reset(self, seed: int, observation: dict[str, Any]) -> None:
        self.memory.reset()

    def act(self, observation: dict[str, Any], *, physical_contact: bool) -> ParadigmDecision:
        raise NotImplementedError

    def after_step(self, info: dict[str, Any], observation: dict[str, Any]) -> None:
        self.memory.record({"info": info, "observation": observation})

    def _decide(
        self,
        representation: Representation,
        observation: dict[str, Any],
        *,
        role: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        state = representation.encode_observation(observation)
        started = perf_counter()
        action = representation.select_action(
            DecisionContext(
                state=state,
                observation=observation,
                memory=self.memory,
                step=int(observation.get("step", 0)),
                role=role,
                metadata=metadata or {},
            )
        )
        return action, perf_counter() - started


def expand_schedule(schedule: list[dict[str, Any]] | None) -> list[str]:
    modes: list[str] = []
    for block in schedule or []:
        mode = str(block.get("mode", "charging"))
        modes.extend([mode] * max(0, int(block.get("steps", 0))))
    return modes


def sat_mode(actions: dict[str, Any]) -> str:
    satellite = actions.get("eventsat_0", {}) if isinstance(actions, dict) else {}
    return str(satellite.get("mode", "charging")) if isinstance(satellite, dict) else "charging"


def mode_action(mode: str, **extra: Any) -> dict[str, Any]:
    return {"eventsat_0": {"mode": mode, **extra}}


def refresh_almanac(stale: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Refresh deterministic planning fields without leaking truth telemetry."""

    refreshed = dict(stale)
    for key in ("step", "epoch_s"):
        if key in current:
            refreshed[key] = current[key]

    stale_satellites = stale.get("satellites")
    current_satellites = current.get("satellites")
    if not isinstance(stale_satellites, dict) or not isinstance(current_satellites, dict):
        return refreshed
    stale_satellite = stale_satellites.get("eventsat_0")
    current_satellite = current_satellites.get("eventsat_0")
    if not isinstance(stale_satellite, dict) or not isinstance(current_satellite, dict):
        return refreshed
    stale_metadata = stale_satellite.get("metadata")
    current_metadata = current_satellite.get("metadata")
    if not isinstance(stale_metadata, dict) or not isinstance(current_metadata, dict):
        return refreshed

    metadata = dict(stale_metadata)
    for key in _ALMANAC_KEYS:
        if key in current_metadata:
            metadata[key] = current_metadata[key]
    satellite = {**stale_satellite, "metadata": metadata}
    refreshed["satellites"] = {**stale_satellites, "eventsat_0": satellite}
    return refreshed
