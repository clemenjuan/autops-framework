"""Observation and action adapters between AUTOPS missions and RLlib.

Ported from the agentic framework: RLlib needs vector observations and Gymnasium
spaces, while missions publish satellite-keyed records and commands. An adapter
applies one organisation agent's observation and actuation scopes identically in
training and evaluation. Each mission registers one contract (``RLSpec``); a
checkpoint records its schema and is rejected under any other. Gymnasium is
imported only when a space is built.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from autops.missions.eventsat.observation import encode_vectors, observation_space
from autops.missions.ssa.observation import SSA_RL_OBSERVATIONS, ssa_rl_vector
from autops.missions.ssa.policy import SSA_MODES
from autops.organisations.base import AgentObservation
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS


@dataclass(frozen=True)
class RLSpec:
    """Stable per-satellite observation/action contract of one mission."""

    mission: str
    modes: tuple[str, ...]
    observation_names: tuple[str, ...]
    schema_id: str

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)


EVENTSAT_RL_SPEC = RLSpec(
    "eventsat",
    EVENTSAT_ACTIONS,
    EVENTSAT_OBSERVATIONS,
    # The onboard vector of trace v5; RL shares the world model's information boundary.
    "autops.eventsat.onboard-observation/v5",
)
SSA_RL_SPEC = RLSpec(
    "ssa",
    SSA_MODES,
    SSA_RL_OBSERVATIONS,
    # Agentic ssa_local_compact order on the lean SSA environment, plus WM custody inputs.
    "autops.ssa.local-observation/v1",
)
RL_SPECS: dict[str, RLSpec] = {"eventsat": EVENTSAT_RL_SPEC, "ssa": SSA_RL_SPEC}


def rl_spec(mission: str) -> RLSpec:
    try:
        return RL_SPECS[mission]
    except KeyError as exc:
        raise ValueError(f"no RL contract registered for mission {mission!r}") from exc


def ground_eventsat_mode(
    mode: str, *, battery_soc: float, health_status: str, battery_min_soc: float
) -> str:
    """Controller-visible EventSat safety shield shared by training and evaluation.

    Contact is an observation, not a command veto: a radio attempt without contact
    reaches the environment and pays its physical cost and failed-action penalty.
    """

    if health_status != "nominal":
        return "safe"
    if battery_soc < battery_min_soc and mode != "charging":
        return "charging"
    return mode


def ground_ssa_mode(
    mode: str,
    satellite: Mapping[str, Any],
    *,
    coordinated: bool,
    low_soc: float = 0.3,
    observe_soc: float = 0.6,
    storage_high: float = 0.7,
) -> str:
    """Controller-visible SSA shield of the agentic framework, on lean SSA telemetry.

    Communication needs a pass and records, observation needs battery and room for a
    batch, detection needs a pending batch, and ISL sharing needs something useful to
    share, a coordinated scope or reachable peer, and battery.
    """

    if satellite.get("health", "nominal") != "nominal":
        return "safe"
    soc = float(satellite.get("battery_soc", 0.5))
    if soc < low_soc and mode != "charging":
        return "charging"
    undelivered = int(satellite.get("undelivered_records", 0))
    admitted = {
        "communication": bool(satellite.get("ground_pass_active")) and undelivered > 0,
        "payload_observe": soc > observe_soc
        and float(satellite.get("storage_used_fraction", 0.0)) < storage_high
        and float(satellite.get("jetson_raw_mb", 0.0))
        + float(satellite.get("observation_size_mb", 0.0))
        <= float(satellite.get("jetson_capacity_mb", 0.0)),
        "payload_detect": int(satellite.get("unprocessed_batches", 0)) > 0,
        "isl_share": (undelivered > 0 or bool(satellite.get("known_objects")))
        and (coordinated or bool(satellite.get("has_isl_peer")))
        and soc > observe_soc,
    }
    if mode not in SSA_MODES:
        return "charging"
    return mode if admitted.get(mode, True) else "charging"


def scoped_record(observation: Any) -> Mapping[str, Any]:
    """The mission record inside an organisation view, or the record itself."""

    if isinstance(observation, AgentObservation):
        return observation.local_state["full_observation"]
    return observation if isinstance(observation, Mapping) else {}


def eventsat_vector(record: Mapping[str, Any], observe_ids: Sequence[str]) -> np.ndarray:
    """Onboard vector per observed satellite; an unseen satellite encodes as zeros."""

    satellites = record.get("satellites", {})
    parts = [
        encode_vectors(record)[0]
        if satellite_id in satellites
        else np.zeros(EVENTSAT_RL_SPEC.obs_dim, np.float32)
        for satellite_id in observe_ids
    ]
    return np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, np.float32)


class RLSpaceAdapter:
    """One agent's scoped observation encoder and action decoder."""

    spec: RLSpec

    def __init__(self, act_ids: Sequence[str], observe_ids: Sequence[str]) -> None:
        self.act_ids = list(act_ids)
        self.observe_ids = list(observe_ids)

    @property
    def action_dims(self) -> list[int]:
        return [len(self.spec.modes)] * len(self.act_ids) or [1]

    def encode_observation(self, observation: Any) -> np.ndarray:
        raise NotImplementedError

    def decode_action(self, action: Any) -> dict[str, dict[str, Any]]:
        values = np.asarray(action, dtype=int).reshape(-1)
        decoded: dict[str, dict[str, Any]] = {}
        for index, satellite_id in enumerate(self.act_ids):
            mode_index = int(values[index]) if values.size > index else 0
            mode_index = max(0, min(mode_index, len(self.spec.modes) - 1))
            decoded[satellite_id] = {"mode": self.spec.modes[mode_index]}
        return decoded

    def ground_decoded_action(
        self, action: dict[str, dict[str, Any]], observation: Any
    ) -> dict[str, dict[str, Any]]:
        """Adapters without a controller-visible shield keep the decoded action."""

        del observation
        return action

    def spaces(self, mission_config: Mapping[str, Any]) -> tuple[Any, Any]:
        """Gymnasium observation and action spaces; requires the rl extra."""

        from gymnasium import spaces

        low, high = self.observation_bounds(mission_config)
        return (
            spaces.Box(low=low, high=high, dtype=np.float32),
            spaces.MultiDiscrete(self.action_dims),
        )

    def observation_bounds(
        self, mission_config: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class EventSatSpaceAdapter(RLSpaceAdapter):
    """EventSat onboard vector per observed satellite and one mode head per command."""

    spec = EVENTSAT_RL_SPEC

    def observation_bounds(
        self, mission_config: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = observation_space(mission_config["power"])
        count = max(1, len(self.observe_ids))
        return (
            np.tile(np.asarray(bounds.low, np.float32), count),
            np.tile(np.asarray(bounds.high, np.float32), count),
        )

    def encode_observation(self, observation: Any) -> np.ndarray:
        return eventsat_vector(scoped_record(observation), self.observe_ids)

    def ground_decoded_action(
        self, action: dict[str, dict[str, Any]], observation: Any
    ) -> dict[str, dict[str, Any]]:
        satellites = scoped_record(observation).get("satellites", {})
        grounded: dict[str, dict[str, Any]] = {}
        for satellite_id, command in action.items():
            record = satellites.get(satellite_id)
            if record is None:
                grounded[satellite_id] = dict(command)
                continue
            metadata = record.get("metadata", {})
            grounded[satellite_id] = {
                **command,
                "mode": ground_eventsat_mode(
                    str(command["mode"]),
                    battery_soc=float(record.get("resources", {}).get("battery_soc", 0.5)),
                    health_status=str(metadata.get("health_status", "nominal")),
                    battery_min_soc=float(metadata.get("battery_min_soc", 0.20)),
                ),
            }
        return grounded


class SSASpaceAdapter(RLSpaceAdapter):
    """Local SSA vector per observed satellite and one six-mode head per command."""

    spec = SSA_RL_SPEC

    def observation_bounds(
        self, mission_config: Mapping[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        del mission_config
        size = self.spec.obs_dim * max(1, len(self.observe_ids))
        return np.zeros(size, np.float32), np.ones(size, np.float32)

    def encode_observation(self, observation: Any) -> np.ndarray:
        record = scoped_record(observation)
        parts = [ssa_rl_vector(record, satellite_id) for satellite_id in self.observe_ids]
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(1, np.float32)

    def ground_decoded_action(
        self, action: dict[str, dict[str, Any]], observation: Any
    ) -> dict[str, dict[str, Any]]:
        satellites = scoped_record(observation).get("satellites", {})
        coordinated = len(self.act_ids) > 1
        return {
            satellite_id: {
                **command,
                "mode": ground_ssa_mode(
                    str(command["mode"]), satellites[satellite_id], coordinated=coordinated
                ),
            }
            if satellite_id in satellites
            else dict(command)
            for satellite_id, command in action.items()
        }


ADAPTERS: dict[str, type[RLSpaceAdapter]] = {
    "eventsat": EventSatSpaceAdapter,
    "ssa": SSASpaceAdapter,
}


def make_space_adapter(
    mission: str, act_ids: Sequence[str], observe_ids: Sequence[str]
) -> RLSpaceAdapter:
    """The scoped adapter registered for ``mission``."""

    try:
        return ADAPTERS[mission](act_ids, observe_ids)
    except KeyError as exc:
        raise ValueError(f"no RL space adapter registered for mission {mission!r}") from exc


__all__ = [
    "EVENTSAT_RL_SPEC",
    "RL_SPECS",
    "SSA_RL_SPEC",
    "EventSatSpaceAdapter",
    "RLSpaceAdapter",
    "RLSpec",
    "SSASpaceAdapter",
    "eventsat_vector",
    "ground_eventsat_mode",
    "ground_ssa_mode",
    "make_space_adapter",
    "rl_spec",
    "scoped_record",
]
