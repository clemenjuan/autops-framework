"""Executable EventSat candidate projection and contact guidance.

Every CEM candidate is projected through the authoritative atomic byte
transitions before either learned or analytical scoring. The projector mirrors
environment settling, progress, health, storage, and battery rules. It never
propagates orbital dynamics. With a privileged contact/sunlight almanac (the
analytical oracle), future contact is known. Without one (the onboard view),
only present station visibility is known, sunlight persists at its current
value, and communication may be commanded before visibility for prepointing;
the environment still gates every transfer on physical contact.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from autops.missions.attitude import settle_mode
from autops.missions.eventsat.physics import (
    advance_projected_battery,
    resolve_mode,
    safety_required,
)
from autops.missions.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
    record_number,
)
from autops.wm.schema import EVENTSAT_ACTIONS

_ACTION = {name: index for index, name in enumerate(EVENTSAT_ACTIONS)}


def has_contact_forecast(state: Mapping[str, Any]) -> bool:
    """Return whether the decision record carries privileged future contact timing."""

    return (
        isinstance(state.get("planning_contact_seconds"), (list, tuple, np.ndarray))
        or isinstance(state.get("_analytic_orbit_cache"), Mapping)
        or "time_to_next_pass" in state
    )


def _currently_visible(state: Mapping[str, Any]) -> bool:
    return (
        bool(state.get("physical_ground_pass_active", False))
        or bool(state.get("station_visible", False))
        or record_number(state, "contact_window_seconds") > 0.0
    )


def contact_capacities(state: Mapping[str, Any], horizon: int) -> np.ndarray:
    """Return link capacity per requested action without rolling orbital physics."""

    capacities = np.zeros(horizon, dtype=np.float64)
    step_s = max(1e-12, record_number(state, "step_duration_s", 60.0))
    rate = max(0.0, record_number(state, "downlink_rate_kbps", 50.0))
    scheduled = state.get("planning_contact_seconds")
    if isinstance(scheduled, (list, tuple, np.ndarray)):
        seconds = np.asarray(scheduled, dtype=np.float64).reshape(-1)
        if seconds.size < horizon:
            raise ValueError("contact forecast does not cover the requested planning horizon")
        capacities[:] = np.maximum(0.0, seconds[:horizon]) * rate / 8000.0
        return capacities
    cache = state.get("_analytic_orbit_cache")
    if isinstance(cache, Mapping):
        first = int(record_number(state, "timestep"))
        for offset in range(horizon):
            snapshot = cache.get(first + offset) or {}
            seconds = record_number(snapshot, "contact_window_seconds")
            if seconds <= 0.0 and snapshot.get("ground_pass_active", False):
                seconds = step_s
            capacities[offset] = rate * max(0.0, seconds) / 8000.0
        if np.any(capacities):
            return capacities

    if _currently_visible(state):
        remaining_s = record_number(
            state,
            "remaining_pass_duration_s",
            record_number(state, "remaining_pass_duration", 1.0) * step_s,
        )
        remaining_s = max(step_s, remaining_s)
        for offset in range(horizon):
            overlap_s = min(step_s, max(0.0, remaining_s - offset * step_s))
            capacities[offset] = rate * overlap_s / 8000.0

    time_to_pass = record_number(state, "time_to_next_pass", float("inf"))
    if np.isfinite(time_to_pass) and time_to_pass > 0.0 and rate > 0.0:
        offset = int(np.ceil(time_to_pass))
        future_mb = max(
            0.0,
            record_number(
                state,
                "future_pass_capacity_mb",
                record_number(state, "achievable_downlink_mb", rate * step_s / 8000.0),
            ),
        )
        future_s = future_mb * 8000.0 / rate
        while offset < horizon and future_s > 0.0:
            overlap_s = min(step_s, future_s)
            capacities[offset] = max(capacities[offset], rate * overlap_s / 8000.0)
            future_s -= overlap_s
            offset += 1
    return capacities


def admissible_action_mask(
    state: Mapping[str, Any],
    *,
    reserve_soc: float,
    comms_soc_floor: float,
    future_contact_mb: np.ndarray | None = None,
) -> np.ndarray:
    """Return mission-policy actions admissible from one projected state."""

    mask = np.zeros(len(EVENTSAT_ACTIONS), dtype=bool)
    mask[_ACTION["charging"]] = True
    health = str(state.get("health_status", "nominal"))
    soc = record_number(state, "battery_soc", 0.5)
    minimum_soc = record_number(state, "battery_min_soc", 0.20)
    if health != "nominal" or soc <= minimum_soc + 0.02:
        mask[_ACTION["charging"]] = False
        mask[_ACTION["safe"]] = True
        return mask

    obc = record_number(state, "obc_data_mb")
    raw = record_number(state, "jetson_raw_mb")
    compressed = record_number(state, "jetson_compressed_mb")
    obc_capacity = max(0.0, record_number(state, "storage_capacity_mb", 4096.0))
    jetson_capacity = max(0.0, record_number(state, "jetson_capacity_mb", 249036.8))
    physical = _currently_visible(state)
    estimated = bool(state.get("contact_window_active", False))
    settling = max(0, int(record_number(state, "settling_time_steps")))
    if future_contact_mb is None and not has_contact_forecast(state):
        precontact = True  # onboard view: prepointing permitted; truth gates transfer
    elif future_contact_mb is None:
        time_to_pass = record_number(state, "time_to_next_pass", float("inf"))
        precontact = 0.0 < time_to_pass <= settling
    else:
        contacts = np.flatnonzero(np.asarray(future_contact_mb) > 0.0)
        precontact = bool(contacts.size and int(contacts[0]) <= settling)
    mask[_ACTION["communication"]] = (
        (physical or estimated or precontact) and obc > 0.01 and soc >= comms_soc_floor
    )
    if soc < reserve_soc:
        return mask
    observation_mb = record_number(state, "observation_size_mb", 9.41)
    mask[_ACTION["payload_observe"]] = raw + compressed + observation_mb <= (
        jetson_capacity + 1e-12
    )
    mask[_ACTION["payload_compress"]] = (
        record_number(state, "uncompressed_observations") >= 1.0 and raw + 1e-12 >= observation_mb
    )
    mask[_ACTION["payload_detect"]] = (
        record_number(state, "undetected_observations") >= 1.0
        and obc + record_number(state, "detection_metadata_mb", 0.01) <= obc_capacity + 1e-12
    )
    mask[_ACTION["payload_send"]] = compressed > 0.01 and obc < obc_capacity - 1e-12
    return mask


@dataclass(frozen=True)
class CandidateProjection:
    """An executable candidate bank and its propagated terminal states."""

    sequences: np.ndarray
    terminal_states: tuple[dict[str, Any], ...]
    repair_counts: np.ndarray
    terminal_forced: np.ndarray


def _transition_parameters(state: Mapping[str, Any]) -> PipelineParameters:
    return PipelineParameters(
        observation_size_mb=max(0.0, record_number(state, "observation_size_mb", 9.41)),
        compression_ratio=max(1e-12, record_number(state, "compression_ratio", 5.11)),
        jetson_capacity_mb=max(0.0, record_number(state, "jetson_capacity_mb", 249036.8)),
        obc_capacity_mb=max(0.0, record_number(state, "storage_capacity_mb", 4096.0)),
        detection_metadata_mb=max(0.0, record_number(state, "detection_metadata_mb", 0.01)),
        jetson_to_obc_rate_kbps=max(0.0, record_number(state, "jetson_to_obc_rate_kbps", 8000.0)),
        downlink_rate_kbps=max(0.0, record_number(state, "downlink_rate_kbps", 50.0)),
        step_duration_s=max(1e-12, record_number(state, "step_duration_s", 60.0)),
    )


def _planning_sunlight(state: Mapping[str, Any], horizon: int) -> np.ndarray:
    scheduled = state.get("planning_sunlight")
    default = bool(state.get("in_sunlight", False))
    if not isinstance(scheduled, (list, tuple, np.ndarray)):
        return np.full(horizon, default, dtype=bool)
    values = np.asarray(scheduled, dtype=bool).reshape(-1)
    if values.size < horizon:
        raise ValueError("sunlight forecast does not cover the requested planning horizon")
    return values[:horizon]


def _fallback(mask: np.ndarray, state: Mapping[str, Any]) -> int:
    safe = _ACTION["safe"]
    if str(state.get("health_status", "nominal")) != "nominal" and mask[safe]:
        return safe
    charging = _ACTION["charging"]
    return charging if mask[charging] else int(np.flatnonzero(mask)[0])


def _resolved_action(state: dict[str, Any], requested: int, settling: int) -> int:
    battery_soc = record_number(state, "battery_soc", 0.5)
    mandatory_safe = safety_required(
        battery_soc=battery_soc,
        minimum_soc=record_number(state, "battery_min_soc", 0.2),
        anomaly_active=state.get("health_status", "nominal") != "nominal",
    )
    resolved = resolve_mode(
        EVENTSAT_ACTIONS[requested],
        mandatory_safe=mandatory_safe,
        battery_soc=battery_soc,
        constraints=state.get("mode_constraints", {}),
    )
    remaining = max(0, int(record_number(state, "transition_steps_remaining")))
    # As in the environment, a command dropped while settling is not an override.
    ignored = remaining > 0 and not mandatory_safe
    state["forced"] = resolved != EVENTSAT_ACTIONS[requested] and not ignored
    effective, previous, remaining, _ = settle_mode(
        resolved,
        str(state.get("previous_mode", state.get("current_mode", "charging"))),
        remaining,
        settling,
        set(state.get("attitude_maneuver_modes", ("payload_observe", "communication"))),
        mandatory_safe=mandatory_safe,
    )
    state["previous_mode"] = previous
    state["transition_steps_remaining"] = remaining
    return _ACTION[effective]


def _apply_projected_action(
    state: dict[str, Any], effective: int, parameters: PipelineParameters, contact_s: float
) -> None:
    previous_effective = str(state.get("current_mode", "charging"))
    if effective != _ACTION["payload_compress"] and previous_effective == "payload_compress":
        state["compression_progress"] = 0
    if effective != _ACTION["payload_detect"] and previous_effective == "payload_detect":
        state["detection_progress"] = 0
    transition = None
    if effective == _ACTION["payload_observe"]:
        transition = apply_observe(state, parameters)
    elif effective == _ACTION["payload_compress"] and record_number(
        state, "uncompressed_observations"
    ):
        state["compression_progress"] = int(record_number(state, "compression_progress")) + 1
        required = max(1, int(np.ceil(record_number(state, "compression_time_factor", 2.0))))
        if state["compression_progress"] >= required:
            transition = apply_compress(state, parameters)
            if transition.accepted:
                state["compression_progress"] = 0
    elif effective == _ACTION["payload_detect"] and record_number(state, "undetected_observations"):
        state["detection_progress"] = int(record_number(state, "detection_progress")) + 1
        required = max(1, int(np.ceil(record_number(state, "detection_time_steps", 5.0))))
        if state["detection_progress"] >= required:
            transition = apply_detect(state, parameters)
            if transition.accepted:
                state["detection_progress"] = 0
    elif effective == _ACTION["payload_send"]:
        transition = apply_can_transfer(state, parameters)
    elif effective == _ACTION["communication"] and contact_s > 0.0:
        transition = apply_downlink(state, parameters, contact_seconds=contact_s)
    if transition is not None and transition.accepted:
        state.update(transition.state)
    state["current_mode"] = EVENTSAT_ACTIONS[effective]


def _set_projected_contact(
    simulation: dict[str, Any], contacts_s: np.ndarray, offset: int, settling: int
) -> None:
    """Describe contact at a projected step; the record's present flags go stale."""

    active = bool(contacts_s[offset] > 0.0)
    simulation["contact_window_seconds"] = float(contacts_s[offset])
    simulation["physical_ground_pass_active"] = active
    simulation["station_visible"] = active
    simulation["contact_window_active"] = bool(
        np.any(contacts_s[offset : offset + settling + 1] > 0.0)
    )


@dataclass(frozen=True)
class _Horizon:
    """Contact, sunlight and transition inputs shared by every projected step."""

    capacities: np.ndarray
    contacts_s: np.ndarray
    sunlight: np.ndarray
    parameters: PipelineParameters
    settling: int


def _horizon(state: Mapping[str, Any], requested: np.ndarray) -> _Horizon:
    if requested.ndim != 2 or not np.issubdtype(requested.dtype, np.integer):
        raise ValueError("candidate sequences must be a two-dimensional integer array")
    if np.any((requested < 0) | (requested >= len(EVENTSAT_ACTIONS))):
        raise ValueError("candidate sequences contain an invalid EventSat action")
    horizon = requested.shape[1]
    settling = max(0, int(record_number(state, "settling_time_steps")))
    capacities = contact_capacities(state, horizon + settling + 1)
    rate = max(1e-12, record_number(state, "downlink_rate_kbps", 50.0))
    return _Horizon(
        capacities=capacities,
        contacts_s=capacities * 8000.0 / rate,
        sunlight=_planning_sunlight(state, horizon),
        parameters=_transition_parameters(state),
        settling=settling,
    )


def _execute(simulation: dict[str, Any], action: int, horizon: _Horizon, offset: int) -> None:
    """Advance a projected record through one requested command, as the environment would."""

    effective = _resolved_action(simulation, action, horizon.settling)
    contact_s = float(horizon.contacts_s[offset])
    _apply_projected_action(simulation, effective, horizon.parameters, contact_s)
    advance_projected_battery(
        simulation, EVENTSAT_ACTIONS[effective], bool(horizon.sunlight[offset])
    )


def project_executable_candidates(
    state: Mapping[str, Any],
    sequences: np.ndarray,
    *,
    reserve_soc: float,
    comms_soc_floor: float,
) -> CandidateProjection:
    """Propagate feasibility through every action of every candidate."""

    requested = np.asarray(sequences)
    horizon = _horizon(state, requested)
    forecast = has_contact_forecast(state)
    projected = requested.astype(np.int64, copy=True)
    repairs = np.zeros(requested.shape[0], dtype=np.int64)
    forced = np.zeros(requested.shape[0], dtype=bool)
    terminal: list[dict[str, Any]] = []
    for sample, row in enumerate(requested):
        simulation = dict(state)
        for offset, requested_value in enumerate(row):
            _set_projected_contact(simulation, horizon.contacts_s, offset, horizon.settling)
            mask = admissible_action_mask(
                simulation,
                reserve_soc=reserve_soc,
                comms_soc_floor=comms_soc_floor,
                future_contact_mb=horizon.capacities[offset:] if forecast else None,
            )
            action = int(requested_value)
            if not mask[action]:
                action = _fallback(mask, simulation)
                repairs[sample] += 1
            projected[sample, offset] = action
            _execute(simulation, action, horizon, offset)
            forced[sample] = simulation["forced"]
        _set_projected_contact(simulation, horizon.contacts_s, requested.shape[1], horizon.settling)
        terminal.append(simulation)
    return CandidateProjection(projected, tuple(terminal), repairs, forced)


def project_command_prefixes(state: Mapping[str, Any], commands: np.ndarray) -> CandidateProjection:
    """Forecast one logged command sequence without mission-policy repair.

    Terminal state ``k`` is the projected record after the first ``k + 1``
    commands. It evaluates the analytical projection as a forecaster of
    requested commands; planning uses ``project_executable_candidates``.
    """

    requested = np.asarray(commands).reshape(1, -1)
    horizon = _horizon(state, requested)
    simulation = dict(state)
    states: list[dict[str, Any]] = []
    forced: list[bool] = []
    for offset, action in enumerate(requested[0]):
        _set_projected_contact(simulation, horizon.contacts_s, offset, horizon.settling)
        _execute(simulation, int(action), horizon, offset)
        forced.append(bool(simulation["forced"]))
        _set_projected_contact(simulation, horizon.contacts_s, offset + 1, horizon.settling)
        states.append(dict(simulation))
    length = requested.shape[1]
    return CandidateProjection(
        np.repeat(requested, length, axis=0),
        tuple(states),
        np.zeros(length, dtype=np.int64),
        np.asarray(forced),
    )


def guided_probabilities(
    state: Mapping[str, Any],
    probabilities: np.ndarray,
    *,
    enabled: bool,
    strength: float,
) -> np.ndarray:
    """Bias proposals around a known pass; scoring can still reject the schedule."""

    if not enabled or strength <= 0.0:
        return probabilities
    contacts = np.flatnonzero(contact_capacities(state, probabilities.shape[0]) > 0.0)
    if contacts.size == 0:
        return probabilities
    targets = {int(offset): _ACTION["communication"] for offset in contacts}
    first = int(contacts[0])
    settling = max(0, int(record_number(state, "settling_time_steps")))
    pointing = max(0, first - settling)
    for offset in range(pointing, first):
        targets[offset] = _ACTION["communication"]
    if pointing > 0:
        targets[pointing - 1] = _ACTION["payload_send"]
    guided = np.asarray(probabilities, dtype=np.float64).copy()
    for offset, action in targets.items():
        guided[offset] *= 1.0 - strength
        guided[offset, action] += strength
    return guided / guided.sum(axis=1, keepdims=True)


__all__ = [
    "CandidateProjection",
    "admissible_action_mask",
    "contact_capacities",
    "guided_probabilities",
    "has_contact_forecast",
    "project_command_prefixes",
    "project_executable_candidates",
]
