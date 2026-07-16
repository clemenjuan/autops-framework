"""Executable EventSat candidate projection and contact guidance.

Every CEM candidate is projected through the authoritative atomic byte
transitions before either learned or analytical scoring. The projector mirrors
environment settling, progress, health, storage, and battery rules. It uses the
onboard contact/sunlight almanac but never propagates orbital dynamics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from autops.missions.eventsat.transitions import (
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
)
from autops.wm.schema import EVENTSAT_ACTIONS

_ACTION = {name: index for index, name in enumerate(EVENTSAT_ACTIONS)}


def _number(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def contact_capacities(state: Mapping[str, Any], horizon: int) -> np.ndarray:
    """Return link capacity per requested action without rolling orbital physics."""

    capacities = np.zeros(horizon, dtype=np.float64)
    step_s = max(1e-12, _number(state, "step_duration_s", 60.0))
    rate = max(0.0, _number(state, "downlink_rate_kbps", 50.0))
    scheduled = state.get("planning_contact_seconds")
    if isinstance(scheduled, (list, tuple, np.ndarray)):
        seconds = np.asarray(scheduled, dtype=np.float64).reshape(-1)
        count = min(horizon, seconds.size)
        capacities[:count] = np.maximum(0.0, seconds[:count]) * rate / 8000.0
        return capacities
    cache = state.get("_analytic_orbit_cache")
    if isinstance(cache, Mapping):
        first = int(_number(state, "timestep"))
        for offset in range(horizon):
            snapshot = cache.get(first + offset) or {}
            seconds = _number(snapshot, "contact_window_seconds")
            if seconds <= 0.0 and snapshot.get("ground_pass_active", False):
                seconds = step_s
            capacities[offset] = rate * max(0.0, seconds) / 8000.0
        if np.any(capacities):
            return capacities

    active = bool(state.get("physical_ground_pass_active", False)) or (
        _number(state, "contact_window_seconds") > 0.0
    )
    if active:
        remaining_s = _number(
            state,
            "remaining_pass_duration_s",
            _number(state, "remaining_pass_duration", 1.0) * step_s,
        )
        remaining_s = max(step_s, remaining_s)
        for offset in range(horizon):
            overlap_s = min(step_s, max(0.0, remaining_s - offset * step_s))
            capacities[offset] = rate * overlap_s / 8000.0

    time_to_pass = _number(state, "time_to_next_pass", float("inf"))
    if np.isfinite(time_to_pass) and time_to_pass > 0.0 and rate > 0.0:
        offset = int(np.ceil(time_to_pass))
        future_mb = max(
            0.0,
            _number(
                state,
                "future_pass_capacity_mb",
                _number(state, "achievable_downlink_mb", rate * step_s / 8000.0),
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
    soc = _number(state, "battery_soc", 0.5)
    minimum_soc = _number(state, "battery_min_soc", 0.20)
    if health != "nominal" or soc <= minimum_soc + 0.02:
        mask[_ACTION["safe"]] = True
        return mask

    obc = _number(state, "obc_data_mb")
    raw = _number(state, "jetson_raw_mb")
    compressed = _number(state, "jetson_compressed_mb")
    obc_capacity = max(0.0, _number(state, "storage_capacity_mb", 4096.0))
    jetson_capacity = max(0.0, _number(state, "jetson_capacity_mb", 249036.8))
    physical = bool(state.get("physical_ground_pass_active", False)) or (
        _number(state, "contact_window_seconds") > 0.0
    )
    estimated = bool(state.get("contact_window_active", state.get("ground_pass_active", False)))
    settling = max(0, int(_number(state, "settling_time_steps")))
    if future_contact_mb is None:
        time_to_pass = _number(state, "time_to_next_pass", float("inf"))
        precontact = 0.0 < time_to_pass <= settling
    else:
        contacts = np.flatnonzero(np.asarray(future_contact_mb) > 0.0)
        precontact = bool(contacts.size and int(contacts[0]) <= settling)
    mask[_ACTION["communication"]] = (
        (physical or estimated or precontact) and obc > 0.01 and soc >= comms_soc_floor
    )
    if soc < reserve_soc:
        return mask
    observation_mb = _number(state, "observation_size_mb", 9.41)
    mask[_ACTION["payload_observe"]] = raw + compressed + observation_mb <= (
        jetson_capacity + 1e-12
    )
    mask[_ACTION["payload_compress"]] = (
        _number(state, "uncompressed_observations") >= 1.0 and raw + 1e-12 >= observation_mb
    )
    mask[_ACTION["payload_detect"]] = (
        _number(state, "undetected_observations") >= 1.0
        and obc + _number(state, "detection_metadata_mb", 0.01) <= obc_capacity + 1e-12
    )
    mask[_ACTION["payload_send"]] = compressed > 0.01 and obc < obc_capacity - 1e-12
    return mask


@dataclass(frozen=True)
class CandidateProjection:
    """An executable candidate bank and its propagated terminal states."""

    sequences: np.ndarray
    terminal_states: tuple[dict[str, Any], ...]
    repair_counts: np.ndarray


def _transition_parameters(state: Mapping[str, Any]) -> PipelineParameters:
    return PipelineParameters(
        observation_size_mb=max(0.0, _number(state, "observation_size_mb", 9.41)),
        compression_ratio=max(1e-12, _number(state, "compression_ratio", 5.11)),
        jetson_capacity_mb=max(0.0, _number(state, "jetson_capacity_mb", 249036.8)),
        obc_capacity_mb=max(0.0, _number(state, "storage_capacity_mb", 4096.0)),
        detection_metadata_mb=max(0.0, _number(state, "detection_metadata_mb", 0.01)),
        jetson_to_obc_rate_kbps=max(0.0, _number(state, "jetson_to_obc_rate_kbps", 8000.0)),
        downlink_rate_kbps=max(0.0, _number(state, "downlink_rate_kbps", 50.0)),
        step_duration_s=max(1e-12, _number(state, "step_duration_s", 60.0)),
    )


def _planning_sunlight(state: Mapping[str, Any], horizon: int) -> np.ndarray:
    scheduled = state.get("planning_sunlight")
    default = bool(state.get("in_sunlight", False))
    if not isinstance(scheduled, (list, tuple, np.ndarray)):
        return np.full(horizon, default, dtype=bool)
    values = np.asarray(scheduled, dtype=bool).reshape(-1)
    if values.size >= horizon:
        return values[:horizon]
    return np.concatenate([values, np.full(horizon - values.size, default, dtype=bool)])


def _advance_battery(state: dict[str, Any], effective: int, sunlight: bool) -> None:
    power = state.get("planning_power")
    if not isinstance(power, Mapping):
        return
    consumption = power.get("consumption")
    mode = EVENTSAT_ACTIONS[effective]
    mode_power = consumption.get(mode) if isinstance(consumption, Mapping) else None
    if not isinstance(mode_power, Mapping):
        return
    phase = "sun_w" if sunlight else "eclipse_w"
    load_w = max(0.0, _number(mode_power, phase))
    solar_w = 0.0
    if sunlight:
        solar_w = max(0.0, _number(power, "generation_peak_w")) * max(
            0.0, _number(power, "panel_efficiency_factor", 1.0)
        )
    hours = max(1e-12, _number(state, "step_duration_s", 60.0)) / 3600.0
    delta_wh = (solar_w - load_w) * hours
    if delta_wh > 0.0:
        delta_wh *= min(1.0, max(0.0, _number(power, "charge_efficiency", 1.0)))
    capacity_wh = max(1e-12, _number(power, "battery_capacity_wh", 70.0))
    state["battery_soc"] = min(
        1.0, max(0.0, _number(state, "battery_soc", 0.5) + delta_wh / capacity_wh)
    )


def _fallback(mask: np.ndarray, state: Mapping[str, Any]) -> int:
    safe = _ACTION["safe"]
    if str(state.get("health_status", "nominal")) != "nominal" and mask[safe]:
        return safe
    charging = _ACTION["charging"]
    return charging if mask[charging] else int(np.flatnonzero(mask)[0])


def _resolved_action(state: dict[str, Any], requested: int, settling: int) -> int:
    transition = max(0, int(_number(state, "transition_steps_remaining")))
    if transition > 0:
        transition -= 1
        state["transition_steps_remaining"] = transition
        if transition == 0:
            state["previous_mode"] = EVENTSAT_ACTIONS[requested]
        return _ACTION["charging"]
    previous = _ACTION.get(
        str(state.get("previous_mode", state.get("current_mode", "charging"))),
        _ACTION["charging"],
    )
    maneuver = {_ACTION["payload_observe"], _ACTION["communication"]}
    if previous != requested and (previous in maneuver or requested in maneuver) and settling > 0:
        state["transition_steps_remaining"] = settling - 1
        if settling == 1:
            state["previous_mode"] = EVENTSAT_ACTIONS[requested]
        return _ACTION["charging"]
    state["previous_mode"] = EVENTSAT_ACTIONS[requested]
    return requested


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
    elif effective == _ACTION["payload_compress"]:
        state["compression_progress"] = int(_number(state, "compression_progress")) + 1
        required = max(1, int(np.ceil(_number(state, "compression_time_factor", 2.0))))
        if state["compression_progress"] >= required:
            transition = apply_compress(state, parameters)
            if transition.accepted:
                state["compression_progress"] = 0
    elif effective == _ACTION["payload_detect"]:
        state["detection_progress"] = int(_number(state, "detection_progress")) + 1
        required = max(1, int(np.ceil(_number(state, "detection_time_steps", 5.0))))
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


def project_executable_candidates(
    state: Mapping[str, Any],
    sequences: np.ndarray,
    *,
    reserve_soc: float,
    comms_soc_floor: float,
) -> CandidateProjection:
    """Propagate feasibility through every action of every candidate."""

    requested = np.asarray(sequences)
    if requested.ndim != 2 or not np.issubdtype(requested.dtype, np.integer):
        raise ValueError("candidate sequences must be a two-dimensional integer array")
    if np.any((requested < 0) | (requested >= len(EVENTSAT_ACTIONS))):
        raise ValueError("candidate sequences contain an invalid EventSat action")
    horizon = requested.shape[1]
    capacities = contact_capacities(state, horizon)
    rate = max(1e-12, _number(state, "downlink_rate_kbps", 50.0))
    contacts_s = capacities * 8000.0 / rate
    sunlight = _planning_sunlight(state, horizon)
    parameters = _transition_parameters(state)
    settling = max(0, int(_number(state, "settling_time_steps")))
    projected = requested.astype(np.int64, copy=True)
    repairs = np.zeros(requested.shape[0], dtype=np.int64)
    terminal: list[dict[str, Any]] = []
    for sample, row in enumerate(requested):
        simulation = dict(state)
        for offset, requested_value in enumerate(row):
            simulation["contact_window_seconds"] = float(contacts_s[offset])
            simulation["physical_ground_pass_active"] = contacts_s[offset] > 0.0
            mask = admissible_action_mask(
                simulation,
                reserve_soc=reserve_soc,
                comms_soc_floor=comms_soc_floor,
                future_contact_mb=capacities[offset:],
            )
            action = int(requested_value)
            if not mask[action]:
                action = _fallback(mask, simulation)
                repairs[sample] += 1
            projected[sample, offset] = action
            effective = _resolved_action(simulation, action, settling)
            _apply_projected_action(simulation, effective, parameters, float(contacts_s[offset]))
            _advance_battery(simulation, effective, bool(sunlight[offset]))
        terminal.append(simulation)
    return CandidateProjection(projected, tuple(terminal), repairs)


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
    settling = max(0, int(_number(state, "settling_time_steps")))
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


def seed_pipeline_candidate(
    state: Mapping[str, Any],
    sequences: np.ndarray,
    *,
    first_action_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Inject one scored proposal that can complete the byte pipeline."""

    values = np.asarray(sequences, dtype=np.int64).copy()
    if values.ndim != 2:
        raise ValueError("sequences must be a [samples, horizon] array")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return values
    horizon = values.shape[1]
    row = np.full(horizon, _ACTION["charging"], dtype=np.int64)
    cursor = 0
    settling = max(0, int(_number(state, "settling_time_steps")))
    ratio = max(1e-12, _number(state, "compression_ratio", 5.11))
    product_mb = max(0.0, _number(state, "observation_size_mb", 9.41)) / ratio
    raw_mb = _number(state, "jetson_raw_mb")
    compressed_mb = _number(state, "jetson_compressed_mb")
    obc_mb = _number(state, "obc_data_mb")
    staged_mb = obc_mb + compressed_mb + raw_mb / ratio
    remaining_mb = max(
        0.0,
        _number(state, "remaining_achievable_downlink_mb", float("inf")),
    )
    has_raw = raw_mb > 0.01 or _number(state, "uncompressed_observations") > 0.0
    produced_compressed = False

    if obc_mb <= 0.01 and compressed_mb <= 0.01:
        if not has_raw and staged_mb + product_mb <= remaining_mb + 1e-9:
            stop = min(horizon, cursor + settling + 1)
            row[cursor:stop] = _ACTION["payload_observe"]
            cursor = stop
            has_raw = True
        if has_raw and cursor < horizon:
            compression_steps = max(1, int(np.ceil(_number(state, "compression_time_factor", 2.0))))
            stop = min(horizon, cursor + settling + compression_steps)
            row[cursor:stop] = _ACTION["payload_compress"]
            produced_compressed = stop - cursor >= compression_steps
            cursor = stop
    if (compressed_mb > 0.01 or produced_compressed) and cursor < horizon:
        row[cursor] = _ACTION["payload_send"]

    contacts = np.flatnonzero(contact_capacities(state, horizon) > 0.0)
    if contacts.size and (obc_mb > 0.01 or compressed_mb > 0.01 or produced_compressed):
        first = int(contacts[0])
        pointing = max(0, first - settling)
        row[pointing:first] = _ACTION["communication"]
        row[contacts] = _ACTION["communication"]
        if compressed_mb > 0.01 and pointing > 0:
            row[pointing - 1] = _ACTION["payload_send"]

    if first_action_mask is not None:
        mask = np.asarray(first_action_mask, dtype=bool)
        if mask.shape != (len(EVENTSAT_ACTIONS),) or not np.any(mask):
            raise ValueError("first_action_mask must allow a canonical EventSat action")
        if not mask[row[0]]:
            charging = _ACTION["charging"]
            row[0] = charging if mask[charging] else int(np.flatnonzero(mask)[0])
    values[0] = row
    return values


def _pipeline_score(
    state: Mapping[str, Any],
    terminal: Mapping[str, Any],
    horizon: int,
    *,
    downlink_scale: float,
    downlink_reward: float,
    pass_stage_reward: float,
    undeliverable_penalty: float,
) -> float:
    ratio = max(1e-12, _number(state, "compression_ratio", 5.11))
    staged = (
        _number(terminal, "obc_data_mb")
        + _number(terminal, "jetson_compressed_mb")
        + _number(terminal, "jetson_raw_mb") / ratio
    )
    downlinked = max(
        0.0,
        _number(terminal, "data_downlinked_mb") - _number(state, "data_downlinked_mb"),
    )
    excess = 0.0
    if state.get("remaining_achievable_downlink_mb") is not None:
        remaining_after = max(
            0.0,
            _number(state, "remaining_achievable_downlink_mb") - downlinked,
        )
        excess = max(0.0, staged - remaining_after)
    period = max(1.0, _number(state, "orbital_period_steps", 94.0))
    time_after_horizon = max(
        0.0,
        _number(state, "time_to_next_pass", period) - horizon,
    )
    proximity = max(0.0, 1.0 - time_after_horizon / period)
    stage_bonus = (
        pass_stage_reward
        * downlink_scale
        * min(_number(terminal, "obc_data_mb"), 10.0)
        / 10.0
        * (0.25 + 0.75 * proximity)
    )
    return (
        downlink_reward * downlink_scale * downlinked + stage_bonus - undeliverable_penalty * excess
    )


def pipeline_scores(
    state: Mapping[str, Any],
    sequences: np.ndarray,
    *,
    downlink_weight: float,
    downlink_reward: float,
    pass_stage_reward: float,
    reference_weight: float,
    undeliverable_penalty: float,
    reserve_soc: float = 0.5,
    comms_soc_floor: float = 0.25,
    projection: CandidateProjection | None = None,
) -> np.ndarray:
    """Score the exact projected byte-pipeline effects of one candidate bank."""

    scores = np.zeros(sequences.shape[0], dtype=np.float64)
    if sequences.shape[1] == 0:
        return scores
    if projection is None:
        projection = project_executable_candidates(
            state, sequences, reserve_soc=reserve_soc, comms_soc_floor=comms_soc_floor
        )
    elif not np.array_equal(projection.sequences, np.asarray(sequences)):
        raise ValueError("pipeline score projection must match its executable candidate bank")
    downlink_scale = downlink_weight / reference_weight
    for sample, terminal in enumerate(projection.terminal_states):
        scores[sample] = _pipeline_score(
            state,
            terminal,
            sequences.shape[1],
            downlink_scale=downlink_scale,
            downlink_reward=downlink_reward,
            pass_stage_reward=pass_stage_reward,
            undeliverable_penalty=undeliverable_penalty,
        )
    return scores


__all__ = [
    "CandidateProjection",
    "admissible_action_mask",
    "contact_capacities",
    "guided_probabilities",
    "pipeline_scores",
    "project_executable_candidates",
    "seed_pipeline_candidate",
]
