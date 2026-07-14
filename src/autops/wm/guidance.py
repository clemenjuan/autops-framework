"""Lightweight EventSat contact guidance for learned world-model planning.

The helpers account for requested actions, byte-pipeline feasibility, and the
onboard contact almanac. They deliberately do not roll power, ADCS physics, or
orbital dynamics, so LeWM-CEM remains distinct from the analytical comparator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

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


@dataclass(frozen=True)
class _PipelineParameters:
    capacities: np.ndarray
    observation_mb: float
    ratio: float
    product_mb: float
    compression_steps: int
    send_mb: float
    obc_capacity: float
    settling: int
    maneuver: set[int]
    initial_previous: int
    downlink_scale: float
    has_remaining_capacity: bool
    downlink_reward: float
    pass_stage_reward: float
    undeliverable_penalty: float


@dataclass
class _PipelineState:
    raw: float
    compressed: float
    obc: float
    uncompressed: int
    progress: float
    transition: int
    previous: int
    downlinked: float = 0.0


def _pipeline_parameters(
    state: Mapping[str, Any],
    horizon: int,
    *,
    downlink_weight: float,
    downlink_reward: float,
    pass_stage_reward: float,
    reference_weight: float,
    undeliverable_penalty: float,
) -> _PipelineParameters:
    step_s = max(1e-12, _number(state, "step_duration_s", 60.0))
    observation_mb = max(0.0, _number(state, "observation_size_mb", 9.41))
    ratio = max(1e-12, _number(state, "compression_ratio", 5.11))
    initial_mode = str(state.get("previous_mode", state.get("current_mode", "charging")))
    return _PipelineParameters(
        capacities=contact_capacities(state, horizon),
        observation_mb=observation_mb,
        ratio=ratio,
        product_mb=observation_mb / ratio,
        compression_steps=max(1, int(np.ceil(_number(state, "compression_time_factor", 2.0)))),
        send_mb=max(0.0, _number(state, "jetson_to_obc_rate_kbps", 8000.0)) * step_s / 8000.0,
        obc_capacity=max(0.0, _number(state, "storage_capacity_mb", 4096.0)),
        settling=max(0, int(_number(state, "settling_time_steps"))),
        maneuver={_ACTION["payload_observe"], _ACTION["communication"]},
        initial_previous=_ACTION.get(initial_mode, _ACTION["charging"]),
        downlink_scale=downlink_weight / reference_weight,
        has_remaining_capacity=state.get("remaining_achievable_downlink_mb") is not None,
        downlink_reward=downlink_reward,
        pass_stage_reward=pass_stage_reward,
        undeliverable_penalty=undeliverable_penalty,
    )


def _pipeline_state(state: Mapping[str, Any], parameters: _PipelineParameters) -> _PipelineState:
    return _PipelineState(
        raw=_number(state, "jetson_raw_mb"),
        compressed=_number(state, "jetson_compressed_mb"),
        obc=_number(state, "obc_data_mb"),
        uncompressed=max(0, int(_number(state, "uncompressed_observations"))),
        progress=max(0.0, _number(state, "compression_progress")),
        transition=max(0, int(_number(state, "transition_steps_remaining"))),
        previous=parameters.initial_previous,
    )


def _effective_action(
    simulation: _PipelineState, requested: int, parameters: _PipelineParameters
) -> int:
    if simulation.transition > 0:
        simulation.transition -= 1
        if simulation.transition == 0:
            simulation.previous = requested
        return _ACTION["charging"]
    if simulation.previous != requested and (
        simulation.previous in parameters.maneuver or requested in parameters.maneuver
    ):
        simulation.transition = max(0, parameters.settling - 1)
        if simulation.transition == 0:
            simulation.previous = requested
        return _ACTION["charging"]
    simulation.previous = requested
    return requested


def _apply_pipeline_action(
    simulation: _PipelineState,
    effective: int,
    capacity: float,
    parameters: _PipelineParameters,
) -> None:
    if effective != _ACTION["payload_compress"]:
        simulation.progress = 0.0
    if effective == _ACTION["payload_observe"]:
        simulation.raw += parameters.observation_mb
        simulation.uncompressed += 1
    elif effective == _ACTION["payload_compress"] and simulation.uncompressed > 0:
        simulation.progress += 1.0
        if simulation.progress >= parameters.compression_steps:
            simulation.raw = max(0.0, simulation.raw - parameters.observation_mb)
            simulation.compressed += parameters.product_mb
            simulation.uncompressed -= 1
            simulation.progress = 0.0
    elif effective == _ACTION["payload_send"]:
        transfer = min(
            simulation.compressed,
            parameters.send_mb,
            max(0.0, parameters.obc_capacity - simulation.obc),
        )
        simulation.compressed -= transfer
        simulation.obc += transfer
    elif effective == _ACTION["communication"]:
        transfer = min(simulation.obc, capacity)
        simulation.obc -= transfer
        simulation.downlinked += transfer


def _pipeline_score(
    state: Mapping[str, Any], row: np.ndarray, parameters: _PipelineParameters
) -> float:
    simulation = _pipeline_state(state, parameters)
    for offset, requested_value in enumerate(row):
        effective = _effective_action(simulation, int(requested_value), parameters)
        _apply_pipeline_action(simulation, effective, parameters.capacities[offset], parameters)
    staged = simulation.obc + simulation.compressed + simulation.raw / parameters.ratio
    excess = 0.0
    if parameters.has_remaining_capacity:
        remaining_after = max(
            0.0,
            _number(state, "remaining_achievable_downlink_mb") - simulation.downlinked,
        )
        excess = max(0.0, staged - remaining_after)
    period = max(1.0, _number(state, "orbital_period_steps", 94.0))
    time_after_horizon = max(
        0.0,
        _number(state, "time_to_next_pass", period) - len(row),
    )
    proximity = max(0.0, 1.0 - time_after_horizon / period)
    stage_bonus = (
        parameters.pass_stage_reward
        * parameters.downlink_scale
        * min(simulation.obc, 10.0)
        / 10.0
        * (0.25 + 0.75 * proximity)
    )
    return (
        parameters.downlink_reward * parameters.downlink_scale * simulation.downlinked
        + stage_bonus
        - parameters.undeliverable_penalty * excess
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
) -> np.ndarray:
    """Score feasible pipeline bytes and contacts, never power or orbital dynamics."""

    scores = np.zeros(sequences.shape[0], dtype=np.float64)
    if sequences.shape[1] == 0:
        return scores
    parameters = _pipeline_parameters(
        state,
        sequences.shape[1],
        downlink_weight=downlink_weight,
        downlink_reward=downlink_reward,
        pass_stage_reward=pass_stage_reward,
        reference_weight=reference_weight,
        undeliverable_penalty=undeliverable_penalty,
    )
    for sample, row in enumerate(sequences):
        scores[sample] = _pipeline_score(state, row, parameters)
    return scores


__all__ = [
    "contact_capacities",
    "guided_probabilities",
    "pipeline_scores",
    "seed_pipeline_candidate",
]
