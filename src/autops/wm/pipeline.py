"""Analytical byte-pipeline aids for EventSat CEM planning.

``seed_pipeline_candidate`` injects one proposal that can complete the payload
pipeline, and ``pipeline_scores`` shapes candidate scores with the projected
byte-pipeline effects. Both are optional aids beside the learned or analytical
scorer; without the privileged almanac they fall back to present-visibility
contact and an unbounded remaining downlink capacity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from autops.missions.eventsat.transitions import record_number
from autops.wm.guidance import (
    CandidateProjection,
    contact_capacities,
    project_executable_candidates,
)
from autops.wm.schema import EVENTSAT_ACTIONS

_ACTION = {name: index for index, name in enumerate(EVENTSAT_ACTIONS)}


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
    settling = max(0, int(record_number(state, "settling_time_steps")))
    ratio = max(1e-12, record_number(state, "compression_ratio", 5.11))
    product_mb = max(0.0, record_number(state, "observation_size_mb", 9.41)) / ratio
    raw_mb = record_number(state, "jetson_raw_mb")
    compressed_mb = record_number(state, "jetson_compressed_mb")
    obc_mb = record_number(state, "obc_data_mb")
    staged_mb = obc_mb + compressed_mb + raw_mb / ratio
    remaining_mb = max(
        0.0,
        record_number(state, "remaining_achievable_downlink_mb", float("inf")),
    )
    has_raw = raw_mb > 0.01 or record_number(state, "uncompressed_observations") > 0.0
    produced_compressed = False

    if obc_mb <= 0.01 and compressed_mb <= 0.01:
        if not has_raw and staged_mb + product_mb <= remaining_mb + 1e-9:
            stop = min(horizon, cursor + settling + 1)
            row[cursor:stop] = _ACTION["payload_observe"]
            cursor = stop
            has_raw = True
        if has_raw and cursor < horizon:
            compression_steps = max(
                1, int(np.ceil(record_number(state, "compression_time_factor", 2.0)))
            )
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
    ratio = max(1e-12, record_number(state, "compression_ratio", 5.11))
    staged = (
        record_number(terminal, "obc_data_mb")
        + record_number(terminal, "jetson_compressed_mb")
        + record_number(terminal, "jetson_raw_mb") / ratio
    )
    downlinked = max(
        0.0,
        record_number(terminal, "data_downlinked_mb") - record_number(state, "data_downlinked_mb"),
    )
    excess = 0.0
    if state.get("remaining_achievable_downlink_mb") is not None:
        remaining_after = max(
            0.0,
            record_number(state, "remaining_achievable_downlink_mb") - downlinked,
        )
        excess = max(0.0, staged - remaining_after)
    period = max(1.0, record_number(state, "orbital_period_steps", 94.0))
    time_after_horizon = max(
        0.0,
        record_number(state, "time_to_next_pass", period) - horizon,
    )
    proximity = max(0.0, 1.0 - time_after_horizon / period)
    stage_bonus = (
        pass_stage_reward
        * downlink_scale
        * min(record_number(terminal, "obc_data_mb"), 10.0)
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
    elif projection.sequences is not sequences and not np.array_equal(
        projection.sequences, np.asarray(sequences)
    ):
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


__all__ = ["pipeline_scores", "seed_pipeline_candidate"]
