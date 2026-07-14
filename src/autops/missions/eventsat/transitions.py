"""Pure atomic transitions for the single EventSat data pipeline.

These functions are shared by the truth environment, analytic rollouts, and
agent what-if tools. Exact fits succeed; rejected discrete products leave all
counters unchanged. Flow transfers may be partial.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

EPSILON_MB = 1e-12


@dataclass(frozen=True)
class PipelineParameters:
    observation_size_mb: float
    compression_ratio: float
    jetson_capacity_mb: float
    obc_capacity_mb: float
    detection_metadata_mb: float
    jetson_to_obc_rate_kbps: float
    downlink_rate_kbps: float
    step_duration_s: float

    @property
    def compressed_observation_mb(self) -> float:
        return self.observation_size_mb / max(self.compression_ratio, EPSILON_MB)


@dataclass(frozen=True)
class Transition:
    state: dict[str, Any]
    accepted: bool
    reason: str | None = None
    transferred_mb: float = 0.0
    raw_equivalent_mb: float = 0.0


def _number(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def jetson_occupancy_mb(state: Mapping[str, Any]) -> float:
    return max(0.0, _number(state, "jetson_raw_mb")) + max(
        0.0, _number(state, "jetson_compressed_mb")
    )


def apply_observe(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    if jetson_occupancy_mb(state) + p.observation_size_mb > p.jetson_capacity_mb + EPSILON_MB:
        return Transition(projected, False, "jetson_capacity")
    projected["jetson_raw_mb"] = _number(state, "jetson_raw_mb") + p.observation_size_mb
    projected["uncompressed_observations"] = _number(state, "uncompressed_observations") + 1
    projected["total_raw_captured_mb"] = (
        _number(state, "total_raw_captured_mb") + p.observation_size_mb
    )
    projected["total_observation_s"] = _number(state, "total_observation_s") + p.step_duration_s
    return Transition(projected, True)


def apply_compress(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    raw_count = _number(state, "uncompressed_observations")
    if raw_count < 1 or _number(state, "jetson_raw_mb") + EPSILON_MB < p.observation_size_mb:
        return Transition(projected, False, "no_raw_product")
    occupied = jetson_occupancy_mb(state) - p.observation_size_mb + p.compressed_observation_mb
    if occupied > p.jetson_capacity_mb + EPSILON_MB:
        return Transition(projected, False, "jetson_capacity")
    projected["jetson_raw_mb"] = max(0.0, _number(state, "jetson_raw_mb") - p.observation_size_mb)
    projected["jetson_compressed_mb"] = (
        _number(state, "jetson_compressed_mb") + p.compressed_observation_mb
    )
    projected["uncompressed_observations"] = raw_count - 1
    projected["undetected_observations"] = _number(state, "undetected_observations") + 1
    return Transition(projected, True)


def apply_detect(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    count = _number(state, "undetected_observations")
    if count < 1:
        return Transition(projected, False, "no_undetected_product")
    if _number(state, "obc_data_mb") + p.detection_metadata_mb > p.obc_capacity_mb + EPSILON_MB:
        return Transition(projected, False, "obc_capacity")
    projected["undetected_observations"] = count - 1
    projected["obc_data_mb"] = _number(state, "obc_data_mb") + p.detection_metadata_mb
    projected["total_detections"] = _number(state, "total_detections") + 1
    return Transition(projected, True)


def apply_can_transfer(
    state: Mapping[str, Any], p: PipelineParameters, *, duration_s: float | None = None
) -> Transition:
    projected = dict(state)
    seconds = p.step_duration_s if duration_s is None else max(0.0, duration_s)
    source = max(0.0, _number(state, "jetson_compressed_mb"))
    headroom = max(0.0, p.obc_capacity_mb - _number(state, "obc_data_mb"))
    rate_limit = p.jetson_to_obc_rate_kbps / 8.0 * seconds / 1000.0
    amount = min(source, headroom, rate_limit)
    if amount <= EPSILON_MB:
        return Transition(
            projected, False, "no_source_data" if source <= EPSILON_MB else "obc_capacity"
        )
    raw_equivalent = amount * p.compression_ratio
    projected["jetson_compressed_mb"] = source - amount
    projected["obc_data_mb"] = _number(state, "obc_data_mb") + amount
    projected["obc_raw_equivalent_mb"] = _number(state, "obc_raw_equivalent_mb") + raw_equivalent
    return Transition(projected, True, transferred_mb=amount, raw_equivalent_mb=raw_equivalent)


def apply_downlink(
    state: Mapping[str, Any], p: PipelineParameters, *, contact_seconds: float
) -> Transition:
    projected = dict(state)
    seconds = max(0.0, contact_seconds)
    source = max(0.0, _number(state, "obc_data_mb"))
    amount = min(source, p.downlink_rate_kbps / 8.0 * seconds / 1000.0)
    if amount <= EPSILON_MB:
        return Transition(projected, False, "no_contact" if seconds <= 0 else "no_source_data")
    raw_backlog = max(0.0, _number(state, "obc_raw_equivalent_mb"))
    raw_equivalent = min(raw_backlog, amount * raw_backlog / source) if source else 0.0
    projected["obc_data_mb"] = source - amount
    projected["obc_raw_equivalent_mb"] = raw_backlog - raw_equivalent
    projected["data_downlinked_mb"] = _number(state, "data_downlinked_mb") + amount
    projected["downlink_raw_equivalent_mb"] = (
        _number(state, "downlink_raw_equivalent_mb") + raw_equivalent
    )
    return Transition(projected, True, transferred_mb=amount, raw_equivalent_mb=raw_equivalent)


def total_storage_mb(state: Mapping[str, Any]) -> float:
    return sum(
        max(0.0, _number(state, key))
        for key in ("jetson_raw_mb", "jetson_compressed_mb", "obc_data_mb")
    )
