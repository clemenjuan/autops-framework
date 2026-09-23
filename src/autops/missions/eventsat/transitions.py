"""Pure atomic transitions for the single EventSat data pipeline.

These functions are shared by the truth environment, analytic rollouts, and
agent what-if tools. Exact fits succeed; rejected discrete products leave all
counters unchanged. Flow transfers may be partial.
"""

from __future__ import annotations

import math
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


def record_number(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read a numeric record field, falling back on absent or malformed values."""

    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def jetson_occupancy_mb(state: Mapping[str, Any]) -> float:
    return max(0.0, record_number(state, "jetson_raw_mb")) + max(
        0.0, record_number(state, "jetson_compressed_mb")
    )


def apply_observe(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    if jetson_occupancy_mb(state) + p.observation_size_mb > p.jetson_capacity_mb + EPSILON_MB:
        return Transition(projected, False, "jetson_capacity")
    projected["jetson_raw_mb"] = record_number(state, "jetson_raw_mb") + p.observation_size_mb
    projected["uncompressed_observations"] = record_number(state, "uncompressed_observations") + 1
    projected["total_raw_captured_mb"] = (
        record_number(state, "total_raw_captured_mb") + p.observation_size_mb
    )
    projected["total_observation_s"] = (
        record_number(state, "total_observation_s") + p.step_duration_s
    )
    return Transition(projected, True)


def apply_compress(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    raw_count = record_number(state, "uncompressed_observations")
    if raw_count < 1 or record_number(state, "jetson_raw_mb") + EPSILON_MB < p.observation_size_mb:
        return Transition(projected, False, "no_raw_product")
    occupied = jetson_occupancy_mb(state) - p.observation_size_mb + p.compressed_observation_mb
    if occupied > p.jetson_capacity_mb + EPSILON_MB:
        return Transition(projected, False, "jetson_capacity")
    projected["jetson_raw_mb"] = max(
        0.0, record_number(state, "jetson_raw_mb") - p.observation_size_mb
    )
    projected["jetson_compressed_mb"] = (
        record_number(state, "jetson_compressed_mb") + p.compressed_observation_mb
    )
    projected["uncompressed_observations"] = raw_count - 1
    projected["undetected_observations"] = record_number(state, "undetected_observations") + 1
    return Transition(projected, True)


def apply_detect(state: Mapping[str, Any], p: PipelineParameters) -> Transition:
    projected = dict(state)
    count = record_number(state, "undetected_observations")
    if count < 1:
        return Transition(projected, False, "no_undetected_product")
    if (
        record_number(state, "obc_data_mb") + p.detection_metadata_mb
        > p.obc_capacity_mb + EPSILON_MB
    ):
        return Transition(projected, False, "obc_capacity")
    projected["undetected_observations"] = count - 1
    projected["obc_data_mb"] = record_number(state, "obc_data_mb") + p.detection_metadata_mb
    projected["total_detections"] = record_number(state, "total_detections") + 1
    return Transition(projected, True)


def apply_can_transfer(
    state: Mapping[str, Any], p: PipelineParameters, *, duration_s: float | None = None
) -> Transition:
    projected = dict(state)
    seconds = p.step_duration_s if duration_s is None else max(0.0, duration_s)
    source = max(0.0, record_number(state, "jetson_compressed_mb"))
    headroom = max(0.0, p.obc_capacity_mb - record_number(state, "obc_data_mb"))
    rate_limit = p.jetson_to_obc_rate_kbps / 8.0 * seconds / 1000.0
    amount = min(source, headroom, rate_limit)
    if amount <= EPSILON_MB:
        return Transition(
            projected, False, "no_source_data" if source <= EPSILON_MB else "obc_capacity"
        )
    raw_equivalent = amount * p.compression_ratio
    projected["jetson_compressed_mb"] = source - amount
    projected["obc_data_mb"] = record_number(state, "obc_data_mb") + amount
    projected["obc_raw_equivalent_mb"] = (
        record_number(state, "obc_raw_equivalent_mb") + raw_equivalent
    )
    return Transition(projected, True, transferred_mb=amount, raw_equivalent_mb=raw_equivalent)


def apply_downlink(
    state: Mapping[str, Any], p: PipelineParameters, *, contact_seconds: float
) -> Transition:
    projected = dict(state)
    seconds = max(0.0, contact_seconds)
    source = max(0.0, record_number(state, "obc_data_mb"))
    amount = min(source, p.downlink_rate_kbps / 8.0 * seconds / 1000.0)
    if amount <= EPSILON_MB:
        return Transition(projected, False, "no_contact" if seconds <= 0 else "no_source_data")
    raw_backlog = max(0.0, record_number(state, "obc_raw_equivalent_mb"))
    raw_equivalent = min(raw_backlog, amount * raw_backlog / source) if source else 0.0
    projected["obc_data_mb"] = source - amount
    projected["obc_raw_equivalent_mb"] = raw_backlog - raw_equivalent
    projected["data_downlinked_mb"] = record_number(state, "data_downlinked_mb") + amount
    projected["downlink_raw_equivalent_mb"] = (
        record_number(state, "downlink_raw_equivalent_mb") + raw_equivalent
    )
    return Transition(projected, True, transferred_mb=amount, raw_equivalent_mb=raw_equivalent)


def preloaded_pipeline(
    initial: Mapping[str, Any], storage: Mapping[str, Any]
) -> dict[str, float | int]:
    """Validated diagnostic science data present at reset; canonical episodes start empty.

    Preloaded data was not captured in the episode, so capture totals stay zero.
    """

    raw = initial["raw_observations"]
    obc_mb = float(initial["obc_data_mb"])
    compressed_mb = float(initial["jetson_compressed_mb"])
    product_mb = float(storage["observation_size_mb"])
    ratio = float(storage["compression_ratio"])
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError("initial_state.raw_observations must be a non-negative integer")
    if not math.isfinite(obc_mb) or not 0.0 <= obc_mb <= float(storage["obc_capacity_mb"]):
        raise ValueError("initial_state.obc_data_mb must be finite and within OBC capacity")
    jetson_mb = compressed_mb + raw * product_mb
    if not math.isfinite(compressed_mb) or compressed_mb < 0.0:
        raise ValueError("initial_state.jetson_compressed_mb must be finite and non-negative")
    if jetson_mb > float(storage["jetson_capacity_mb"]):
        raise ValueError("initial_state Jetson data exceeds Jetson capacity")
    return {
        "obc_data_mb": obc_mb,
        "obc_raw_equivalent_mb": obc_mb * ratio,
        "jetson_compressed_mb": compressed_mb,
        "undetected_observations": int(compressed_mb * ratio / product_mb),
        "jetson_raw_mb": raw * product_mb,
        "uncompressed_observations": raw,
    }


def failure_reason(
    mode: str, outcome: Transition | None, had_product: bool, contact_s: float
) -> str | None:
    """Why an executed mode made no pipeline progress, or None when it did or could not."""

    if outcome is not None:
        return None if outcome.accepted else outcome.reason
    if mode == "payload_compress" and not had_product:
        return "no_raw_product"
    if mode == "payload_detect" and not had_product:
        return "no_undetected_product"
    if mode == "communication" and contact_s <= 0.0:
        return "no_contact"
    return None


def total_storage_mb(state: Mapping[str, Any]) -> float:
    return sum(
        max(0.0, record_number(state, key))
        for key in ("jetson_raw_mb", "jetson_compressed_mb", "obc_data_mb")
    )
