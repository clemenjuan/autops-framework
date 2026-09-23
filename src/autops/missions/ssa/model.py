"""SSA mission defaults and compact mutable episode state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from autops.config import asset_root, load_yaml, strict_deep_merge


def merge_config(update: dict[str, Any] | None) -> dict[str, Any]:
    """Load the mission authority and reject alternate/unknown configuration keys."""

    base = load_yaml(asset_root() / "configs" / "missions" / "ssa.yaml")
    extra = deepcopy(update or {})
    # Fixed geometry is a named test/calibration seam with object IDs as keys.
    for section in ("constellation", "targets"):
        positions = extra.get(section, {}).get("fixed_positions_km", {})
        base[section]["fixed_positions_km"] = deepcopy(positions)
    config = strict_deep_merge(base, extra)
    if config["orbit"]["propagator"] != "keplerian":
        raise ValueError("SSA currently implements only the keplerian orbital backend")
    return config


@dataclass
class DetectionBatch:
    observation_step: int
    raw_mb: float
    detections: list[dict[str, Any]]


@dataclass
class SatelliteRuntime:
    satellite_id: str
    battery_soc: float
    detection_row: list[int]
    mode: str = "charging"
    previous_mode: str = "charging"
    transition_steps_remaining: int = 0
    health: str = "nominal"
    jetson_raw_mb: float = 0.0
    detection_progress_s: float = 0.0
    pending_batches: list[DetectionBatch] = field(default_factory=list)
    estimates: dict[str, dict[str, Any]] = field(default_factory=dict)
    ground_catalog_steps: dict[str, int] = field(default_factory=dict)
    first_known_steps: dict[str, int] = field(default_factory=dict)
    undelivered: dict[str, dict[str, Any]] = field(default_factory=dict)
    last_command: dict[str, Any] = field(default_factory=lambda: {"mode": "charging"})
    energy_consumed_wh: float = 0.0

    @property
    def oldest_record_step(self) -> int | None:
        if not self.undelivered:
            return None
        return min(record_step(record) for record in self.undelivered.values())


def record_step(record: dict[str, Any]) -> int:
    return int(record.get("obs_step", record.get("step", 0)))


def _origin(record: dict[str, Any]) -> tuple[str, str]:
    return str(record.get("satellite_id", "")), str(record.get("object_id", ""))


def estimate_is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    """Knowledge ranks by track quality, then acquisition step, then stable origin."""

    candidate_quality, current_quality = (
        float(candidate.get("quality", 0.0)),
        float(current.get("quality", 0.0)),
    )
    if candidate_quality != current_quality:
        return candidate_quality > current_quality
    if record_step(candidate) != record_step(current):
        return record_step(candidate) > record_step(current)
    return _origin(candidate) < _origin(current)


def custody_record_is_better(candidate: dict[str, Any], held: dict[str, Any]) -> bool:
    """Custody records rank by recency first, since custody values fresh delivery."""

    if record_step(candidate) != record_step(held):
        return record_step(candidate) > record_step(held)
    candidate_quality, held_quality = (
        float(candidate.get("quality", 0.0)),
        float(held.get("quality", 0.0)),
    )
    if candidate_quality != held_quality:
        return candidate_quality > held_quality
    return _origin(candidate) < _origin(held)
