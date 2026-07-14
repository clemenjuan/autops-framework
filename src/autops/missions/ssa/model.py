"""SSA mission defaults and compact mutable episode state."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

MISSION_DEFAULTS: dict[str, Any] = {
    "simulation": {"timestep_s": 60.0, "max_steps": 10080},
    "constellation": {
        "size": 3,
        "shared_plane": True,
        "in_plane_spacing_deg": 2.0,
        "fixed_positions_km": {},
    },
    "orbit": {
        "altitude_km": 775.0,
        "inclination_deg": 98.6,
        "eccentricity": 0.001,
        "period_s": 6012.0,
    },
    "power": {
        "solar_generation_w": 120.0,
        "battery_capacity_wh": 300.0,
        "initial_soc": 0.8,
        "min_soc": 0.2,
        "consumption": {
            "charging": {"sun_w": 9.6, "eclipse_w": 9.2},
            "payload_observe": {"sun_w": 25.4, "eclipse_w": 25.0},
            "payload_detect": {"sun_w": 30.2, "eclipse_w": 29.8},
            "communication": {"sun_w": 61.0, "eclipse_w": 60.6},
            "safe": {"sun_w": 12.0, "eclipse_w": 12.0},
        },
    },
    "storage": {
        "obc_capacity_mb": 4096.0,
        "jetson_capacity_mb": 249036.8,
        "observation_size_mb": 2016.0,
    },
    "payload": {"detection_time_s": 300.0},
    "transitions": {
        "settling_time_s": 135.0,
        "attitude_modes": ["payload_observe", "communication"],
    },
    "targets": {
        "count": 100,
        "parent_altitude_km": 805.0,
        "parent_inclination_deg": 98.6,
        "raan_spread_deg": 0.3,
        "along_track_sigma_ms": 13.0,
        "normal_sigma_ms": 26.0,
        "size_bounds_m": [0.01, 0.10],
        "fov_half_angle_deg": 1.9,
        "boresight_pitch_deg": 12.0,
        "range_cap_km": 150.0,
        "magnitude_limit": 15.0,
        "magnitude_sigma": 0.5,
        "albedo": 0.13,
        "fixed_positions_km": {},
    },
    "ssa": {
        "custody_tau_steps": 4320,
        "record_size_kb": 10.0,
        "isl_relay": True,
        "isl_min_soc": 0.3,
    },
    "isl": {
        "frequency_hz": 437e6,
        "tx_power_w": 2.0,
        "tx_gain_db": 2.15,
        "rx_gain_db": 2.15,
        "tx_loss_db": 3.0,
        "rx_loss_db": 0.5,
        "bandwidth_hz": 9600.0,
        "symbol_rate_hz": 9600.0,
        "modulation_order": 4,
        "sensitivity_dbw": -132.0,
        "noise_temperature_k": 290.0,
        "substep_resolution_s": 10.0,
        "power_overhead_w": 5.0,
        "unicast": True,
    },
    "ground_station": {
        "latitude_deg": 48.0483,
        "longitude_deg": 11.6567,
        "min_elevation_deg": 10.0,
        "always_visible": False,
        "substep_resolution_s": 10.0,
    },
    "reward": {
        "collective_negative": True,
        "mission_scale": 1.0,
        "failed_action_penalty": 0.1,
        "safe_penalty": 0.3,
    },
}


def merge_config(update: dict[str, Any] | None) -> dict[str, Any]:
    def merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(base)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result

    return merge(MISSION_DEFAULTS, update or {})


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
