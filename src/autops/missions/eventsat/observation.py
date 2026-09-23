"""EventSat onboard information boundary and its vector encoding.

An observation at step ``t`` holds only what the spacecraft could know by
``t``: an ideal GNSS fix with present Sun/station geometry, housekeeping,
software/attitude state, and the outcome of the interval that just ended.
Future contact and eclipse timing (``almanac.py``) is privileged: it remains
available to ground planning, the analytical oracle planner, and trace labels,
but ``onboard_view`` removes it and the observation vector never encodes it.

Declared constants (capacities, rates, durations) scale the vector but are not
encoded: they are fixed in every trace row. Cumulative counters are replaced by
last-interval increments, because monotone totals act as an elapsed-time clock.
The encoded net energy is the platform's: the declared planner charge is left
out because rule-based training collectors never plan, so a planner input would
be constant in training and unseen at deployment. The battery state of charge
still carries the drain.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import timedelta
from typing import Any

import numpy as np

from autops.core.types import SpaceSpec
from autops.missions.eventsat.transitions import record_number
from autops.orbital import NavigationTrack
from autops.wm.schema import EVENTSAT_ACTIONS as MODES
from autops.wm.schema import EVENTSAT_OBSERVATIONS, EVENTSAT_STATES

POSITION_SCALE_KM = 7_000.0
VELOCITY_SCALE_KM_S = 8.0
CENSORED = -1.0

FORECAST_KEYS = (
    "time_to_next_eclipse",
    "time_to_next_pass",
    "next_eclipse_known",
    "next_pass_known",
    "remaining_pass_duration",
    "remaining_pass_duration_s",
    "contact_window_seconds",
    "contact_window_active",
    "physical_ground_pass_active",
    "next_gap_steps",
    "following_gap_steps",
    "planning_gap_steps",
    "future_pass_capacity_mb",
    "achievable_downlink_mb",
    "remaining_achievable_downlink_mb",
    "max_achievable_downlink_mb",
    "planning_contact_seconds",
    "planning_sunlight",
)


def onboard_view(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a decision record without privileged future-event information."""

    return {key: value for key, value in raw.items() if key not in FORECAST_KEYS}


def idle_interval() -> dict[str, Any]:
    """Feedback before the first command: an accepted idle charging interval."""

    return {
        "requested_mode": "charging",
        "safety_resolved_mode": "charging",
        "executed_mode": "charging",
        "action_accepted": True,
        "captured_mb": 0.0,
        "compressed_products": 0,
        "detections": 0,
        "obc_transfer_mb": 0.0,
        "downlinked_mb": 0.0,
        "net_energy_wh": 0.0,
        "planner_energy_wh": 0.0,
    }


def interval_feedback(
    info: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """Onboard-knowable outcome of the interval that just ended."""

    detections = after["total_detections"] - before["total_detections"]
    return {
        "requested_mode": info["requested_mode"],
        "safety_resolved_mode": info["safety_resolved_mode"],
        "executed_mode": info["resolved_mode"],
        "action_accepted": bool(info["action_accepted"]),
        "captured_mb": after["total_raw_captured_mb"] - before["total_raw_captured_mb"],
        "compressed_products": (
            after["undetected_observations"] - before["undetected_observations"] + detections
        ),
        "detections": detections,
        "obc_transfer_mb": info["step_obc_transfer_mb"],
        "downlinked_mb": info["step_downlinked_mb"],
        "net_energy_wh": info["solar_generation_wh"] - info["gross_energy_consumed_wh"],
        "planner_energy_wh": info["planner_compute_energy_wh"],
    }


def navigation_fix(track: NavigationTrack | None, step: int) -> dict[str, Any]:
    """Ideal GNSS fix and present geometry at a step start; invalid without a track."""

    if track is None:
        return {"valid": False}
    return {
        "valid": True,
        "utc": (track.epoch + timedelta(seconds=step * track.step_s)).isoformat(),
        "frame": "ITRF/IERS-2010",
        "position_km": track.position_km[step].tolist(),
        "velocity_km_s": track.velocity_km_s[step].tolist(),
        "sun_unit": track.sun_unit[step].tolist(),
        "station_elevation_deg": float(track.station_elevation_deg[step]),
    }


def _ratio(value: float, denominator: float) -> float:
    return min(1.0, max(0.0, value / max(denominator, 1e-12)))


def _log_fill(value_mb: float, product_mb: float, capacity_mb: float) -> float:
    """Stored products on a log scale that reaches one only at physical capacity.

    A linear capacity ratio maps one science product to ~1e-5 of Jetson storage;
    counting products logarithmically keeps first-product resolution.
    """

    products = max(0.0, value_mb) / product_mb
    return min(1.0, math.log1p(products) / math.log1p(max(1.0, capacity_mb / product_mb)))


def _vector(navigation: Mapping[str, Any], key: str, scale: float) -> tuple[float, ...]:
    if not navigation.get("valid", False):
        return (0.0, 0.0, 0.0)
    return tuple(float(value) / scale for value in navigation[key])


def _one_hot(prefix: str, mode: str) -> dict[str, float]:
    return {f"{prefix}_{name}": float(name == mode) for name in MODES}


def _observation_values(raw: Mapping[str, Any]) -> dict[str, float]:
    navigation = raw.get("navigation") or {"valid": False}
    last = raw.get("last_interval") or idle_interval()
    step_s = record_number(raw, "step_duration_s", 60.0)
    obc_capacity = record_number(raw, "storage_capacity_mb", 4096.0)
    jetson_capacity = record_number(raw, "jetson_capacity_mb", 249036.8)
    product_mb = max(1e-12, record_number(raw, "observation_size_mb", 9.41))
    compressed_mb = product_mb / max(1e-12, record_number(raw, "compression_ratio", 5.11))
    log_capacity = math.log1p(max(1.0, jetson_capacity / product_mb))
    raw_mb = record_number(raw, "jetson_raw_mb")
    jetson_compressed_mb = record_number(raw, "jetson_compressed_mb")
    solar = (raw.get("planning_power") or {}).get("solar_panels", {})
    energy_scale = max(
        1e-12,
        float(solar.get("generation_peak_w", 1.0))
        * float(solar.get("panel_efficiency_factor", 1.0))
        * step_s
        / 3600.0,
    )
    elevation = float(navigation.get("station_elevation_deg", 0.0))
    resolved = str(last["safety_resolved_mode"])
    requested = str(last["requested_mode"])
    values = {
        **dict(
            zip(
                ("position_itrf_x_norm", "position_itrf_y_norm", "position_itrf_z_norm"),
                _vector(navigation, "position_km", POSITION_SCALE_KM),
                strict=True,
            )
        ),
        **dict(
            zip(
                ("velocity_itrf_x_norm", "velocity_itrf_y_norm", "velocity_itrf_z_norm"),
                _vector(navigation, "velocity_km_s", VELOCITY_SCALE_KM_S),
                strict=True,
            )
        ),
        **dict(
            zip(
                ("sun_itrf_x", "sun_itrf_y", "sun_itrf_z"),
                _vector(navigation, "sun_unit", 1.0),
                strict=True,
            )
        ),
        "station_elevation_sin": (
            math.sin(math.radians(elevation)) if navigation.get("valid", False) else 0.0
        ),
        "station_visible": float(bool(raw.get("station_visible", False))),
        "in_sunlight": float(bool(raw.get("in_sunlight", False))),
        "battery_soc": record_number(raw, "battery_soc"),
        "obc_fill_log": _log_fill(record_number(raw, "obc_data_mb"), compressed_mb, obc_capacity),
        "jetson_fill_log": _log_fill(raw_mb + jetson_compressed_mb, product_mb, jetson_capacity),
        "jetson_compressed_fill_log": _log_fill(
            jetson_compressed_mb, compressed_mb, jetson_capacity
        ),
        "health_nominal": float(raw.get("health_status", "nominal") == "nominal"),
        "uncompressed_observations_log": math.log1p(
            max(0.0, record_number(raw, "uncompressed_observations"))
        )
        / log_capacity,
        "compression_progress": _ratio(
            record_number(raw, "compression_progress"),
            record_number(raw, "compression_time_factor", 2.0),
        ),
        "undetected_observations_log": math.log1p(
            max(0.0, record_number(raw, "undetected_observations"))
        )
        / log_capacity,
        "detection_progress": _ratio(
            record_number(raw, "detection_progress"),
            record_number(raw, "detection_time_steps", 5.0),
        ),
        "settling_remaining": _ratio(
            record_number(raw, "transition_steps_remaining"),
            max(1.0, record_number(raw, "settling_time_steps", 1.0)),
        ),
        "last_forced_safe": float(resolved == "safe" and requested != "safe"),
        "last_forced_charging": float(resolved == "charging" and requested != "charging"),
        "last_action_accepted": float(bool(last["action_accepted"])),
        "last_captured": _ratio(float(last["captured_mb"]), product_mb),
        "last_compressed": float(last["compressed_products"] > 0),
        "last_detected": float(last["detections"] > 0),
        "last_obc_transfer_norm": _ratio(
            float(last["obc_transfer_mb"]),
            record_number(raw, "jetson_to_obc_rate_kbps", 8000.0) * step_s / 8000.0,
        ),
        "last_downlink_norm": _ratio(
            float(last["downlinked_mb"]),
            record_number(raw, "downlink_rate_kbps", 50.0) * step_s / 8000.0,
        ),
        "last_platform_energy_norm": (
            float(last["net_energy_wh"]) + float(last["planner_energy_wh"])
        )
        / energy_scale,
        **_one_hot("current_mode", str(raw.get("current_mode", "charging"))),
        **_one_hot("attitude_target", str(raw.get("previous_mode", "charging"))),
    }
    return values


def _state_values(raw: Mapping[str, Any]) -> dict[str, float]:
    mode = str(raw.get("current_mode", "charging"))
    known = {
        "time_to_next_eclipse": bool(raw.get("next_eclipse_known", True)),
        "time_to_next_pass": bool(raw.get("next_pass_known", True)),
    }
    values = {
        name: record_number(raw, name)
        for name in EVENTSAT_STATES
        if name not in {"current_mode_idx", "in_sunlight", "station_visible", "health_nominal"}
    }
    values.update(
        current_mode_idx=float(MODES.index(mode) if mode in MODES else 0),
        in_sunlight=float(bool(raw.get("in_sunlight", False))),
        station_visible=float(bool(raw.get("station_visible", False))),
        contact_window_active=float(bool(raw.get("contact_window_active", False))),
        physical_contact_seconds=record_number(raw, "contact_window_seconds"),
        health_nominal=float(raw.get("health_status", "nominal") == "nominal"),
    )
    for name, is_known in known.items():
        if not is_known:
            values[name] = CENSORED
    return values


def encode_vectors(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Encode the permitted observation vector and privileged state labels."""

    sat = observation.get("satellites", {}).get("eventsat_0", {})
    raw = {
        **sat.get("resources", {}),
        **sat.get("metadata", {}),
        "current_mode": sat.get("status", "charging"),
    }
    observation_values = _observation_values(raw)
    state_values = _state_values(raw)
    obs = np.asarray([observation_values[name] for name in EVENTSAT_OBSERVATIONS], np.float32)
    state = np.asarray([state_values[name] for name in EVENTSAT_STATES], np.float32)
    return obs, state, raw


_SIGNED_INPUTS = frozenset(
    (
        *(f"position_itrf_{axis}_norm" for axis in "xyz"),
        *(f"velocity_itrf_{axis}_norm" for axis in "xyz"),
        *(f"sun_itrf_{axis}" for axis in "xyz"),
        "station_elevation_sin",
    )
)


def observation_space(power: Mapping[str, Any]) -> SpaceSpec:
    """Per-input bounds of the encoded onboard vector under one mission power model.

    Every input is a fraction, flag, log fill, or one-hot in [0, 1] except the signed
    geometry in [-1, 1] and the net platform energy, whose lower bound is the largest
    mode load over one step of peak solar generation (about -2 for EventSat).
    """

    solar = power["solar_panels"]
    generation_w = float(solar["generation_peak_w"]) * float(solar["panel_efficiency_factor"])
    peak_load_w = max(float(load) for mode in MODES for load in power["consumption"][mode].values())
    bounds = {name: (-1.0, 1.0) for name in _SIGNED_INPUTS}
    bounds["last_platform_energy_norm"] = (-peak_load_w / max(generation_w, 1e-12), 1.0)
    low, high = zip(*(bounds.get(name, (0.0, 1.0)) for name in EVENTSAT_OBSERVATIONS), strict=True)
    return SpaceSpec((len(EVENTSAT_OBSERVATIONS),), "float32", low, high, EVENTSAT_OBSERVATIONS)


__all__ = [
    "FORECAST_KEYS",
    "encode_vectors",
    "idle_interval",
    "interval_feedback",
    "navigation_fix",
    "observation_space",
    "onboard_view",
]
