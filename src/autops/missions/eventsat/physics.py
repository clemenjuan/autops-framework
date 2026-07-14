"""EventSat state, encoding, and power primitives.

The model parameters are injected from one validated mission YAML. The power
bookkeeping follows the mission design inputs and treats planner inference as
an electrical load; this deployment coupling is central to onboard autonomy
comparisons (see also Hafner et al. 2023, arXiv:2301.04104 for world models).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from autops.missions.eventsat.transitions import total_storage_mb
from autops.wm.schema import EVENTSAT_ACTIONS as MODES


@dataclass
class EventSatState:
    step: int = 0
    battery_soc: float = 0.8
    current_mode: str = "charging"
    previous_mode: str = "charging"
    jetson_raw_mb: float = 0.0
    jetson_compressed_mb: float = 0.0
    obc_data_mb: float = 0.0
    data_downlinked_mb: float = 0.0
    total_raw_captured_mb: float = 0.0
    obc_raw_equivalent_mb: float = 0.0
    downlink_raw_equivalent_mb: float = 0.0
    uncompressed_observations: int = 0
    undetected_observations: int = 0
    compression_progress: int = 0
    detection_progress: int = 0
    total_observation_s: float = 0.0
    total_detections: int = 0
    total_contact_s: float = 0.0
    transition_steps_remaining: int = 0
    transition_target: str | None = None
    active_anomaly: str | None = None
    forced_safe_steps: int = 0
    cumulative_gross_wh: float = 0.0
    cumulative_solar_wh: float = 0.0
    cumulative_planner_wh: float = 0.0
    orbit_elements: dict[str, float] = field(default_factory=dict)

    def pipeline(self) -> dict[str, Any]:
        return {
            key: getattr(self, key)
            for key in (
                "jetson_raw_mb",
                "jetson_compressed_mb",
                "obc_data_mb",
                "data_downlinked_mb",
                "total_raw_captured_mb",
                "obc_raw_equivalent_mb",
                "downlink_raw_equivalent_mb",
                "uncompressed_observations",
                "undetected_observations",
                "total_observation_s",
                "total_detections",
            )
        }

    def accept_pipeline(self, values: dict[str, Any]) -> None:
        for key in self.pipeline():
            if key in values:
                setattr(self, key, values[key])

    @property
    def data_stored_mb(self) -> float:
        return total_storage_mb(self.pipeline())


def _ratio(value: Any, denominator: float) -> float:
    try:
        return min(1.0, max(0.0, float(value) / max(denominator, 1e-12)))
    except (TypeError, ValueError):
        return 0.0


def encode_vectors(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    sat = observation.get("satellites", {}).get("eventsat_0", {})
    resources = sat.get("resources", {})
    raw = {**resources, **sat.get("metadata", {}), "current_mode": sat.get("status", "charging")}
    step = int(observation.get("step", 0))
    max_steps = max(1, int(observation.get("global", {}).get("max_steps", 10080)))
    period = max(1.0, float(raw.get("orbital_period_steps", 92)))
    obc_capacity = float(raw.get("storage_capacity_mb", 4096.0))
    jetson_capacity = float(raw.get("jetson_capacity_mb", 249036.8))
    phase = float(raw.get("orbital_phase", 0.0))
    mode = str(raw.get("current_mode", "charging"))
    obs = np.zeros(25, dtype=np.float32)
    obs[:18] = (
        float(raw.get("battery_soc", 0.0)),
        _ratio(raw.get("obc_data_mb", 0.0), obc_capacity),
        _ratio(raw.get("jetson_raw_mb", 0.0), jetson_capacity),
        _ratio(raw.get("jetson_compressed_mb", 0.0), jetson_capacity),
        math.sin(phase * 2 * math.pi),
        math.cos(phase * 2 * math.pi),
        min(float(raw.get("time_to_next_eclipse", period)) / period, 1.0),
        min(float(raw.get("time_to_next_pass", period)) / period, 1.0),
        min(float(raw.get("remaining_pass_duration", 0.0)) / 10.0, 1.0),
        step / max_steps,
        float(bool(raw.get("in_sunlight", False))),
        float(bool(raw.get("contact_window_active", False))),
        float(raw.get("health_status", "nominal") == "nominal"),
        min(float(raw.get("uncompressed_observations", 0.0)) / 10.0, 1.0),
        min(float(raw.get("compression_progress", 0.0)) / 2.0, 1.0),
        min(float(raw.get("undetected_observations", 0.0)) / 10.0, 1.0),
        min(float(raw.get("detection_progress", 0.0)) / 5.0, 1.0),
        _ratio(raw.get("data_downlinked_mb", 0.0), raw.get("max_achievable_downlink_mb", 1.0)),
    )
    obs[18 + (MODES.index(mode) if mode in MODES else 0)] = 1.0
    state_values = [
        raw.get("battery_soc", 0.0),
        MODES.index(mode) if mode in MODES else 0,
        float(bool(raw.get("in_sunlight", False))),
        float(bool(raw.get("contact_window_active", False))),
        phase,
        raw.get("time_to_next_eclipse", period),
        raw.get("time_to_next_pass", period),
        raw.get("remaining_pass_duration", 0.0),
        raw.get("following_gap_steps", period),
        raw.get("data_stored_mb", 0.0),
        raw.get("obc_data_mb", 0.0),
        raw.get("jetson_raw_mb", 0.0),
        raw.get("jetson_compressed_mb", 0.0),
        raw.get("data_downlinked_mb", 0.0),
        raw.get("uncompressed_observations", 0.0),
        raw.get("compression_progress", 0.0),
        raw.get("undetected_observations", 0.0),
        raw.get("detection_progress", 0.0),
        raw.get("total_observation_s", 0.0),
        raw.get("total_detections", 0.0),
        obc_capacity,
        jetson_capacity,
        raw.get("remaining_achievable_downlink_mb", 0.0),
        raw.get("achievable_downlink_mb", 0.0),
        float(raw.get("health_status", "nominal") == "nominal"),
    ]
    return obs, np.asarray(state_values, dtype=np.float32), raw


def power_step(
    state: EventSatState,
    config: dict[str, Any],
    mode: str,
    in_sunlight: bool,
    *,
    planner_power_w: float = 0.0,
) -> dict[str, float]:
    power = config["power"]
    phase = "sun_w" if in_sunlight else "eclipse_w"
    load_w = float(power["consumption"][mode][phase])
    if mode not in set(power.get("jetson_active_modes", [])):
        load_w += max(0.0, planner_power_w)
    solar = power["solar_panels"]
    generation_w = (
        float(solar["generation_peak_w"]) * float(solar["panel_efficiency_factor"])
        if in_sunlight
        else 0.0
    )
    hours = float(config["simulation"]["timestep_s"]) / 3600.0
    gross_wh = load_w * hours
    solar_wh = generation_w * hours
    energy_delta = solar_wh - gross_wh
    if energy_delta > 0:
        energy_delta *= float(power["battery"]["charge_efficiency"])
    capacity = float(power["battery"]["capacity_wh"])
    previous = state.battery_soc
    state.battery_soc = min(1.0, max(0.0, previous + energy_delta / capacity))
    planner_wh = (
        max(0.0, planner_power_w) * hours
        if mode not in power.get("jetson_active_modes", [])
        else 0.0
    )
    state.cumulative_gross_wh += gross_wh
    state.cumulative_solar_wh += solar_wh
    state.cumulative_planner_wh += planner_wh
    return {
        "gross_energy_consumed_wh": gross_wh,
        "solar_generation_wh": solar_wh,
        "net_battery_depletion_wh": max(0.0, (previous - state.battery_soc) * capacity),
        "planner_compute_energy_wh": planner_wh,
    }
