"""EventSat state, safety, settling, and power primitives.

The model parameters are injected from one validated mission YAML. The power
bookkeeping follows the mission design inputs and treats planner inference as
an electrical load; this deployment coupling is central to onboard autonomy
comparisons (see also Hafner et al. 2023, arXiv:2301.04104 for world models).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from autops.missions.eventsat.transitions import total_storage_mb
from autops.wm.schema import EVENTSAT_ACTIONS as MODES


def safety_required(*, battery_soc: float, minimum_soc: float, anomaly_active: bool) -> bool:
    """Return whether the environment must enforce safe mode this step."""

    return anomaly_active or battery_soc <= minimum_soc


def resolve_mode(
    requested: str,
    *,
    mandatory_safe: bool,
    battery_soc: float,
    constraints: Mapping[str, Any],
) -> str:
    """Apply mandatory safety before validating an optional mission command."""

    if mandatory_safe:
        return "safe"
    if requested not in MODES:
        return "charging"
    if battery_soc < float(constraints.get(requested, {}).get("min_battery_soc", 0.0)):
        return "charging"
    return requested


def settle_mode(
    resolved: str,
    target: str,
    remaining: int,
    settling: int,
    maneuver_modes: set[str],
    *,
    mandatory_safe: bool,
) -> tuple[str, str, int, bool]:
    """Return effective mode, attitude target, countdown, and transition flag.

    A slew fixes its target when it starts; commands issued while settling are
    dropped, not queued. Only environment-enforced safety aborts the slew.
    """

    if mandatory_safe:
        return "safe", "safe", 0, False
    if remaining > 0:
        return "charging", target, remaining - 1, True
    maneuver = target != resolved and (resolved in maneuver_modes or target in maneuver_modes)
    if maneuver and settling > 0:
        return "charging", resolved, settling - 1, True
    return resolved, resolved, 0, False


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


def battery_soc_after_energy(
    soc: float, energy_delta_wh: float, capacity_wh: float, charge_efficiency: float
) -> float:
    """Apply charging losses only to surplus energy and clamp the battery state."""

    delta = energy_delta_wh * charge_efficiency if energy_delta_wh > 0 else energy_delta_wh
    return min(1.0, max(0.0, soc + delta / capacity_wh))


def mode_energy_wh(
    power: Mapping[str, Any],
    mode: str,
    in_sunlight: bool,
    step_duration_s: float,
    *,
    planner_energy_wh: float = 0.0,
) -> tuple[float, float]:
    """Return gross load and solar generation for one mission transition."""

    phase = "sun_w" if in_sunlight else "eclipse_w"
    load_w = float(power["consumption"][mode][phase])
    solar = power["solar_panels"]
    generation_w = (
        float(solar["generation_peak_w"]) * float(solar["panel_efficiency_factor"])
        if in_sunlight
        else 0.0
    )
    hours = step_duration_s / 3600.0
    return load_w * hours + max(0.0, planner_energy_wh), generation_w * hours


def power_step(
    state: EventSatState,
    config: dict[str, Any],
    mode: str,
    in_sunlight: bool,
    *,
    planner_energy_wh: float = 0.0,
) -> dict[str, float]:
    power = config["power"]
    planner_wh = max(0.0, float(planner_energy_wh))
    gross_wh, solar_wh = mode_energy_wh(
        power,
        mode,
        in_sunlight,
        float(config["simulation"]["timestep_s"]),
        planner_energy_wh=planner_wh,
    )
    energy_delta = solar_wh - gross_wh
    capacity = float(power["battery"]["capacity_wh"])
    previous = state.battery_soc
    state.battery_soc = battery_soc_after_energy(
        previous, energy_delta, capacity, float(power["battery"]["charge_efficiency"])
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


def planner_event_energy_wh(config: dict[str, Any], mode: str) -> float:
    """Return the declared incremental energy of one planning event.

    Planning powers the Jetson for the whole decision step in which it plans, so
    the charge is independent of the host that runs the simulation. Modes that
    already power the Jetson pay nothing extra, and safe mode keeps the payload
    computer off, so no planning happens there. Measured planning time remains a
    diagnostic and never enters the energy budget.
    """

    power = config["power"]
    if mode == "safe" or mode in set(power.get("jetson_active_modes", [])):
        return 0.0
    model = power.get("planner_compute", {})
    active_w = max(0.0, float(power.get("onboard_compute_w", 0.0)))
    active_s = float(config["simulation"]["timestep_s"])
    boot_wh = max(0.0, float(model.get("boot_energy_wh", 0.0)))
    idle_w = max(0.0, float(model.get("idle_power_w", 0.0)))
    idle_s = max(0.0, float(model.get("idle_time_s", 0.0)))
    return active_w * active_s / 3600.0 + boot_wh + idle_w * idle_s / 3600.0


def advance_projected_battery(state: dict[str, Any], mode: str, sunlight: bool) -> None:
    power = state.get("planning_power")
    if not isinstance(power, Mapping):
        return
    gross_wh, solar_wh = mode_energy_wh(
        power,
        mode,
        sunlight,
        float(state.get("step_duration_s", 60.0)),
    )
    battery = power["battery"]
    state["battery_soc"] = battery_soc_after_energy(
        float(state.get("battery_soc", 0.5)),
        solar_wh - gross_wh,
        float(battery["capacity_wh"]),
        float(battery["charge_efficiency"]),
    )
