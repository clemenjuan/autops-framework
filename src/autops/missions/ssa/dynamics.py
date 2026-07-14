"""SSA action decoding, transition masking, power, and collective reward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autops.missions.ssa.geometry import satellite_sunlit
from autops.missions.ssa.policy import SSA_MODES
from autops.missions.ssa.transport import contact_seconds

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def decode_actions(env: SSAEnvironment, actions: dict[str, Any]) -> dict[str, str]:
    decoded: dict[str, str] = {}
    for satellite_id in env.satellite_ids:
        payload = actions.get(satellite_id, {}) if isinstance(actions, dict) else {}
        mode: Any = payload.get("mode") if isinstance(payload, dict) else payload
        if isinstance(mode, (list, tuple)):
            mode = _decode_one_hot(mode)
        decoded[satellite_id] = str(mode) if mode in SSA_MODES else "charging"
    return decoded


def resolve_actions(
    env: SSAEnvironment,
    requested: dict[str, str],
    epoch_s: float,
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    resolved: dict[str, str] = {}
    information: dict[str, dict[str, Any]] = {}
    settling_steps = max(
        0,
        int(float(env.config["transitions"]["settling_time_s"]) / env.timestep_s),
    )
    attitude_modes = set(env.config["transitions"]["attitude_modes"])
    for satellite_id, requested_mode in requested.items():
        runtime = env.satellites[satellite_id]
        logical_mode = _resolve_physical_gate(env, runtime, requested_mode)
        in_transition = False
        if settling_steps and runtime.transition_steps_remaining > 0:
            runtime.transition_steps_remaining -= 1
            effective_mode = "charging"
            in_transition = True
            if runtime.transition_steps_remaining == 0:
                runtime.previous_mode = logical_mode
        elif settling_steps and _requires_maneuver(
            runtime.previous_mode,
            logical_mode,
            attitude_modes,
        ):
            runtime.transition_steps_remaining = max(0, settling_steps - 1)
            effective_mode = "charging"
            in_transition = True
            if runtime.transition_steps_remaining == 0:
                runtime.previous_mode = logical_mode
        else:
            effective_mode = logical_mode
            runtime.previous_mode = effective_mode
        runtime.mode = effective_mode
        contact = contact_seconds(env, satellite_id, epoch_s)
        resolved[satellite_id] = effective_mode
        information[satellite_id] = {
            "requested_mode": requested_mode,
            "resolved_mode": effective_mode,
            "logical_mode": logical_mode,
            "in_transition": in_transition,
            "contact_seconds": contact,
            "physical_ground_pass_active": contact > 0.0,
            "downlinked_records": 0,
        }
    return resolved, information


def apply_power(
    env: SSAEnvironment,
    modes: dict[str, str],
    epoch_s: float,
    per_satellite: dict[str, dict[str, Any]],
) -> None:
    power = env.config["power"]
    capacity_wh = float(power["battery_capacity_wh"])
    charge_efficiency = float(power.get("charge_efficiency", 0.9))
    for satellite_id, mode in modes.items():
        runtime = env.satellites[satellite_id]
        position = env.satellite_position(satellite_id, epoch_s)
        in_sunlight = satellite_sunlit(position, epoch_s)
        consumption_mode = "charging" if mode == "isl_share" else mode
        phase = "sun_w" if in_sunlight else "eclipse_w"
        load_w = float(power["consumption"].get(consumption_mode, {}).get(phase, 12.0))
        if mode == "isl_share":
            load_w += float(env.config["isl"]["power_overhead_w"])
        generation_w = (
            float(power["solar_generation_w"]) * charge_efficiency if in_sunlight else 0.0
        )
        duration_h = env.timestep_s / 3600.0
        previous_soc = runtime.battery_soc
        runtime.battery_soc = min(
            1.0,
            max(0.0, previous_soc + (generation_w - load_w) * duration_h / capacity_wh),
        )
        gross_energy = load_w * duration_h
        runtime.energy_consumed_wh += gross_energy
        per_satellite[satellite_id].update(
            {
                "in_sunlight": in_sunlight,
                "prev_battery_soc": previous_soc,
                "battery_soc": runtime.battery_soc,
                "gross_energy_consumed_wh": gross_energy,
                "isl_energy_consumed_wh": (
                    float(env.config["isl"]["power_overhead_w"]) * duration_h
                    if mode == "isl_share"
                    else 0.0
                ),
            }
        )


def collective_reward(
    env: SSAEnvironment,
    modes: dict[str, str],
    per_satellite: dict[str, dict[str, Any]],
) -> float:
    reward_config = env.config["reward"]
    target_count = len(env.target_ids)
    custody_fraction = len(env.custody_object_ids) / target_count if target_count else 0.0
    mission_scale = float(reward_config["mission_scale"])
    mission = (
        -mission_scale * (1.0 - custody_fraction)
        if bool(reward_config["collective_negative"])
        else mission_scale * custody_fraction
    )
    denominator = max(1, len(modes))
    failures = sum(bool(info.get("failure_reason")) for info in per_satellite.values())
    safe_steps = sum(mode == "safe" for mode in modes.values())
    return (
        mission
        - float(reward_config["failed_action_penalty"]) * failures / denominator
        - float(reward_config["safe_penalty"]) * safe_steps / denominator
    )


def _resolve_physical_gate(env: SSAEnvironment, runtime: Any, requested: str) -> str:
    if runtime.health != "nominal" or runtime.battery_soc <= float(env.config["power"]["min_soc"]):
        return "safe"
    minimum_soc = 0.3
    if requested in {"payload_observe", "payload_detect"} and runtime.battery_soc < minimum_soc:
        return "charging"
    if requested == "isl_share" and runtime.battery_soc < float(env.config["ssa"]["isl_min_soc"]):
        return "charging"
    # Communication is a pointing mode and may begin before AOS; transfer is
    # independently gated by contact duration in the transport layer.
    return requested


def _requires_maneuver(previous: str, requested: str, attitude_modes: set[str]) -> bool:
    return previous != requested and (previous in attitude_modes or requested in attitude_modes)


def _decode_one_hot(values: list[Any] | tuple[Any, ...]) -> str:
    if len(values) != len(SSA_MODES):
        return "charging"
    ones = [index for index, value in enumerate(values) if int(value) == 1]
    return SSA_MODES[ones[0]] if len(ones) == 1 else "charging"
