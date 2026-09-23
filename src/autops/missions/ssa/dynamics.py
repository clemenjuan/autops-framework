"""SSA action decoding, transition masking, power, and collective reward."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from autops.missions.attitude import settle_mode
from autops.missions.ssa.geometry import satellite_sunlit
from autops.missions.ssa.policy import SSA_MODES
from autops.missions.ssa.transport import contact_seconds

if TYPE_CHECKING:
    from autops.missions.ssa.env import SSAEnvironment


def decode_actions(env: SSAEnvironment, actions: dict[str, Any]) -> dict[str, str]:
    """Decode one command per satellite; misrouted or missing commands are rejected."""

    if not isinstance(actions, dict):
        raise ValueError("SSA actions must map satellite ids to commands")
    unknown = sorted(set(actions) - set(env.satellite_ids))
    missing = sorted(set(env.satellite_ids) - set(actions))
    if unknown or missing:
        raise ValueError(f"SSA commands: unknown satellites {unknown}, missing {missing}")
    decoded: dict[str, str] = {}
    for satellite_id in env.satellite_ids:
        payload = actions[satellite_id]
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
        int(float(env.config["modes"]["transition_overhead"]["settling_time_s"]) / env.timestep_s),
    )
    attitude_modes = set(env.config["modes"]["transition_overhead"]["attitude_maneuver_modes"])
    for satellite_id, requested_mode in requested.items():
        runtime = env.satellites[satellite_id]
        mandatory_safe = runtime.health != "nominal" or runtime.battery_soc <= float(
            env.config["power"]["battery"]["min_soc"]
        )
        logical_mode = _resolve_physical_gate(env, runtime, requested_mode)
        # A slew keeps its initial target; commands while settling are dropped.
        ignored = runtime.transition_steps_remaining > 0 and not mandatory_safe
        (
            effective_mode,
            runtime.previous_mode,
            runtime.transition_steps_remaining,
            in_transition,
        ) = settle_mode(
            logical_mode,
            runtime.previous_mode,
            runtime.transition_steps_remaining,
            settling_steps,
            attitude_modes,
            mandatory_safe=mandatory_safe,
        )
        runtime.mode = effective_mode
        contact = contact_seconds(env, satellite_id, epoch_s)
        resolved[satellite_id] = effective_mode
        information[satellite_id] = {
            "requested_mode": requested_mode,
            "resolved_mode": effective_mode,
            "logical_mode": logical_mode,
            "in_transition": in_transition,
            "command_ignored": ignored,
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
    capacity_wh = float(power["battery"]["capacity_wh"])
    charge_efficiency = float(power["battery"]["charge_efficiency"])
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
            float(power["solar_panels"]["generation_peak_w"])
            * float(power["solar_panels"]["panel_efficiency_factor"])
            * charge_efficiency
            if in_sunlight
            else 0.0
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


def custody_mission_term(env: SSAEnvironment) -> float:
    """Shared custody term: the weighted uncovered (or covered) fraction of the catalog."""

    reward_config = env.config["ssa"]
    target_count = len(env.target_ids)
    custody_fraction = len(env.custody_object_ids) / target_count if target_count else 0.0
    weight = float(reward_config["collective_weight"])
    if bool(reward_config["collective_negative"]):
        return -weight * (1.0 - custody_fraction)
    return weight * custody_fraction


def collective_reward(
    env: SSAEnvironment,
    modes: dict[str, str],
    per_satellite: dict[str, dict[str, Any]],
) -> float:
    reward_config = env.config["ssa"]
    mission = custody_mission_term(env)
    denominator = max(1, len(modes))
    failures = sum(bool(info.get("failure_reason")) for info in per_satellite.values())
    safe_steps = sum(mode == "safe" for mode in modes.values())
    return (
        mission
        - float(reward_config["failed_action_penalty"]) * failures / denominator
        - float(reward_config["safe_penalty"]) * safe_steps / denominator
    )


def _resolve_physical_gate(env: SSAEnvironment, runtime: Any, requested: str) -> str:
    if runtime.health != "nominal" or runtime.battery_soc <= float(
        env.config["power"]["battery"]["min_soc"]
    ):
        return "safe"
    constraint = env.config["modes"]["constraints"].get(requested, {})
    if runtime.battery_soc < float(constraint.get("min_battery_soc", 0.0)):
        return "charging"
    if requested == "isl_share" and runtime.battery_soc < float(env.config["ssa"]["isl_min_soc"]):
        return "charging"
    # Communication is a pointing mode and may begin before AOS; transfer is
    # independently gated by contact duration in the transport layer.
    return requested


def _decode_one_hot(values: list[Any] | tuple[Any, ...]) -> str:
    if len(values) != len(SSA_MODES):
        return "charging"
    ones = [index for index, value in enumerate(values) if int(value) == 1]
    return SSA_MODES[ones[0]] if len(ones) == 1 else "charging"
