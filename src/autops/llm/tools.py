"""Side-effect-free what-if tools for the bounded EventSat agentic planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autops.missions.eventsat.physics import MODES
from autops.missions.eventsat.transitions import record_number


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, str]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


_TOOLS = {
    "check_constraints": ToolDefinition(
        "check_constraints",
        "Pre-validate whether a proposed mode is feasible given the current state. "
        "Returns violations and warnings.",
        {"state": "Current satellite state dict", "proposed_mode": "Mode to check (string)"},
    ),
    "evaluate_plan": ToolDefinition(
        "evaluate_plan",
        "Evaluate a proposed mode using incremental contact-deliverable value and physical "
        "risk factors.",
        {
            "state": "Current satellite state dict",
            "proposed_mode": "Mode to evaluate (string)",
        },
    ),
}

# Echo tools are intentionally folded into prompts. Only genuine what-if actions
# cost a model round trip; this is the qwen thinking-spiral latency fix.
SCHEDULE_TOOL_NAMES = ["check_constraints", "evaluate_plan"]


def _has_almanac(state: dict[str, Any]) -> bool:
    """Ground records carry the contact almanac; the onboard view does not."""

    return "contact_window_seconds" in state


def _get_pipeline_bottleneck(state: dict[str, Any]) -> str:
    if record_number(state, "uncompressed_observations", 0.0) > 0:
        return "compression_needed"
    if record_number(state, "undetected_observations", 0.0) > 0:
        return "detection_needed"
    if record_number(state, "jetson_compressed_mb", 0.0) > 0:
        return "send_to_obc_needed"
    if record_number(state, "obc_data_mb", 0.0) > 0:
        return "downlink_needed"
    return "none"


def _battery_and_health(
    state: dict[str, Any], proposed_mode: str
) -> tuple[list[dict[str, str]], list[dict[str, str]], bool]:
    """Violations, warnings and whether the mode's own battery threshold blocks it."""

    violations: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    soc = record_number(state, "battery_soc", 0.5)
    health = str(state.get("health_status", "nominal"))
    hard_soc = record_number(state, "battery_min_soc", 0.20)
    if proposed_mode not in MODES:
        violations.append({"constraint": "mode", "reason": "Unknown EventSat mode."})
    if health != "nominal" and proposed_mode != "safe":
        violations.append(
            {"constraint": "anomaly", "reason": f"Anomaly active ({health}); safe is required."}
        )
    if soc <= hard_soc and proposed_mode != "safe":
        violations.append(
            {
                "constraint": "battery_critical",
                "reason": f"SoC {soc:.2f} is at/below the hard limit {hard_soc:.2f}.",
            }
        )
    threshold = (state.get("mode_constraints") or {}).get(proposed_mode, {})
    minimum = record_number(threshold, "min_battery_soc", 0.0)
    # Below a mode's own threshold the environment substitutes charging.
    below_threshold = soc > hard_soc and health == "nominal" and soc < minimum
    if below_threshold:
        violations.append(
            {
                "constraint": "mode_battery",
                "reason": f"SoC {soc:.2f} is below the {proposed_mode} threshold {minimum:.2f}.",
            }
        )
    if 0.20 < soc < 0.35 and proposed_mode not in {"charging", "safe"}:
        warnings.append(
            {"constraint": "battery_preferred", "reason": "SoC is below the preferred 0.35."}
        )
    return violations, warnings, below_threshold


def _mode_progress(
    state: dict[str, Any],
    proposed_mode: str,
    violations: list[dict[str, str]],
    warnings: list[dict[str, str]],
) -> bool:
    """Whether the mode advances the pipeline now, adding its own findings."""

    if proposed_mode == "communication":
        if _has_almanac(state):
            contact = bool(state.get("ground_pass_active", False))
            if not contact or record_number(state, "contact_window_seconds", 0.0) <= 0:
                violations.append(
                    {"constraint": "ground_pass", "reason": "No positive contact window is active."}
                )
        elif not state.get("station_visible", False):
            warnings.append(
                {
                    "constraint": "station_visibility",
                    "reason": "Station not visible now; communication only points the antenna.",
                }
            )
            return False
        return record_number(state, "obc_data_mb", 0.0) > 0
    if proposed_mode == "payload_observe":
        capacity = record_number(state, "jetson_capacity_mb", 249036.8)
        stored = record_number(state, "jetson_raw_mb", 0.0) + record_number(
            state, "jetson_compressed_mb", 0.0
        )
        fits = stored + record_number(state, "observation_size_mb", 9.41) <= capacity
        if not fits:
            violations.append(
                {"constraint": "jetson_capacity", "reason": "A complete product would not fit."}
            )
        return fits
    if proposed_mode == "payload_compress":
        return record_number(state, "uncompressed_observations", 0.0) >= 1
    if proposed_mode == "payload_detect":
        return record_number(state, "undetected_observations", 0.0) >= 1
    if proposed_mode == "payload_send":
        return record_number(state, "jetson_compressed_mb", 0.0) > 0
    return True


def check_constraints(state: dict[str, Any], proposed_mode: str = "charging") -> dict[str, Any]:
    """Validate a candidate against declared telemetry, without changing state."""

    violations, warnings, below_threshold = _battery_and_health(state, proposed_mode)
    soc = record_number(state, "battery_soc", 0.5)
    safety_forced = str(state.get("health_status", "nominal")) != "nominal" or soc <= (
        record_number(state, "battery_min_soc", 0.20)
    )
    settling = _settling(state, proposed_mode, safety_forced=safety_forced)
    if settling["command_ignored"]:
        # An ongoing slew keeps its initial target: the command is ignored, not queued,
        # and therefore neither a violation beyond an invalid mode nor productive.
        return {
            "proposed_mode": proposed_mode,
            "feasible": proposed_mode in MODES,
            "productive_this_step": False,
            **settling,
            "violations": [item for item in violations if item["constraint"] == "mode"],
            "warnings": [_settling_warning(settling)],
        }
    productive = _mode_progress(state, proposed_mode, violations, warnings)
    productive = productive and not below_threshold
    if settling["transition_steps_required"]:
        productive = False
        warnings.append(_settling_warning(settling))
    if not productive and proposed_mode not in {"charging", "safe"}:
        warnings.append(
            {"constraint": "pipeline", "reason": "The candidate makes no pipeline progress now."}
        )
    return {
        "proposed_mode": proposed_mode,
        "feasible": not violations,
        "productive_this_step": productive,
        **settling,
        "violations": violations,
        "warnings": warnings,
    }


def _settling(state: dict[str, Any], proposed_mode: str, *, safety_forced: bool) -> dict[str, Any]:
    """Attitude consequence of a command under the environment's slew rule.

    A slew fixes its target when it starts; commands while settling are ignored,
    and only environment-enforced safety preempts it.
    """

    settling_steps = int(record_number(state, "settling_time_steps", 0.0))
    remaining = int(record_number(state, "transition_steps_remaining", 0.0))
    target = str(state.get("previous_mode", "charging"))
    maneuver_modes = set(state.get("attitude_maneuver_modes") or ())
    active = not safety_forced and settling_steps > 0
    ignored = active and remaining > 0
    starts = (
        active
        and not ignored
        and proposed_mode != target
        and (proposed_mode in maneuver_modes or target in maneuver_modes)
    )
    return {
        "command_ignored": ignored,
        "transition_steps_required": remaining if ignored else settling_steps if starts else 0,
        "transition_target_mode": target if ignored else proposed_mode if starts else None,
    }


def _settling_warning(settling: dict[str, Any]) -> dict[str, str]:
    steps = settling["transition_steps_required"]
    target = settling["transition_target_mode"]
    if settling["command_ignored"]:
        reason = (
            f"Slew toward {target} in progress ({steps} settling step(s) left); this command "
            "is ignored and not queued; this step resolves to charging."
        )
    else:
        reason = (
            f"Request starts {steps} non-productive settling step(s) toward {target}; this "
            "step resolves to charging and later commands cannot retarget the slew."
        )
    return {"constraint": "attitude_settling", "reason": reason}


def _get_feasible_modes(state: dict[str, Any]) -> list[str]:
    return [mode for mode in MODES if check_constraints(state, mode)["feasible"]]


def evaluate_plan(state: dict[str, Any], proposed_mode: str = "charging") -> dict[str, Any]:
    """Score only incremental physically deliverable value, with no mode bonus.

    With the ground almanac, value is relative to pass capacity. In the onboard
    view, which has no pass forecast, it is relative to one step of link capacity.
    """

    constraints = check_constraints(state, proposed_mode)
    utility = 0.0
    progress_mb = 0.0
    if constraints["feasible"] and constraints["productive_this_step"]:
        link_mb = (
            record_number(state, "downlink_rate_kbps", 50.0)
            * record_number(state, "step_duration_s", 60.0)
        ) / 8000.0
        if proposed_mode == "communication":
            available = (
                record_number(state, "remaining_achievable_downlink_mb", 0.0)
                if _has_almanac(state)
                else link_mb
            )
            progress_mb = min(record_number(state, "obc_data_mb", 0.0), max(0.0, available))
            utility = progress_mb / max(available, 1e-12)
        elif proposed_mode == "payload_send":
            rate = record_number(state, "jetson_to_obc_rate_kbps", 8000.0)
            duration = record_number(state, "step_duration_s", 60.0)
            progress_mb = min(
                record_number(state, "jetson_compressed_mb", 0.0), rate * duration / 8000
            )
        elif proposed_mode == "payload_observe":
            ratio = max(record_number(state, "compression_ratio", 5.11), 1e-12)
            progress_mb = record_number(state, "observation_size_mb", 9.41) / ratio
        elif proposed_mode == "payload_compress":
            required = max(record_number(state, "compression_time_factor", 2.0), 1.0)
            progress_mb = (
                record_number(state, "observation_size_mb", 9.41)
                / max(record_number(state, "compression_ratio", 5.11), 1e-12)
                / required
            )
        elif proposed_mode == "payload_detect":
            progress_mb = record_number(state, "detection_metadata_mb", 0.01) / max(
                record_number(state, "detection_steps", 5.0), 1.0
            )
        # The onboard view has no pass capacity; one step of link capacity is the scale.
        reference = (
            max(
                record_number(state, "future_pass_capacity_mb", 0.0),
                record_number(state, "achievable_downlink_mb", 0.0),
            )
            if _has_almanac(state)
            else link_mb
        )
        if proposed_mode != "communication" and reference > 0:
            utility = min(progress_mb, reference) / reference
    risks = [item["reason"] for item in constraints["violations"] + constraints["warnings"]]
    return {
        "proposed_mode": proposed_mode,
        "estimated_utility": round(utility, 6),
        "pipeline_progress_mb": round(progress_mb, 6),
        "feasible": constraints["feasible"],
        "risk_factors": risks,
    }


def execute_tool(
    tool_name: str,
    args: dict[str, Any],
    state: dict[str, Any],
    memory: Any = None,
) -> dict[str, Any]:
    del memory
    proposed = str(args.get("proposed_mode", "charging"))
    if tool_name == "check_constraints":
        return check_constraints(state, proposed)
    if tool_name == "evaluate_plan":
        return evaluate_plan(state, proposed)
    return {"error": f"Unknown tool '{tool_name}'", "available": list(SCHEDULE_TOOL_NAMES)}


def get_tool_schemas(
    include_writable: bool = False,
    tool_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return the advertised what-if tools in deterministic order."""

    if include_writable:
        raise ValueError("Writable-memory tools are intentionally deferred")
    names = SCHEDULE_TOOL_NAMES if tool_names is None else tool_names
    return [_TOOLS[name].schema() for name in names if name in _TOOLS]
