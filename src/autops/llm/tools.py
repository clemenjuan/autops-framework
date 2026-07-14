"""Side-effect-free what-if tools for the bounded EventSat agentic planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autops.missions.eventsat.physics import MODES


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


def _number(state: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


def _get_pipeline_bottleneck(state: dict[str, Any]) -> str:
    if _number(state, "uncompressed_observations", 0.0) > 0:
        return "compression_needed"
    if _number(state, "undetected_observations", 0.0) > 0:
        return "detection_needed"
    if _number(state, "jetson_compressed_mb", 0.0) > 0:
        return "send_to_obc_needed"
    if _number(state, "obc_data_mb", 0.0) > 0:
        return "downlink_needed"
    return "none"


def check_constraints(
    state: dict[str, Any], proposed_mode: str = "charging"
) -> dict[str, Any]:
    """Validate a candidate against declared telemetry, without changing state."""

    violations: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    soc = _number(state, "battery_soc", 0.5)
    health = str(state.get("health_status", "nominal"))
    hard_soc = _number(state, "battery_min_soc", 0.20)
    productive = True

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
    if 0.20 < soc < 0.35 and proposed_mode not in {"charging", "safe"}:
        warnings.append(
            {"constraint": "battery_preferred", "reason": "SoC is below the preferred 0.35."}
        )

    if proposed_mode == "communication":
        contact = bool(state.get("ground_pass_active", False))
        contact_s = _number(state, "contact_window_seconds", 0.0)
        if not contact or contact_s <= 0:
            violations.append(
                {"constraint": "ground_pass", "reason": "No positive contact window is active."}
            )
        productive = _number(state, "obc_data_mb", 0.0) > 0
    elif proposed_mode == "payload_observe":
        capacity = _number(state, "jetson_capacity_mb", 249036.8)
        stored = _number(state, "jetson_raw_mb", 0.0) + _number(
            state, "jetson_compressed_mb", 0.0
        )
        productive = stored + _number(state, "observation_size_mb", 9.41) <= capacity
        if not productive:
            violations.append(
                {"constraint": "jetson_capacity", "reason": "A complete product would not fit."}
            )
    elif proposed_mode == "payload_compress":
        productive = _number(state, "uncompressed_observations", 0.0) >= 1
    elif proposed_mode == "payload_detect":
        productive = _number(state, "undetected_observations", 0.0) >= 1
    elif proposed_mode == "payload_send":
        productive = _number(state, "jetson_compressed_mb", 0.0) > 0

    if not productive and proposed_mode not in {"charging", "safe"}:
        warnings.append(
            {"constraint": "pipeline", "reason": "The candidate makes no pipeline progress now."}
        )
    return {
        "proposed_mode": proposed_mode,
        "feasible": not violations,
        "productive_this_step": productive,
        "violations": violations,
        "warnings": warnings,
    }


def _get_feasible_modes(state: dict[str, Any]) -> list[str]:
    return [mode for mode in MODES if check_constraints(state, mode)["feasible"]]


def evaluate_plan(
    state: dict[str, Any], proposed_mode: str = "charging"
) -> dict[str, Any]:
    """Score only incremental physically deliverable value, with no mode bonus."""

    constraints = check_constraints(state, proposed_mode)
    utility = 0.0
    progress_mb = 0.0
    if constraints["feasible"] and constraints["productive_this_step"]:
        if proposed_mode == "communication":
            available = _number(state, "remaining_achievable_downlink_mb", 0.0)
            progress_mb = min(_number(state, "obc_data_mb", 0.0), max(0.0, available))
            utility = progress_mb / max(available, 1e-12)
        elif proposed_mode == "payload_send":
            rate = _number(state, "jetson_to_obc_rate_kbps", 8000.0)
            duration = _number(state, "step_duration_s", 60.0)
            progress_mb = min(_number(state, "jetson_compressed_mb", 0.0), rate * duration / 8000)
        elif proposed_mode == "payload_observe":
            ratio = max(_number(state, "compression_ratio", 5.11), 1e-12)
            progress_mb = _number(state, "observation_size_mb", 9.41) / ratio
        elif proposed_mode == "payload_compress":
            required = max(_number(state, "compression_time_factor", 2.0), 1.0)
            progress_mb = (
                _number(state, "observation_size_mb", 9.41)
                / max(_number(state, "compression_ratio", 5.11), 1e-12)
                / required
            )
        elif proposed_mode == "payload_detect":
            progress_mb = _number(state, "detection_metadata_mb", 0.01) / max(
                _number(state, "detection_steps", 5.0), 1.0
            )
        future_capacity = max(
            _number(state, "future_pass_capacity_mb", 0.0),
            _number(state, "achievable_downlink_mb", 0.0),
        )
        if proposed_mode != "communication" and future_capacity > 0:
            utility = min(progress_mb, future_capacity) / future_capacity
    risks = [
        item["reason"]
        for item in constraints["violations"] + constraints["warnings"]
    ]
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
