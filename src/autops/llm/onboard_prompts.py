"""Plan-and-hold prompts for EventSat onboard LLM scheduling."""

from __future__ import annotations

from typing import Any

from autops.llm.tools import (
    SCHEDULE_TOOL_NAMES,
    _get_feasible_modes,
    _get_pipeline_bottleneck,
)

ONBOARD_SCHEDULE_SYSTEM_PROMPT = """\
You are the autonomous onboard scheduler for a single Earth observation satellite
in low Earth orbit (400 km SSO). You receive fresh spacecraft telemetry and a
deterministic orbital almanac. Choose the immediate mode and a short plan that the
spacecraft will hold until the next onboard planning event.

MISSION: Maximise observation data downlinked to ground while maintaining satellite
health and safety.

AVAILABLE MODES:
- charging: Recharge from the solar panels (only effective in sunlight).
- payload_observe: Capture imagery into the Jetson raw-data pool.
- payload_compress: Compress Jetson raw data by about 5:1.
- payload_detect: Run CV detection on compressed observations (about 5 min each).
- payload_send: Transfer Jetson products to the OBC at about 8 Mbps.
- communication: Downlink OBC data at 50 kbps effective, only during a ground pass.
- safe: Minimal-power anomaly mode; the environment may enforce it.

DATA PIPELINE: Jetson raw -> Jetson compressed -> OBC -> ground.

CONSTRAINTS:
- Battery SoC must remain above 0.20 and preferably above 0.35.
- ADCS settling takes 135 s across modes with different attitudes.
- Put communication only where the contact lookahead reports a ground pass.
- Prepare OBC data before a pass and respect finite pass downlink capacity.
- Keep reasoning concise and do not invent telemetry.

OUTPUT FORMAT: JSON only:
  {"mode": "<immediate_mode>", "schedule": [["<later_mode>", <integer_steps>], ...],
   "rationale": "<brief explanation>"}
The immediate mode executes now. The schedule supplies subsequent held actions and
should cover the requested remaining plan steps."""


ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT = (
    ONBOARD_SCHEDULE_SYSTEM_PROMPT
    + """

Use a bounded Plan-Tool-Reflect-Decide loop. The only tools are
check_constraints and evaluate_plan; they perform what-if checks and do not reveal
new telemetry. Use at most three model turns and keep INTERNAL reasoning CONCISE.
At most one check is usually sufficient. Emit either:
  {"plan": "<brief>", "tool_call": {"name": "<tool>", "args": {}}}
or a final object under "decision" using the schedule schema above. Do not include
text outside the JSON object."""
)


def _active_offsets(values: Any, *, limit: int) -> list[int]:
    if not isinstance(values, (list, tuple)):
        return []
    return [index for index, value in enumerate(values[:limit]) if bool(value)]


def format_onboard_schedule_prompt(state: dict[str, Any], remaining_steps: int) -> str:
    """Format fresh telemetry and almanac data for one onboard planning event."""

    if not state:
        return (
            "No state is available. Return charging now and a charging schedule: "
            f'{{"mode":"charging","schedule":[["charging",{remaining_steps}]],'
            '"rationale":"no state"}'
        )
    horizon = max(1, remaining_steps + 1)
    contacts = _active_offsets(state.get("planning_contact_seconds"), limit=horizon)
    sunlight = _active_offsets(state.get("planning_sunlight"), limit=horizon)
    feasible = ", ".join(_get_feasible_modes(state))
    return "\n".join(
        [
            f"PLAN NOW PLUS THE NEXT {remaining_steps} HELD STEPS (60 s each).",
            "Offsets use 0 for the immediate action.",
            f"Battery SoC: {float(state.get('battery_soc', 0.5)):.3f}",
            f"Health: {state.get('health_status', 'nominal')}",
            f"Current mode: {state.get('current_mode', 'charging')}",
            f"Contact-active offsets: {contacts or 'none'}",
            f"Sunlight offsets: {sunlight or 'none'}",
            f"Jetson raw: {float(state.get('jetson_raw_mb', 0.0)):.2f} MB",
            f"Jetson compressed: {float(state.get('jetson_compressed_mb', 0.0)):.2f} MB",
            f"OBC ready: {float(state.get('obc_data_mb', 0.0)):.2f} MB",
            "Next-pass achievable downlink: "
            f"{float(state.get('achievable_downlink_mb', 0.0)):.2f} MB",
            f"Feasible immediate modes: {feasible}",
            f"Pipeline bottleneck: {_get_pipeline_bottleneck(state)}",
            "Return the immediate mode and subsequent schedule as JSON.",
        ]
    )


def format_onboard_tool_result_prompt(
    tool_name: str,
    tool_result: dict[str, Any],
    accumulated_context: list[dict[str, Any]],
    remaining_steps: int,
) -> str:
    """Request a bounded reflection or final onboard schedule."""

    prior = [str(item.get("content", "")) for item in accumulated_context if item.get("content")]
    return (
        f"Prior reasoning: {prior[-2:] or 'none'}. Tool result ({tool_name}): {tool_result}. "
        "Reflect briefly, then emit the final "
        f"immediate mode and {remaining_steps}-step held schedule under `decision`. "
        f"Only these tools exist: {', '.join(SCHEDULE_TOOL_NAMES)}. Respond with JSON."
    )


def format_forced_onboard_schedule_prompt(remaining_steps: int) -> str:
    """Close an agentic event after the fixed tool budget."""

    return (
        "Your tool budget is exhausted. Tool calls are not available. Emit only the "
        f"decision JSON with an immediate mode and {remaining_steps}-step held schedule."
    )


__all__ = [
    "ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT",
    "ONBOARD_SCHEDULE_SYSTEM_PROMPT",
    "format_forced_onboard_schedule_prompt",
    "format_onboard_schedule_prompt",
    "format_onboard_tool_result_prompt",
]
