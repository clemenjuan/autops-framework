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
in low Earth orbit (400 km SSO). You receive fresh spacecraft telemetry: resources,
payload state, attitude settling, the outcome of the last interval, whether the ground
station is visible now, and whether the spacecraft is in sunlight now. No ground-pass
schedule or eclipse forecast is available onboard. Choose the immediate mode and a
short plan that the spacecraft will hold until the next onboard planning event.

MISSION: Maximise observation data downlinked to ground while maintaining satellite
health and safety.

AVAILABLE MODES:
- charging: Recharge from the solar panels (only effective in sunlight).
- payload_observe: Capture imagery into the Jetson raw-data pool.
- payload_compress: Compress Jetson raw data by about 5:1.
- payload_detect: Run CV detection on compressed observations (about 5 min each).
- payload_send: Transfer Jetson products to the OBC at about 8 Mbps.
- communication: Downlink OBC data at 50 kbps effective while the station is visible.
- safe: Minimal-power anomaly mode; the environment may enforce it.

DATA PIPELINE: Jetson raw -> Jetson compressed -> OBC -> ground.

CONSTRAINTS:
- Battery SoC must remain above 0.20 and preferably above 0.35.
- ADCS settling takes 135 s across modes with different attitudes.
- An ongoing slew keeps its initial target. Ordinary commands received during settling,
  including a policy-requested safe mode, are ignored and are not queued.
  Environment-enforced safe mode immediately aborts the slew. Once settling is complete,
  command the target mode again to begin productive operations.
- Communication transfers data only while the ground station is visible. Commanding it
  earlier only points the antenna, which also needs the 135 s settling.
- Passes occur only a few times per day and last a few minutes, so keep OBC data
  ready and use a visible pass whenever OBC data is waiting.
- Keep reasoning concise and do not invent telemetry.

OUTPUT FORMAT: JSON only:
  {"mode": "<immediate_mode>", "schedule": [["<later_mode>", <integer_steps>], ...],
   "rationale": "<brief explanation>"}
The immediate mode executes now. The schedule supplies subsequent held actions and
should cover the requested remaining plan steps. The schedule must never be empty:
if nothing else applies, use a single charging segment covering the remaining steps.
Keep the rationale to one or two concise sentences -- never think out loud inside it."""


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


def _station_line(state: dict[str, Any]) -> str:
    visible = "visible now" if state.get("station_visible", False) else "not visible now"
    navigation = state.get("navigation") or {}
    if not navigation.get("valid", False):
        return f"Ground station: {visible}"
    return f"Ground station: {visible} (elevation {navigation['station_elevation_deg']:.1f} deg)"


def _last_interval_line(state: dict[str, Any]) -> str:
    last = state.get("last_interval") or {}
    if not last:
        return "Last interval: none"
    outcome = "accepted" if last.get("action_accepted", True) else "rejected"
    return f"Last interval: executed {last.get('executed_mode', 'charging')}, {outcome}"


def format_onboard_schedule_prompt(state: dict[str, Any], remaining_steps: int) -> str:
    """Format fresh onboard telemetry for one planning event; no event forecast."""

    if not state:
        return (
            "No state is available. Return charging now and a charging schedule: "
            f'{{"mode":"charging","schedule":[["charging",{remaining_steps}]],'
            '"rationale":"no state"}'
        )
    feasible = ", ".join(_get_feasible_modes(state))
    settling = int(float(state.get("transition_steps_remaining", 0)))
    return "\n".join(
        [
            f"PLAN NOW PLUS THE NEXT {remaining_steps} HELD STEPS (60 s each).",
            f"Battery SoC: {float(state.get('battery_soc', 0.5)):.3f}",
            f"Health: {state.get('health_status', 'nominal')}",
            f"Current mode: {state.get('current_mode', 'charging')}",
            f"Attitude settling: {settling} steps remaining "
            f"(target {state.get('previous_mode', 'charging')})",
            _last_interval_line(state),
            _station_line(state),
            f"Sunlight now: {'yes' if state.get('in_sunlight', False) else 'no'}",
            f"Jetson raw: {float(state.get('jetson_raw_mb', 0.0)):.2f} MB",
            f"Jetson compressed: {float(state.get('jetson_compressed_mb', 0.0)):.2f} MB",
            f"OBC ready: {float(state.get('obc_data_mb', 0.0)):.2f} MB",
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
