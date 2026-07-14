# ruff: noqa
# fmt: off
"""
LLM Prompt Templates for EventSat.

Structured prompt design for satellite mode selection following
Rodriguez-Fernandez et al. (2024) §3.2 prompt engineering patterns
for spacecraft operations.

All prompts are pure functions (no side effects) for testability.
"""
from __future__ import annotations

from typing import Any, Dict

DEFAULT_STORAGE_CAPACITY_MB = 4096.0


# ======================================================================
# System prompt
# ======================================================================

SYSTEM_PROMPT = """\
You are an autonomous satellite operations agent managing a single Earth \
observation satellite in low Earth orbit (400 km SSO).

MISSION: Maximise observation data downlinked to ground while maintaining \
satellite health and safety.

AVAILABLE MODES (exactly one per timestep):
- charging: Recharge battery from solar panels (only effective in sunlight).
- payload_observe: Capture Earth observation imagery (consumes power, produces raw data on Jetson).
- payload_compress: Compress raw observations on Jetson (reduces size ~5:1, takes ~2x observation time).
- payload_detect: Run CV detection on compressed observations (5 min per observation).
- payload_send: Transfer compressed/detected data from Jetson to OBC over the CAN bus (~8 Mbps; one 60 s step moves up to ~60 MB — rarely the bottleneck).
- communication: Downlink data from OBC to ground station during a ground pass.
- safe: Minimal power mode for anomaly recovery (environment may force this).

DATA PIPELINE (3-pool):
  Jetson raw → (compress) → Jetson compressed → (send) → OBC → (communicate) → Ground

CONSTRAINTS:
- Battery SoC must stay above 0.20 (hard limit) and above 0.35 (preferred).
- Ground passes are limited windows; OBC data must be ready before pass starts.
- ADCS settling takes 135 seconds when switching between modes with different attitudes.
- Ground-pass downlink capacity is finite at the effective 50 kbps S-band rate.
- Anomalies force safe mode; you cannot override environment-enforced safe mode.

OUTPUT FORMAT: Respond with a JSON object containing exactly two fields:
  {"mode": "<mode_name>", "rationale": "<brief explanation>"}
Do not include any other text outside the JSON object."""


# ======================================================================
# State formatter
# ======================================================================

def format_state_prompt(state: Dict[str, Any], enrichments: Dict[str, Any] | None = None) -> str:
    """Format satellite state into a structured user prompt.

    Args:
        state: Encoded observation dict from encode_observation().
        enrichments: Optional loop-specific enrichments (representation).

    Returns:
        Formatted prompt string for the LLM.
    """
    if not state:
        return "No satellite state available. Select the safest mode."

    soc = state.get("battery_soc", 0.5)
    mode = state.get("current_mode", "unknown")
    pass_active = state.get("ground_pass_active", False)
    in_sunlight = state.get("in_sunlight", False)
    obc_mb = state.get("obc_data_mb", 0.0)
    jetson_raw = state.get("jetson_raw_mb", 0.0)
    jetson_comp = state.get("jetson_compressed_mb", 0.0)
    data_mb = state.get("data_stored_mb", 0.0)
    cap_mb = state.get("storage_capacity_mb", DEFAULT_STORAGE_CAPACITY_MB)
    uncomp = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    health = state.get("health_status", "nominal")
    achievable = state.get("achievable_downlink_mb")

    lines = [
        "CURRENT SATELLITE STATE:",
        f"  Battery SoC: {soc:.2f} (sunlight: {'yes' if in_sunlight else 'no'})",
        f"  Current mode: {mode}",
        f"  Health: {health}",
        f"  Ground pass active: {'YES' if pass_active else 'no'}",
        "",
        "DATA PIPELINE:",
        f"  Jetson raw: {jetson_raw:.2f} MB ({uncomp} uncompressed obs)",
        f"  Jetson compressed: {jetson_comp:.2f} MB ({undetected} undetected obs)",
        f"  OBC ready for downlink: {obc_mb:.2f} / {cap_mb:.0f} MB",
        f"  Total stored: {data_mb:.1f} / {cap_mb:.0f} MB ({data_mb/cap_mb*100:.0f}%)",
    ]
    if achievable is not None:
        lines.append(f"  Downlink achievable at next pass: {achievable:.2f} MB "
                     f"(50 kbps effective × contact; 128 kbps RF is limited by the "
                     f"OBC→transceiver link) — observing beyond this just fills storage")
    else:
        lines.append("  Downlink achievable at next pass: unavailable; rely on contact telemetry when available")

    # Add loop-specific enrichments
    if enrichments:
        lines.append("")
        lines.append("SITUATION ASSESSMENT:")
        if "situation_class" in enrichments:
            lines.append(f"  Situation: {enrichments['situation_class']}")
        if "urgency" in enrichments:
            lines.append(f"  Urgency: {enrichments['urgency']:.2f}")
        if "battery_trending_down" in enrichments:
            trend = "declining" if enrichments["battery_trending_down"] else "stable/rising"
            lines.append(f"  Battery trend: {trend}")
        if "entered_eclipse" in enrichments:
            lines.append(f"  Eclipse transition: {'entering eclipse' if enrichments['entered_eclipse'] else 'no'}")
        if "reasoning_steps" in enrichments:
            steps = enrichments["reasoning_steps"]
            if steps:
                lines.append(f"  Prior reasoning steps: {len(steps)}")
                for step in steps[-3:]:  # Show last 3
                    lines.append(f"    - {step.get('check', '?')}: {step.get('implication', '?')}")

    lines.append("")
    lines.append("Select the optimal mode. Respond with JSON: {\"mode\": \"<mode>\", \"rationale\": \"<why>\"}")

    return "\n".join(lines)


# ======================================================================
# Reasoning prompt (for explanation step)
# ======================================================================

def format_reasoning_prompt(state: Dict[str, Any], memory: Any) -> str:
    """Format a reasoning prompt for the explanation step.

    Args:
        state: Encoded observation dict.
        memory: Agent memory (currently unused, reserved for future).

    Returns:
        Prompt asking the LLM to produce structured reasoning steps.
    """
    if not state:
        return "No state available. List the key factors for choosing safe mode."

    soc = state.get("battery_soc", 0.5)
    health = state.get("health_status", "nominal")
    pass_active = state.get("ground_pass_active", False)
    uncomp = state.get("uncompressed_observations", 0)
    obc_mb = state.get("obc_data_mb", 0.0)

    pass_str = "active" if pass_active else "inactive"
    example = '[{"check": "battery", "value": 0.45, "implication": "charging_preferred"}]'
    schema = '{"check": "<what you checked>", "value": <numeric or string>, "implication": "<conclusion>"}'

    return (
        f"Analyze the current satellite state and identify key decision factors.\n"
        f"State summary: SoC={soc:.2f}, health={health}, pass={pass_str}, "
        f"uncompressed={uncomp}, obc_data={obc_mb:.1f}MB\n\n"
        f"Respond with a JSON array of reasoning steps, each with fields:\n"
        f"  {schema}\n"
        f"Example: {example}"
    )


# ======================================================================
# Schedule-planning prompt (single-shot LLM ground planner — hllm-s)
# ======================================================================

SCHEDULE_SYSTEM_PROMPT = """\
You are an autonomous satellite operations planner for a single Earth observation satellite in low Earth orbit (400 km SSO). At each ground contact you receive the ground planner's current telemetry, which may be stale if the satellite did not communicate during a previous pass. You must choose the immediate contact-step mode and produce ONE schedule of operating modes for the satellite to execute autonomously until the next ground contact.

MISSION: Maximise observation data downlinked to ground while maintaining satellite health and safety.

AVAILABLE MODES:
- charging: Recharge battery from solar panels (only effective in sunlight).
- payload_observe: Capture Earth observation imagery (produces raw data on Jetson).
- payload_compress: Compress raw observations on Jetson (~5:1, ~2x observation time).
- payload_detect: Run CV detection on compressed observations (~5 min each).
- payload_send: Transfer compressed/detected data from Jetson to OBC over the CAN bus (~8 Mbps; one 60 s step moves up to ~60 MB — rarely the bottleneck).
- communication: Downlink data from OBC to ground (only works as the immediate mode during a pass).
- safe: Minimal-power anomaly mode.

DATA PIPELINE (3-pool): Jetson raw -> (compress) -> Jetson compressed -> (send) -> OBC -> (communicate) -> Ground

CONSTRAINTS:
- Battery SoC must stay above 0.20 (hard) and preferably above 0.35.
- You must output an immediate contact-step mode. Downlinking requires selecting communication while a ground pass is active.
- Fresh telemetry reaches the ground planner only if the satellite actually communicates during a pass; otherwise future plans use stale state.
- The schedule runs BETWEEN passes (no ground link), so do not put communication in the between-pass schedule.
- ADCS settling costs ~135 s when switching between modes with different attitudes.
- Reserve battery near the end so the satellite is charged for the next pass.
- The next ground pass has finite contact-limited downlink capacity; avoid over-observing.

OUTPUT FORMAT: a JSON object with exactly:
  {"mode": "<immediate_contact_mode>", "schedule": [["<mode>", <integer_steps>], ...], "rationale": "<brief explanation>"}
The mode is the action for the current contact step. The schedule is a list of [mode, duration_in_steps] segments (1 step = 60 s) that together should cover about N steps after the pass. Use only the modes above. Output JSON only."""


def format_schedule_prompt(state: Dict[str, Any], gap_steps: int) -> str:
    """Format a schedule-planning prompt: plan ~gap_steps until the next pass."""
    if not state:
        return (
            "No satellite state available. Return a safe charging contact mode and schedule: "
            '{"mode": "charging", "schedule": [["charging", %d]], "rationale": "no state"}'
            % max(1, gap_steps)
        )

    soc = state.get("battery_soc", 0.5)
    in_sunlight = state.get("in_sunlight", False)
    pass_active = state.get("ground_pass_active", False)
    staleness = state.get("staleness_steps", 0)
    time_to_next = state.get("time_to_next_pass")
    remaining_pass = state.get("remaining_pass_duration")
    following_gap = state.get("following_gap_steps")
    obc_mb = state.get("obc_data_mb", 0.0)
    jetson_raw = state.get("jetson_raw_mb", 0.0)
    jetson_comp = state.get("jetson_compressed_mb", 0.0)
    cap_mb = state.get("storage_capacity_mb", DEFAULT_STORAGE_CAPACITY_MB)
    uncomp = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    achievable = state.get("achievable_downlink_mb")

    cap_line = (
        f"  Downlink achievable at next pass: {achievable:.2f} MB "
        f"(50 kbps effective × contact; 128 kbps RF is limited by the OBC→transceiver link) "
        f"— observing more than this just fills storage you cannot deliver"
        if achievable is not None else "  Downlink achievable at next pass: unavailable; rely on contact telemetry when available"
    )

    lines = [
        f"PLAN THE NEXT {gap_steps} STEPS (1 step = 60 s) until the next ground contact.",
        "",
        "CURRENT STATE (ground telemetry available to planner):",
        f"  Battery SoC: {soc:.2f} (sunlight: {'yes' if in_sunlight else 'no'})",
        f"  Ground pass active now: {'YES' if pass_active else 'no'}",
        f"  Telemetry staleness: {staleness} steps since last successful downlink",
        f"  Jetson raw: {jetson_raw:.2f} MB ({uncomp} uncompressed obs)",
        f"  Jetson compressed: {jetson_comp:.2f} MB ({undetected} undetected obs)",
        f"  OBC ready for downlink: {obc_mb:.2f} / {cap_mb:.0f} MB",
        cap_line,
    ]
    timing = []
    if time_to_next is not None:
        timing.append(f"time_to_next_pass={time_to_next} steps")
    if remaining_pass is not None:
        timing.append(f"remaining_pass_duration={remaining_pass} steps")
    if following_gap is not None:
        timing.append(f"following_gap_steps={following_gap} steps")
    if timing:
        lines.append(f"  Contact timing: {', '.join(timing)}")
    lines.extend([
        "",
        "Choose the immediate contact-step mode yourself. Select communication now if "
        "you want this pass step to downlink OBC data and refresh ground telemetry; "
        "selecting another mode is allowed and means no telemetry refresh this step.",
        f"Produce a between-pass schedule whose segment durations sum to about {gap_steps} steps. "
        'Respond with JSON: {"mode": "<contact_mode>", "schedule": [["<mode>", <steps>], ...], "rationale": "<why>"}',
    ])
    return "\n".join(lines)
# fmt: on
