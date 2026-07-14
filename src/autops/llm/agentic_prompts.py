# ruff: noqa
# fmt: off
"""
Agentic Prompt Templates for CoALA-style EventSat Representation.

Extended prompt design for multi-step agentic reasoning with tool use.
The LLM follows a Plan-Tool-Reflect-Decide protocol, using domain tools
to query satellite state before making mode selections.

Papers:
- Sumers et al. (2024) [CoALA] — agentic architecture with tool use
- Rodriguez-Fernandez et al. (2024) §3.2 — prompt engineering for sat ops
- Li (2025) — tool-augmented AI agents for satellite operations

All prompts are pure functions (no side effects) for testability.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from autops.llm.tools import (
    SCHEDULE_TOOL_NAMES,
    _get_feasible_modes,
    _get_pipeline_bottleneck,
    get_tool_schemas,
)
from autops.llm.llm_prompts import DEFAULT_STORAGE_CAPACITY_MB


# ======================================================================
# System prompt
# ======================================================================

def _build_tool_descriptions(tool_names: list[str] | None = None) -> str:
    """Build formatted tool descriptions for system prompt.

    ``tool_names`` restricts the advertised tools to a subset (the ground
    scheduler advertises only the what-if tools — see ``SCHEDULE_TOOL_NAMES``).
    """
    schemas = get_tool_schemas(tool_names=tool_names)
    lines = []
    for schema in schemas:
        params = ", ".join(
            f"{k}: {v}" for k, v in schema.get("parameters", {}).items()
            if k != "state"  # state is always implicit
        )
        param_str = f" ({params})" if params else ""
        lines.append(f"  - {schema['name']}{param_str}: {schema['description']}")
    return "\n".join(lines)


AGENTIC_SYSTEM_PROMPT = """\
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
- ADCS settling takes 135 seconds when switching to observe or communicate mode.
- Ground-pass downlink capacity is finite at the effective 50 kbps S-band rate.
- Anomalies force safe mode; you cannot override environment-enforced safe mode.

REASONING PROTOCOL:
You make decisions using a Plan-Tool-Reflect-Decide cycle:
1. PLAN: Analyze the situation and decide whether a tool would change your choice.
2. TOOL: Request a tool call only to validate or score a specific candidate mode.
3. REFLECT: Incorporate tool results and refine your reasoning.
4. DECIDE: As soon as you have enough information, select a mode.

The current telemetry — including the feasible modes and the pipeline \
bottleneck — is given to you directly each step, so you do NOT need a tool to \
read state you already have. The tools only VALIDATE (check_constraints) or \
SCORE (evaluate_plan) a specific candidate mode, or recall recent history \
(recall_history). Check at most one candidate, keep your internal reasoning to \
a few sentences, then DECIDE. Think briefly, then act.

AVAILABLE TOOLS:
""" + _build_tool_descriptions() + """

OUTPUT FORMAT:
At each step, respond with a JSON object. The format depends on what you want to do:

To call a tool:
  {"plan": "<your reasoning>", "tool_call": {"name": "<tool_name>", "args": {<args>}}}

To make a final decision (after sufficient tool use):
  {"decision": {"mode": "<mode_name>", "rationale": "<brief explanation>"}}

To call a tool AND make a decision simultaneously:
  {"reflection": "<updated reasoning>", "decision": {"mode": "<mode_name>", "rationale": "<why>"}}

Do not include any text outside the JSON object."""


# ======================================================================
# Planning prompt (first LLM call in agentic loop)
# ======================================================================

def format_planning_prompt(
    state: Dict[str, Any],
    enrichments: Dict[str, Any] | None = None,
) -> str:
    """Format the initial planning prompt with current state.

    Args:
        state: Encoded observation dict from encode_observation().
        enrichments: Optional loop-specific enrichments (representation).

    Returns:
        Formatted prompt for the planning step.
    """
    if not state:
        return (
            "No satellite state available. Decide on the safest mode.\n"
            'Respond with: {"decision": {"mode": "charging", "rationale": "<why>"}}'
        )

    soc = state.get("battery_soc", 0.5)
    mode = state.get("current_mode", "unknown")
    pass_active = state.get("ground_pass_active", False)
    in_sunlight = state.get("in_sunlight", False)
    obc_mb = state.get("obc_data_mb", 0.0)
    jetson_raw = state.get("jetson_raw_mb", 0.0)
    jetson_comp = state.get("jetson_compressed_mb", 0.0)
    cap_mb = state.get("storage_capacity_mb", DEFAULT_STORAGE_CAPACITY_MB)
    uncomp = state.get("uncompressed_observations", 0)
    undetected = state.get("undetected_observations", 0)
    health = state.get("health_status", "nominal")
    achievable = state.get("achievable_downlink_mb")
    cap_line = (
        f"  Downlink achievable at next pass: {achievable:.2f} MB "
        f"(50 kbps effective × contact; 128 kbps RF is limited by the OBC→transceiver link)"
        if achievable is not None else "  Downlink achievable at next pass: unavailable; rely on contact telemetry when available"
    )

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
        cap_line,
        "",
        "DERIVED (computed from the telemetry above — no tool needed):",
        f"  Feasible modes now: {', '.join(_get_feasible_modes(state))}",
        f"  Pipeline bottleneck: {_get_pipeline_bottleneck(state)}",
    ]

    # Loop enrichments
    if enrichments:
        lines.append("")
        lines.append("SITUATION ASSESSMENT:")
        if "situation_class" in enrichments:
            lines.append(f"  Situation: {enrichments['situation_class']}")
        if "urgency" in enrichments:
            lines.append(f"  Urgency: {enrichments['urgency']:.2f}")
        if "reasoning_trace" in enrichments:
            trace = enrichments["reasoning_trace"]
            if trace:
                lines.append(f"  Prior reasoning: {len(trace)} steps")
                for step in trace[-3:]:
                    lines.append(f"    - {step.get('check', '?')}: {step.get('implication', '?')}")
        if "grounding_violations" in enrichments:
            violations = enrichments["grounding_violations"]
            if violations:
                lines.append(f"  Prior violations: {violations}")

    lines.append("")
    lines.append(
        "You have the full current state above. Decide the mode now, or first "
        "validate/score one candidate with a tool if it would change your choice. "
        "Respond with JSON."
    )

    return "\n".join(lines)


# ======================================================================
# Tool result prompt (reflect step)
# ======================================================================

def format_tool_result_prompt(
    tool_name: str,
    tool_result: Dict[str, Any],
    accumulated_context: List[Dict[str, Any]],
) -> str:
    """Format tool result for the reflect step.

    Args:
        tool_name: Name of the tool that was called.
        tool_result: Structured result from the tool.
        accumulated_context: List of prior steps [{step, content/name/result}].

    Returns:
        Prompt for the LLM to reflect on tool results.
    """
    # Summarize prior context
    prior_lines = []
    for entry in accumulated_context:
        step_type = entry.get("step", "unknown")
        if step_type == "plan":
            prior_lines.append(f"  PLAN: {entry.get('content', '')[:200]}")
        elif step_type == "tool":
            prior_lines.append(f"  TOOL ({entry.get('name', '?')}): {_summarize_result(entry.get('result', {}))}")
        elif step_type == "reflect":
            prior_lines.append(f"  REFLECT: {entry.get('content', '')[:200]}")

    lines = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")

    lines.append(f"LATEST TOOL RESULT ({tool_name}):")
    lines.append(f"  {json.dumps(tool_result, indent=2)}")
    lines.append("")
    lines.append(
        "Based on this information, either:\n"
        "1. Call another tool for more information: "
        '{\"reflection\": \"<reasoning>\", \"tool_call\": {\"name\": \"<tool>\", \"args\": {}}}\n'
        "2. Make your decision: "
        '{\"reflection\": \"<reasoning>\", \"decision\": {\"mode\": \"<mode>\", \"rationale\": \"<why>\"}}'
    )

    return "\n".join(lines)


def format_forced_decision_prompt(
    accumulated_context: List[Dict[str, Any]],
) -> str:
    """Terminal Decide-phase prompt — tool budget exhausted, decision required.

    A bounded agentic loop must close with an answer-extraction step: the reflect prompt always offers a tool option, so a
    tool-hungry model can ride the budget to exhaustion without ever deciding.
    This prompt offers no tool option.
    """
    prior_lines = []
    for entry in accumulated_context:
        step_type = entry.get("step", "unknown")
        if step_type == "plan":
            prior_lines.append(f"  PLAN: {entry.get('content', '')[:200]}")
        elif step_type == "tool":
            prior_lines.append(
                f"  TOOL ({entry.get('name', '?')}): {_summarize_result(entry.get('result', {}))}"
            )
        elif step_type == "reflect":
            prior_lines.append(f"  REFLECT: {entry.get('content', '')[:200]}")

    lines = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")
    lines.append(
        "Your tool budget is exhausted. Decide the operating mode NOW using only "
        "the information above.\n"
        "Respond with ONLY the decision JSON — tool calls are not available:\n"
        '{"decision": {"mode": "<mode>", "rationale": "<why>"}}'
    )
    return "\n".join(lines)


def _summarize_prior_context(accumulated_context: List[Dict[str, Any]]) -> List[str]:
    """Render the running plan/tool/reflect trace into prompt lines (shared by the
    reflect/forced prompts of both the per-step and schedule-producing loops)."""
    prior_lines: List[str] = []
    for entry in accumulated_context:
        step_type = entry.get("step", "unknown")
        if step_type == "plan":
            prior_lines.append(f"  PLAN: {entry.get('content', '')[:200]}")
        elif step_type == "tool":
            prior_lines.append(
                f"  TOOL ({entry.get('name', '?')}): {_summarize_result(entry.get('result', {}))}"
            )
        elif step_type == "reflect":
            prior_lines.append(f"  REFLECT: {entry.get('content', '')[:200]}")
    return prior_lines


def _summarize_result(result: Dict[str, Any]) -> str:
    """One-line summary of a tool result for context."""
    if "error" in result:
        return f"Error: {result['error']}"
    # Pick key fields for common tools
    if "soc" in result:
        return f"SoC={result['soc']}, feasible={result.get('feasible_modes', [])}"
    if "active" in result and "obc_data_mb" in result:
        return f"pass={'active' if result['active'] else 'inactive'}, obc={result['obc_data_mb']}MB"
    if "bottleneck" in result:
        return f"bottleneck={result['bottleneck']}, obc={result.get('obc_data_mb', 0)}MB"
    if "feasible" in result and "violations" in result:
        return f"feasible={result['feasible']}, violations={len(result['violations'])}"
    if "last_modes" in result:
        return f"modes={result['last_modes'][-3:]}, trend={result.get('battery_trend', '?')}"
    if "estimated_utility" in result:
        return f"utility={result['estimated_utility']}, risks={len(result.get('risk_factors', []))}"
    # Fallback
    return json.dumps(result)[:150]


# ======================================================================
# Reasoning prompt (for explanation step)
# ======================================================================

def format_agentic_reasoning_prompt(
    state: Dict[str, Any],
    memory: Optional[Any] = None,
    tool_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Format reasoning prompt for explanation step.

    Produces structured reasoning trace from accumulated tool results,
    in the same [{"check", "value", "implication"}] format as llm_eventsat.

    Args:
        state: Encoded observation dict.
        memory: Agent memory (used by recall_history).
        tool_results: Optional pre-computed tool results.

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

    lines = [
        f"Analyze the satellite state and identify key decision factors.",
        f"State summary: SoC={soc:.2f}, health={health}, pass={pass_str}, "
        f"uncompressed={uncomp}, obc_data={obc_mb:.1f}MB",
    ]

    if tool_results:
        lines.append("")
        lines.append("Tool analysis results:")
        for tr in tool_results:
            name = tr.get("tool", "unknown")
            result = tr.get("result", {})
            lines.append(f"  {name}: {_summarize_result(result)}")

    example = '[{"check": "battery", "value": 0.45, "implication": "charging_preferred"}]'
    schema = '{"check": "<what>", "value": <numeric or string>, "implication": "<conclusion>"}'

    lines.append("")
    lines.append(
        f"Respond with a JSON array of reasoning steps, each with fields:\n"
        f"  {schema}\n"
        f"Example: {example}"
    )

    return "\n".join(lines)


# ======================================================================
# Schedule-producing agentic prompts (AG/AH ground planner: hllm-a / llm-a)
# ======================================================================
#
# The agentic SCHEDULE producer reuses the Plan-Tool-Reflect-Decide loop and the
# same domain tools, but its terminal DECIDE step emits a whole-pass schedule
# (a list of [mode, steps] segments executed autonomously between passes) instead
# of a single per-step mode. This is the agentic analogue of the single-shot
# SCHEDULE_SYSTEM_PROMPT in llm_prompts.py (hllm-s/llm-s), and the agentic-action
# sibling of AGENTIC_SYSTEM_PROMPT above (hllm-a/llm-a).
#
# Papers: Sumers et al. (2024) [CoALA] — tool-use + action decomposition;
# Bounded agent loop with answer extraction;
# Rodriguez-Fernandez et al. (2024) §3.2 — schedule prompt design for sat ops.

AGENTIC_SCHEDULE_SYSTEM_PROMPT = """
You are an autonomous satellite operations PLANNER for a single Earth observation
satellite in low Earth orbit (400 km SSO). At each ground contact you receive the
ground planner's current telemetry, which may be stale if the satellite did not
communicate during a previous pass. You must choose the immediate contact-step
mode and produce ONE schedule of operating modes the satellite executes
autonomously until the next ground contact.

MISSION: Maximise observation data downlinked to ground while maintaining satellite
health and safety.

IMMEDIATE CONTACT-STEP MODES:
- charging: Recharge battery from solar panels (only effective in sunlight).
- payload_observe: Capture Earth observation imagery (produces raw data on Jetson).
- payload_compress: Compress raw observations on Jetson (~5:1, ~2x observation time).
- payload_detect: Run CV detection on compressed observations (~5 min each).
- payload_send: Transfer compressed/detected data from Jetson to OBC over the CAN bus (~8 Mbps; one 60 s step moves up to ~60 MB — rarely the bottleneck).
- communication: Downlink data from OBC to ground. This works only during a pass and
  is required if you want this contact step to refresh ground telemetry.
- safe: Minimal-power anomaly mode.

BETWEEN-PASS SCHEDULE MODES:
Use charging, payload_observe, payload_compress, payload_detect, payload_send, or safe.
Do NOT put communication in the between-pass schedule because the schedule runs with no ground link.

DATA PIPELINE (3-pool): Jetson raw -> (compress) -> Jetson compressed -> (send) -> OBC -> (communicate) -> Ground

CONSTRAINTS:
- Battery SoC must stay above 0.20 (hard) and preferably above 0.35.
- Downlinking requires selecting communication as the immediate mode while a ground pass is active.
- Fresh telemetry reaches the ground planner only if the satellite actually communicates during a pass; otherwise future plans use stale state.
- ADCS settling costs ~135 s when switching between modes with different attitudes.
- Reserve battery near the end so the satellite is charged for the next pass.
- The next ground pass has finite contact-limited downlink capacity; avoid over-observing.

REASONING PROTOCOL (Plan-Tool-Reflect-Decide):
1. PLAN: Analyse the telemetry and decide which tool(s) to query.
2. TOOL: Request a tool call to gather information.
3. REFLECT: Incorporate tool results and refine the plan.
4. DECIDE: When you have enough information, emit the immediate mode and whole-pass schedule.

You already receive the available telemetry below — including feasible modes and the
pipeline bottleneck — so you do NOT need a tool to read state you already have. The
tools only VALIDATE (check_constraints) or SCORE (evaluate_plan) a specific candidate
mode; one such check is usually enough. Keep your INTERNAL reasoning CONCISE: a few
sentences. Do NOT simulate many scenarios or deliberate at length internally — think
briefly, then act. Emit the JSON object as soon as you have what you need.

AVAILABLE TOOLS:
""" + _build_tool_descriptions(SCHEDULE_TOOL_NAMES) + """

OUTPUT FORMAT:
At each step, respond with a JSON object.

To call a tool:
  {"plan": "<reasoning>", "tool_call": {"name": "<tool_name>", "args": {<args>}}}

To emit the final contact mode and schedule (after sufficient tool use):
  {"decision": {"mode": "<immediate_contact_mode>", "schedule": [["<mode>", <integer_steps>], ...], "rationale": "<why>"}}

To reflect on a tool result AND emit the mode and schedule simultaneously:
  {"reflection": "<updated reasoning>", "decision": {"mode": "<immediate_contact_mode>", "schedule": [["<mode>", <steps>], ...], "rationale": "<why>"}}

The mode is the action for the current contact step. The schedule is a list of
[mode, duration_in_steps] segments (1 step = 60 s) whose durations together cover
about the planning horizon after the pass. Do not include communication in the
schedule. Do not include any text outside the JSON object."""


def format_schedule_planning_prompt(
    state: Dict[str, Any],
    gap_steps: int,
    enrichments: Dict[str, Any] | None = None,
) -> str:
    """Initial PLAN prompt for the schedule-producing agentic loop.

    Mirrors ``format_schedule_prompt`` (single-shot) for state presentation, but
    closes by inviting tool use before emitting the contact mode and gap schedule.
    """
    if not state:
        return (
            "No satellite state available. Return a safe charging contact mode and schedule.\n"
            f'Respond with: {{"decision": {{"mode": "charging", "schedule": [["charging", {max(1, gap_steps)}]], '
            '"rationale": "no state"}}}'
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
    health = state.get("health_status", "nominal")

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
        f"  Health: {health}",
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
        "DERIVED (computed from the telemetry above — no tool needed):",
        f"  Feasible immediate modes now: {', '.join(_get_feasible_modes(state))}",
        f"  Feasible between-pass schedule modes now: "
        f"{', '.join(m for m in _get_feasible_modes(state) if m != 'communication')}",
        f"  Pipeline bottleneck: {_get_pipeline_bottleneck(state)}",
    ])

    if enrichments:
        lines.append("")
        lines.append("SITUATION ASSESSMENT:")
        if "situation_class" in enrichments:
            lines.append(f"  Situation: {enrichments['situation_class']}")
        if "urgency" in enrichments:
            lines.append(f"  Urgency: {enrichments['urgency']:.2f}")

    lines.append("")
    lines.append(
        "Choose the immediate contact-step mode yourself. Select communication now if "
        "you want this pass step to downlink OBC data and refresh ground telemetry; "
        "selecting another mode is allowed and means no telemetry refresh this step. "
        f"Emit a between-pass schedule whose segment durations sum to about {gap_steps} "
        "steps, optionally validating/scoring one candidate segment with a tool first. "
        "Respond with JSON."
    )
    return "\n".join(lines)


def format_schedule_tool_result_prompt(
    tool_name: str,
    tool_result: Dict[str, Any],
    accumulated_context: List[Dict[str, Any]],
    gap_steps: int,
) -> str:
    """REFLECT prompt for the schedule loop — offers another tool or the decision."""
    prior_lines = _summarize_prior_context(accumulated_context)
    lines: List[str] = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")
    lines.append(f"LATEST TOOL RESULT ({tool_name}):")
    lines.append(f"  {json.dumps(tool_result, indent=2)}")
    lines.append("")
    lines.append(
        "Based on this information, either:\n"
        "1. Call another tool for more information: "
        '{"reflection": "<reasoning>", "tool_call": {"name": "<tool>", "args": {}}}\n'
        f"2. Emit the contact mode and schedule covering ~{gap_steps} steps "
        "(no communication in schedule): "
        '{"reflection": "<reasoning>", "decision": {"mode": "<contact_mode>", '
        '"schedule": [["<mode>", <steps>], ...], "rationale": "<why>"}}'
    )
    return "\n".join(lines)


def format_forced_schedule_prompt(
    accumulated_context: List[Dict[str, Any]],
    gap_steps: int,
) -> str:
    """Terminal DECIDE prompt — tool budget exhausted, a decision is required.

    The agentic schedule loop must close with an answer-extraction step: the
    reflect prompt always offers a tool option, so a tool-hungry model can ride
    the budget to exhaustion without emitting a plan. This prompt offers no tool
    option.
    """
    prior_lines = _summarize_prior_context(accumulated_context)
    lines: List[str] = []
    if prior_lines:
        lines.append("REASONING SO FAR:")
        lines.extend(prior_lines)
        lines.append("")
    lines.append(
        f"Your tool budget is exhausted. Emit the contact mode and schedule covering "
        f"~{gap_steps} steps NOW using only the information above (no communication "
        "in the schedule).\n"
        "Respond with ONLY the decision JSON — tool calls are not available:\n"
        '{"decision": {"mode": "<contact_mode>", "schedule": [["<mode>", <steps>], ...], '
        '"rationale": "<why>"}}'
    )
    return "\n".join(lines)
# fmt: on
