from autops.llm.agentic_prompts import (
    AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    format_forced_schedule_prompt,
    format_schedule_planning_prompt,
)
from autops.llm.llm_prompts import (
    SCHEDULE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_schedule_prompt,
)
from autops.llm.onboard_prompts import (
    ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    ONBOARD_SCHEDULE_SYSTEM_PROMPT,
    format_onboard_schedule_prompt,
)
from autops.llm.tools import SCHEDULE_TOOL_NAMES, get_tool_schemas


def test_operational_prompt_invariants_are_preserved() -> None:
    for prompt in (
        SYSTEM_PROMPT,
        SCHEDULE_SYSTEM_PROMPT,
        AGENTIC_SCHEDULE_SYSTEM_PROMPT,
        ONBOARD_SCHEDULE_SYSTEM_PROMPT,
        ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT,
    ):
        assert "400 km SSO" in prompt
        assert "135" in prompt
        assert "Jetson" in prompt
    state = {"achievable_downlink_mb": 1.0}
    assert "50 kbps" in SYSTEM_PROMPT
    assert "50 kbps" in format_schedule_prompt(state, 92)
    assert "50 kbps" in format_schedule_planning_prompt(state, 92)
    assert "Plan-Tool-Reflect-Decide" in AGENTIC_SCHEDULE_SYSTEM_PROMPT
    assert "INTERNAL reasoning CONCISE" in AGENTIC_SCHEDULE_SYSTEM_PROMPT
    assert "at most three model turns" in ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT
    assert "INTERNAL reasoning CONCISE" in ONBOARD_AGENTIC_SCHEDULE_SYSTEM_PROMPT


def test_onboard_schedule_prompt_exposes_only_deterministic_lookahead() -> None:
    prompt = format_onboard_schedule_prompt(
        {
            "battery_soc": 0.6,
            "planning_contact_seconds": [0.0, 60.0, 0.0],
            "planning_sunlight": [True, False, True],
        },
        2,
    )

    assert "Contact-active offsets: [1]" in prompt
    assert "Sunlight offsets: [0, 2]" in prompt
    assert "PLAN NOW PLUS THE NEXT 2 HELD STEPS" in prompt


def test_agentic_registry_advertises_only_what_if_tools() -> None:
    assert SCHEDULE_TOOL_NAMES == ["check_constraints", "evaluate_plan"]
    assert [item["name"] for item in get_tool_schemas()] == SCHEDULE_TOOL_NAMES
    assert "check_constraints" in AGENTIC_SCHEDULE_SYSTEM_PROMPT
    assert "evaluate_plan" in AGENTIC_SCHEDULE_SYSTEM_PROMPT
    assert "check_battery" not in AGENTIC_SCHEDULE_SYSTEM_PROMPT


def test_forced_prompt_removes_tool_option() -> None:
    prompt = format_forced_schedule_prompt([], 92)
    assert "tool budget is exhausted" in prompt
    assert "tool calls are not available" in prompt
    assert '"decision"' in prompt
