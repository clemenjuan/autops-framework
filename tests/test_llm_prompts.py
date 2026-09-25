import pytest

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
from autops.llm.tools import (
    SCHEDULE_TOOL_NAMES,
    _get_feasible_modes,
    check_constraints,
    evaluate_plan,
    get_tool_schemas,
)


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


def test_onboard_schedule_prompt_reports_present_geometry_without_forecasts() -> None:
    prompt = format_onboard_schedule_prompt(
        {
            "battery_soc": 0.6,
            "station_visible": True,
            "in_sunlight": False,
            "navigation": {"valid": True, "station_elevation_deg": 23.44},
            "last_interval": {"executed_mode": "payload_send", "action_accepted": False},
        },
        2,
    )

    assert "PLAN NOW PLUS THE NEXT 2 HELD STEPS" in prompt
    assert "Ground station: visible now (elevation 23.4 deg)" in prompt
    assert "Sunlight now: no" in prompt
    assert "Last interval: executed payload_send, rejected" in prompt
    assert "offsets" not in prompt.lower() and "achievable" not in prompt.lower()
    assert "No ground-pass\nschedule or eclipse forecast" in ONBOARD_SCHEDULE_SYSTEM_PROMPT


def test_onboard_llm_prompt_is_invariant_to_hidden_future_tables() -> None:
    from copy import deepcopy

    from autops.config import expand_coordinate
    from autops.core.plugin import create_representation
    from autops.missions.eventsat.env import EventSatEnvironment

    config = deepcopy(expand_coordinate("eventsat/sas/ao/llm-s").mission_config)
    observation = EventSatEnvironment(config, max_steps=4, prefer_orekit=False).reset(7)
    changed = deepcopy(observation)
    changed["satellites"]["eventsat_0"]["metadata"].update(
        time_to_next_pass=1.0, remaining_achievable_downlink_mb=999.0
    )
    planner = create_representation("eventsat", "llm-s", "onboard", {"llm_replay": []})
    prompts = [
        format_onboard_schedule_prompt(planner.encode_observation(record), 2)
        for record in (observation, changed)
    ]
    assert prompts[0] == prompts[1]


def test_onboard_tools_permit_prepointing_while_ground_tools_require_a_pass() -> None:
    onboard = {"battery_soc": 0.8, "obc_data_mb": 2.0, "station_visible": False}
    ground = {**onboard, "ground_pass_active": False, "contact_window_seconds": 0.0}
    prepoint = check_constraints(onboard, "communication")
    assert prepoint["feasible"] and not prepoint["productive_this_step"]
    assert not check_constraints(ground, "communication")["feasible"]
    visible = evaluate_plan({**onboard, "station_visible": True}, "communication")
    assert visible["pipeline_progress_mb"] == pytest.approx(0.375)


def test_what_if_tools_follow_the_slew_rule() -> None:
    state = {
        "battery_soc": 0.8,
        "obc_data_mb": 2.0,
        "station_visible": True,
        "settling_time_steps": 2,
        "attitude_maneuver_modes": ("communication", "payload_observe"),
        "previous_mode": "charging",
    }
    starts = check_constraints(state, "payload_observe")
    assert starts["feasible"] and not starts["productive_this_step"]
    assert starts["transition_steps_required"] == 2
    assert starts["transition_target_mode"] == "payload_observe"
    settling = {**state, "transition_steps_remaining": 1, "previous_mode": "payload_observe"}
    ignored = check_constraints(settling, "communication")
    assert ignored["command_ignored"] and not ignored["productive_this_step"]
    assert ignored["transition_target_mode"] == "payload_observe"
    assert evaluate_plan(settling, "communication")["estimated_utility"] == 0.0
    assert not check_constraints({**settling, "battery_soc": 0.1}, "safe")["command_ignored"]
    arrived = check_constraints({**state, "previous_mode": "payload_observe"}, "payload_observe")
    assert arrived["transition_steps_required"] == 0 and arrived["productive_this_step"]


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


def test_what_if_tools_apply_each_mode_battery_threshold() -> None:
    state = {
        "battery_soc": 0.38,
        "previous_mode": "payload_observe",
        "mode_constraints": {"payload_observe": {"min_battery_soc": 0.4}},
    }
    # The environment would substitute charging below the observation threshold.
    below = check_constraints(state, "payload_observe")
    assert not below["feasible"] and not below["productive_this_step"]
    assert [item["constraint"] for item in below["violations"]] == ["mode_battery"]
    assert "payload_observe" not in _get_feasible_modes(state)
    above = check_constraints({**state, "battery_soc": 0.41}, "payload_observe")
    assert above["feasible"] and above["productive_this_step"]
