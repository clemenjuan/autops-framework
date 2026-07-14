from __future__ import annotations

import json

import pytest

from autops.core.plugin import create_representation, registered_plugins
from autops.core.types import DecisionContext


def _state(**updates):
    state = {
        "battery_soc": 0.7,
        "battery_min_soc": 0.20,
        "health_status": "nominal",
        "ground_pass_active": True,
        "contact_window_seconds": 60.0,
        "current_mode": "charging",
        "in_sunlight": True,
        "obc_data_mb": 2.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "data_stored_mb": 2.0,
        "storage_capacity_mb": 100.0,
        "jetson_capacity_mb": 100.0,
        "uncompressed_observations": 0,
        "undetected_observations": 0,
        "achievable_downlink_mb": 4.0,
        "remaining_achievable_downlink_mb": 1.0,
        "future_pass_capacity_mb": 4.0,
        "estimated_gap_steps": 5,
    }
    state.update(updates)
    return state


def _context(state=None):
    value = state or _state()
    return DecisionContext(value, {}, None, 0, role="ground")


def _response(mode="communication", schedule=None):
    return json.dumps(
        {
            "mode": mode,
            "schedule": schedule or [["payload_send", 2], ["charging", 3]],
            "rationale": "replay plan",
        }
    )


def test_all_matrix_tokens_register_as_eventsat_ground_plugins() -> None:
    plugins = registered_plugins("eventsat")
    expected = {("eventsat", token, "ground") for token in ("llm-s", "llm-a", "hllm-s", "hllm-a")}
    assert expected <= plugins.keys()


def test_single_shot_replay_returns_new_paradigm_schedule_shape() -> None:
    planner = create_representation("eventsat", "llm-s", "ground", {"llm_replay": [_response()]})
    action = planner.select_action(_context())
    assert action["eventsat_0"]["mode"] == "communication"
    assert action["schedule"] == [
        {"mode": "payload_send", "steps": 2},
        {"mode": "charging", "steps": 3},
    ]
    assert "replay plan" in (planner.last_rationale or "")


def test_pure_and_hybrid_schedule_validation_are_distinct() -> None:
    raw = _response(schedule=[["communication", 2], ["payload_observe", 8]])
    pure = create_representation("eventsat", "llm-s", "ground", {"llm_replay": [raw]})
    hybrid = create_representation("eventsat", "hllm-s", "ground", {"llm_replay": [raw]})
    assert pure.select_action(_context())["schedule"] == [
        {"mode": "communication", "steps": 2},
        {"mode": "payload_observe", "steps": 8},
    ]
    assert hybrid.select_action(_context())["schedule"] == [{"mode": "payload_observe", "steps": 5}]


def test_hybrid_shield_replaces_unsafe_work_and_pads_gap() -> None:
    raw = _response(schedule=[["payload_observe", 2]])
    planner = create_representation("eventsat", "hllm-s", "ground", {"llm_replay": [raw]})
    action = planner.select_action(_context(_state(battery_soc=0.30)))
    assert action["schedule"] == [{"mode": "charging", "steps": 5}]
    assert planner.diagnostics()["grounding_overrides"] == 1.0


def test_agentic_loop_executes_only_what_if_tool_then_decides() -> None:
    replay = [
        json.dumps(
            {
                "plan": "validate observe",
                "tool_call": {
                    "name": "check_constraints",
                    "args": {"proposed_mode": "payload_observe"},
                },
            }
        ),
        json.dumps(
            {
                "reflection": "feasible",
                "decision": {
                    "mode": "communication",
                    "schedule": [["payload_observe", 2], ["charging", 3]],
                    "rationale": "validated",
                },
            }
        ),
    ]
    planner = create_representation("eventsat", "hllm-a", "ground", {"llm_replay": replay})
    action = planner.select_action(_context())
    assert sum(block["steps"] for block in action["schedule"]) == 5
    assert planner.diagnostics()["agentic_tool_calls"] == 1.0
    assert planner.diagnostics()["llm_calls"] == 2.0
    assert "check_constraints" in (planner.last_rationale or "")


def test_agentic_budget_forces_answer_extraction_after_three_turns() -> None:
    tool = {
        "reflection": "check again",
        "tool_call": {"name": "evaluate_plan", "args": {"proposed_mode": "charging"}},
    }
    replay = [json.dumps({"plan": "start", **tool}), json.dumps(tool), json.dumps(tool)]
    replay.append(
        json.dumps(
            {
                "decision": {
                    "mode": "communication",
                    "schedule": [["charging", 5]],
                    "rationale": "forced",
                }
            }
        )
    )
    planner = create_representation(
        "eventsat", "llm-a", "ground", {"llm_replay": replay, "max_agentic_steps": 3}
    )
    planner.select_action(_context())
    diagnostics = planner.diagnostics()
    assert diagnostics["llm_calls"] == 4.0
    assert diagnostics["agentic_tool_calls"] == 2.0


def test_invalid_outputs_fail_instead_of_becoming_symbolic_runs() -> None:
    invalid = json.dumps({"mode": "warp", "schedule": [["charging", 5]]})
    planner = create_representation("eventsat", "llm-s", "ground", {"llm_replay": [invalid] * 3})
    with pytest.raises(RuntimeError, match="substrate integrity"):
        planner.select_action(_context())


def test_markdown_fence_is_tolerated_without_relaxing_schema() -> None:
    fenced = f"```json\n{_response()}\n```"
    planner = create_representation("eventsat", "llm-s", "ground", {"llm_replay": [fenced]})
    assert planner.select_action(_context())["eventsat_0"]["mode"] == "communication"
