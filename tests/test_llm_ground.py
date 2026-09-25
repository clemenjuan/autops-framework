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


def test_all_llm_tokens_register_as_eventsat_onboard_plugins() -> None:
    plugins = registered_plugins("eventsat")
    expected = {("eventsat", token, "onboard") for token in ("llm-s", "llm-a", "hllm-s", "hllm-a")}
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


@pytest.mark.parametrize("role", ["ground", "onboard"])
def test_agentic_follow_ups_keep_telemetry_and_tool_evidence(role) -> None:
    tool = json.dumps(
        {
            "plan": "check",
            "tool_call": {"name": "check_constraints", "args": {"proposed_mode": "charging"}},
        }
    )
    planner = create_representation(
        "eventsat", "llm-a", role, {"llm_replay": [tool, tool, tool, _response("charging")]}
    )
    prompts: list[str] = []
    generate = planner.client.generate

    def recording(system_prompt, user_prompt, **kwargs):
        prompts.append(user_prompt)
        return generate(system_prompt, user_prompt, **kwargs)

    planner.client.generate = recording
    planner.select_action(DecisionContext(_state(battery_soc=0.61), {}, None, 0, role=role))

    # Three model turns run two tools; the forced decision must still see both results.
    assert len(prompts) == 4
    assert all("Battery SoC: 0.61" in prompt for prompt in prompts)
    assert prompts[-1].count("check_constraints") == 2


@pytest.mark.parametrize("token", ["llm-s", "llm-a", "hllm-s", "hllm-a"])
def test_deterministic_mock_plans_every_onboard_substrate(token) -> None:
    planner = create_representation(
        "eventsat", token, "onboard", {"llm_mock": True, "plan_hold": 3}
    )
    action = planner.select_action(DecisionContext(_state(), {}, None, 0, role="onboard"))
    assert action["eventsat_0"]["mode"] in {"charging", "communication"}
    assert planner.diagnostics()["llm_calls"] == 1.0


def test_onboard_single_shot_holds_schedule_without_extra_inference() -> None:
    replay = [_response("communication", [["payload_send", 2]]), _response("charging")]
    planner = create_representation(
        "eventsat", "llm-s", "onboard", {"llm_replay": replay, "plan_hold": 3}
    )
    first = planner.select_action(_context())
    second = planner.select_action(_context())
    third = planner.select_action(_context())
    fourth = planner.select_action(_context())

    assert first["eventsat_0"]["mode"] == "communication"
    assert first["eventsat_0"]["jetson_planned"] is True
    assert [second["eventsat_0"]["mode"], third["eventsat_0"]["mode"]] == [
        "payload_send",
        "payload_send",
    ]
    assert not second["eventsat_0"]["jetson_planned"]
    assert not third["eventsat_0"]["jetson_planned"]
    assert fourth["eventsat_0"]["jetson_planned"] is True
    assert planner.diagnostics()["planning_events"] == 2
    assert planner.diagnostics()["held_action_steps"] == 2
    assert planner.diagnostics()["llm_calls"] == 2.0


def test_hybrid_onboard_rechecks_each_held_action_against_fresh_telemetry() -> None:
    planner = create_representation(
        "eventsat",
        "hllm-s",
        "onboard",
        {"llm_replay": [_response("charging", [["payload_observe", 2]])], "plan_hold": 3},
    )
    planner.select_action(_context())
    held = planner.select_action(_context(_state(battery_soc=0.30)))

    assert held["eventsat_0"]["mode"] == "charging"
    assert planner.diagnostics()["grounding_overrides"] == 1.0


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
    # Pin the retry budget: this asserts the failure mode, not the default, and
    # the replay list must outlast every attempt for the raise to be the schema
    # rejection rather than an exhausted replay.
    planner = create_representation(
        "eventsat", "llm-s", "ground", {"llm_replay": [invalid] * 3, "llm_parse_retries": 2}
    )
    with pytest.raises(RuntimeError, match="substrate integrity"):
        planner.select_action(_context())


def test_markdown_fence_is_tolerated_without_relaxing_schema() -> None:
    fenced = f"```json\n{_response()}\n```"
    planner = create_representation("eventsat", "llm-s", "ground", {"llm_replay": [fenced]})
    assert planner.select_action(_context())["eventsat_0"]["mode"] == "communication"


@pytest.mark.parametrize("base_seed", [None, 73])
def test_parse_retry_reaches_provider_and_then_replays_cache(
    tmp_path, monkeypatch, base_seed
) -> None:
    config = {"llm_provider": "ollama", "llm_cache_dir": str(tmp_path), "llm_parse_retries": 2}
    if base_seed is not None:
        config["llm_seed"] = base_seed
    planner = create_representation("eventsat", "llm-s", "ground", config)
    planner.reset(42)
    calls = []

    def provider(*args):
        calls.append(args[-1])
        return "not json" if len(calls) == 1 else _response()

    monkeypatch.setattr(planner.client, "_call_provider", provider)
    action = planner.select_action(_context())
    assert action["eventsat_0"]["mode"] == "communication"
    first_seed = 42 if base_seed is None else base_seed
    assert calls == [first_seed, first_seed + 1]
    planner.reset(42)
    assert planner.select_action(_context()) == action
    assert len(calls) == 2
    assert planner.client.metrics()["llm_cache_hits"] == 2.0


def test_hybrid_onboard_shield_permits_prepointing_only_with_obc_data() -> None:
    def decide(**updates):
        planner = create_representation(
            "eventsat",
            "hllm-s",
            "onboard",
            {"llm_replay": [_response("communication", [["charging", 2]])], "plan_hold": 3},
        )
        state = _state(ground_pass_active=False, **updates)
        return planner.select_action(DecisionContext(state, {}, None, 0, role="onboard"))

    assert decide(obc_data_mb=2.0)["eventsat_0"]["mode"] == "communication"
    assert decide(obc_data_mb=0.0)["eventsat_0"]["mode"] == "charging"
