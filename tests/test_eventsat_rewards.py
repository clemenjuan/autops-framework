"""EventSat reward: agentic Individual Negative terms and their environment inputs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from autops.config import expand_coordinate
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.rewards import EventSatRewardFunction


@pytest.fixture
def rf() -> EventSatRewardFunction:
    return EventSatRewardFunction({"reward_scale": 1.0})


def _environment(max_steps: int = 20, **rewards: Any) -> EventSatEnvironment:
    config = deepcopy(expand_coordinate("eventsat/sas/ao/symb").mission_config)
    config["anomalies"]["probability_per_step"] = 0.0
    config["rewards"].update(rewards)
    return EventSatEnvironment(config, max_steps=max_steps, prefer_orekit=False)


def _isolated_actions(**rewards: Any) -> EventSatEnvironment:
    """Only the action term is non-zero, at unit scale."""

    return _environment(reward_scale=1.0, resource_penalty_factor=0.0, mission_scale=0.0, **rewards)


class TestResourcePenalty:
    def test_no_penalty_for_healthy_resources(self, rf):
        assert rf.resource_penalty(0.8, 100.0, 512.0) == 0.0

    def test_low_battery_and_high_storage_are_penalised(self, rf):
        battery = rf.resource_penalty(0.1, 0.0, 512.0)
        storage = rf.resource_penalty(0.8, 500.0, 512.0)
        both = rf.resource_penalty(0.1, 500.0, 512.0)
        assert battery < 0.0 and storage < 0.0
        assert both < battery and both < storage


class TestActionReward:
    @pytest.mark.parametrize(
        "mode,info,failed",
        [
            ("communication", {"pass_active": False}, True),
            ("communication", {"pass_active": True, "data_downlinked_mb": 0.0}, True),
            ("communication", {"pass_active": True, "data_downlinked_mb": 1.0}, False),
            ("payload_observe", {"storage_overflow": True}, True),
            ("payload_compress", {}, True),
            ("payload_compress", {"had_data_to_compress": True}, False),
            ("payload_detect", {}, True),
            ("payload_send", {}, True),
            ("charging", {"constraint_violation": True}, True),
            ("charging", {}, False),
            ("safe", {}, False),
        ],
    )
    def test_failure_classification_excludes_equal_safe_penalty(self, mode, info, failed):
        reward = EventSatRewardFunction({"failed_action_penalty": 0.3, "safe_penalty": 0.3})
        assert reward.is_failed_action(mode, info) is failed
        if failed:
            assert reward.action_reward(mode, info) == -0.3

    def test_pipeline_work_is_neutral_by_default(self, rf):
        assert rf.action_reward("payload_observe", {"storage_overflow": False}) == 0.0
        assert rf.action_reward("payload_compress", {"had_data_to_compress": True}) == 0.0
        shaped = EventSatRewardFunction({"observe_reward": 1.0})
        assert shaped.action_reward("payload_observe", {"storage_overflow": False}) > 0.0

    def test_delivery_is_rewarded_up_to_the_cap(self, rf):
        assert rf.action_reward("communication", {"pass_active": True, "data_downlinked_mb": 2.0})
        capped = rf.action_reward(
            "communication", {"pass_active": True, "data_downlinked_mb": 100.0}
        )
        assert capped == rf.comm_reward_cap

    @pytest.mark.parametrize(
        "info,failed",
        [
            ({"pass_active": False, "communication_failure": "no_contact"}, True),
            ({"pass_active": True, "communication_failure": "no_contact"}, True),
            ({"pass_active": True, "communication_failure": "no_source_data"}, False),
            ({"pass_active": True}, False),
        ],
    )
    def test_empty_downlink_failure_is_explicit_ablation(self, info, failed):
        reward = EventSatRewardFunction({"empty_downlink_is_failure": False})
        assert reward.is_failed_action("communication", info) is failed

    def test_safe_is_worse_than_neutral_charging(self, rf):
        assert rf.action_reward("charging", {}) == 0.0
        assert rf.action_reward("safe", {}) < 0.0


_TARGETS = {"obs_target_hours": 2.0, "downlink_target_mb": 240.0, "max_mission_steps": 10080}


class TestMissionPenalty:
    def test_penalty_tracks_only_delivered_data_by_default(self, rf):
        met = rf.mission_penalty(True, 0.0, 240.0, episode_steps=10080, **_TARGETS)
        hoarded = rf.mission_penalty(True, 2.0, 0.0, episode_steps=10080, **_TARGETS)
        assert met == 0.0
        assert hoarded == pytest.approx(-rf.mission_scale)

    def test_observation_credit_is_explicit_ablation(self):
        rf = EventSatRewardFunction({"mission_observation_weight": 1.0})
        penalty = rf.mission_penalty(True, 2.0, 0.0, episode_steps=10080, **_TARGETS)
        assert penalty == pytest.approx(-0.5 * rf.mission_scale)

    def test_penalty_grows_with_episode_progress(self, rf):
        early = rf.mission_penalty(False, 0.0, 0.0, episode_steps=100, **_TARGETS)
        late = rf.mission_penalty(False, 0.0, 0.0, episode_steps=9000, **_TARGETS)
        assert early > late


def test_total_reward_is_scaled() -> None:
    kwargs = {
        "mode": "payload_observe",
        "battery_soc": 0.8,
        "data_stored_mb": 100.0,
        "storage_capacity_mb": 512.0,
        "action_info": {"storage_overflow": False},
        "obs_hours": 1.0,
        "downlinked_mb": 100.0,
        "obs_target_hours": 2.0,
        "downlink_target_mb": 240.0,
        "episode_step": 5000,
        "max_steps": 10080,
        "is_final_step": False,
    }
    scaled = EventSatRewardFunction({"reward_scale": 0.01}).compute(**kwargs)
    assert scaled == pytest.approx(
        0.01 * EventSatRewardFunction({"reward_scale": 1.0}).compute(**kwargs)
    )


def test_environment_penalises_a_command_clamped_to_charging() -> None:
    env = _isolated_actions()
    env.reset(7)
    env.state.battery_soc = 0.35
    step = env.step({"eventsat_0": {"mode": "payload_observe"}})
    assert step.info["resolved_mode"] == "charging"
    assert step.info["constraint_violation"] and step.info["failed_action"]
    assert step.reward == pytest.approx(-env.reward_function.failed_action_penalty)
    assert step.info["failed_action_penalty"] == pytest.approx(step.reward)


def test_environment_scores_forced_safe_mode_with_the_safe_penalty_only() -> None:
    env = _isolated_actions()
    env.reset(7)
    env.state.battery_soc = 0.1
    step = env.step({"eventsat_0": {"mode": "payload_observe"}})
    assert step.info["resolved_mode"] == "safe"
    assert not step.info["constraint_violation"] and not step.info["failed_action"]
    assert step.reward == pytest.approx(-env.reward_function.safe_penalty)


def test_communication_outside_contact_is_a_failed_radio_attempt() -> None:
    env = _isolated_actions()
    env.reset(7)
    while env.physical_contact_active():
        env.step({"eventsat_0": {"mode": "charging"}})
    env.state.previous_mode = "communication"
    step = env.step({"eventsat_0": {"mode": "communication"}})
    assert step.info["resolved_mode"] == "communication"
    assert step.info["failure_reason"] == "no_contact"
    assert step.info["failed_action"]
    assert step.info["gross_energy_consumed_wh"] > 0.1


def test_in_progress_compression_is_not_a_failed_action() -> None:
    env = _isolated_actions()
    env.reset(7)
    env.state.uncompressed_observations = 1
    env.state.jetson_raw_mb = env.config["storage"]["observation_size_mb"]
    step = env.step({"eventsat_0": {"mode": "payload_compress"}})
    assert step.info["action_accepted"] and not step.info["failed_action"]
    assert step.info["failure_reason"] is None
    assert step.reward == 0.0


def test_storage_penalty_uses_obc_occupancy(monkeypatch) -> None:
    env = _environment()
    env.reset(7)
    env.state.jetson_raw_mb = 12_000.0
    env.state.obc_data_mb = 123.0
    captured: dict[str, Any] = {}

    def capture(**kwargs: Any) -> float:
        captured.update(kwargs)
        return 0.0

    monkeypatch.setattr(env.reward_function, "compute", capture)
    env.step({"eventsat_0": {"mode": "charging"}})
    assert captured["data_stored_mb"] == pytest.approx(123.0)
    assert captured["storage_capacity_mb"] == env.config["storage"]["obc_capacity_mb"]


def test_ignored_command_while_settling_is_neither_forced_nor_violating() -> None:
    env = _isolated_actions()
    env.reset(7)
    env.step({"eventsat_0": {"mode": "payload_observe"}})
    env.state.battery_soc = 0.35
    step = env.step({"eventsat_0": {"mode": "payload_observe"}})
    assert step.info["in_transition"] and step.info["command_ignored"]
    assert not step.info["forced"] and not step.info["constraint_violation"]
