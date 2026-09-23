"""SSA multi-agent RL: local vector, shield, per-satellite reward, bridge, and training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner
from autops.core.ssa_runner import ssa_environment
from autops.missions.ssa.observation import SSA_RL_OBSERVATIONS, ssa_rl_vector
from autops.missions.ssa.rewards import SSARewardFunction
from autops.rl.spaces import SSASpaceAdapter, ground_ssa_mode


def _spec(token: str = "dmas", size: int = 3, steps: int = 12, **overrides):
    return expand_coordinate(
        f"ssa/{token}/ao/rl", steps=steps, constellation_size=size, overrides=overrides or None
    )


def test_countdowns_are_published_only_for_rl_coordinates() -> None:
    rl = ssa_environment(_spec()).reset(3)["satellites"]["sat_0"]
    symbolic_spec = expand_coordinate("ssa/dmas/ao/symb", steps=12, constellation_size=3)
    symbolic = ssa_environment(symbolic_spec).reset(3)["satellites"]["sat_0"]
    countdowns = {"time_to_next_pass", "time_to_next_eclipse", "remaining_pass_duration"}
    assert countdowns <= set(rl)
    assert not countdowns & set(symbolic)
    for record in (rl, symbolic):
        assert {"detection_progress", "catalog_size", "orbital_period_steps"} <= set(record)


def test_local_vector_is_bounded_and_zero_for_unseen_satellites() -> None:
    observation = ssa_environment(_spec()).reset(5)
    vector = ssa_rl_vector(observation, "sat_1")
    assert vector.shape == (len(SSA_RL_OBSERVATIONS),) == (23,)
    assert np.all(vector >= 0.0) and np.all(vector <= 1.0)
    np.testing.assert_array_equal(ssa_rl_vector(observation, "sat_9"), np.zeros(23))
    adapter = SSASpaceAdapter(["sat_0"], ["sat_0", "sat_9"])
    assert adapter.encode_observation(observation).shape == (46,)


def _satellite(**updates):
    return {
        "battery_soc": 0.9,
        "health": "nominal",
        "ground_pass_active": False,
        "storage_used_fraction": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_capacity_mb": 1000.0,
        "observation_size_mb": 10.0,
        "unprocessed_batches": 0,
        "undelivered_records": 0,
        "known_objects": [],
        "has_isl_peer": False,
        **updates,
    }


@pytest.mark.parametrize(
    "mode,satellite,coordinated,expected",
    [
        ("payload_observe", _satellite(health="fault"), False, "safe"),
        ("payload_observe", _satellite(battery_soc=0.2), False, "charging"),
        ("communication", _satellite(undelivered_records=1), False, "charging"),
        (
            "communication",
            _satellite(ground_pass_active=True, undelivered_records=1),
            False,
            "communication",
        ),
        ("payload_observe", _satellite(battery_soc=0.5), False, "charging"),
        ("payload_observe", _satellite(), False, "payload_observe"),
        ("payload_detect", _satellite(), False, "charging"),
        ("isl_share", _satellite(undelivered_records=2), False, "charging"),
        ("isl_share", _satellite(undelivered_records=2, has_isl_peer=True), False, "isl_share"),
        ("isl_share", _satellite(known_objects=["rso_0"]), True, "isl_share"),
    ],
)
def test_ssa_shield_maps_the_agentic_rules_to_lean_telemetry(
    mode: str, satellite: dict, coordinated: bool, expected: str
) -> None:
    assert ground_ssa_mode(mode, satellite, coordinated=coordinated) == expected


def test_per_satellite_reward_blends_local_and_team_terms_and_adds_custody_once() -> None:
    env = ssa_environment(_spec())
    env.reset(2)
    modes = {"sat_0": "safe", "sat_1": "charging", "sat_2": "charging"}
    per_satellite = {
        "sat_0": {},
        "sat_1": {"failure_reason": "no_contact"},
        "sat_2": {},
    }
    blended = SSARewardFunction({"local_weight": 0.7, "team_weight": 0.3})
    rewards = blended.satellite_rewards(env, modes, per_satellite)
    config = env.config["ssa"]
    local = {
        "sat_0": -config["safe_penalty"],
        "sat_1": -config["failed_action_penalty"],
        "sat_2": 0.0,
    }
    team = sum(local.values()) / 3
    custody = rewards["sat_2"] - 0.3 * team
    for satellite, value in local.items():
        assert rewards[satellite] == pytest.approx(0.7 * value + 0.3 * team + custody)
    assert custody <= 0.0
    with pytest.raises(ValueError, match="team_reducer"):
        SSARewardFunction({"team_reducer": "median"})


@pytest.mark.parametrize("token", ["sas", "cmas", "dmas", "hmas", "imas"])
def test_mock_policies_run_every_organisation_through_the_ssa_runner(token: str) -> None:
    spec = _spec(token, steps=6, representation={"rl_mock": True})
    result = ExperimentRunner(spec, save=False).run()
    assert result["experiment"]["rl_policy_identity"] == {"source": "mock"}
    assert result["episodes"][0]["steps"] == 6


def test_representation_overrides_reach_symbolic_ssa_policies(monkeypatch) -> None:
    from autops.organisations import loops

    seen: list[dict] = []
    original = loops.create_representation

    def recording(mission: str, token: str, role: str, config: dict | None = None):
        seen.append(dict(config or {}))
        return original(mission, token, role, config)

    monkeypatch.setattr(loops, "create_representation", recording)
    spec = expand_coordinate(
        "ssa/imas/ao/symb",
        steps=2,
        constellation_size=2,
        overrides={"representation": {"observe_soc": 0.5}},
    )
    ExperimentRunner(spec, save=False).run()
    assert seen and all(config["observe_soc"] == 0.5 for config in seen)


@pytest.mark.rl
def test_bridge_gives_each_agent_its_satellites_rewards() -> None:
    pytest.importorskip("ray")
    from autops.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

    spec = _spec("dmas")
    bridge = AUTOPSRLLibMultiAgentEnv(
        {
            "spec": spec.model_dump(mode="json"),
            "recipe": {"reward": {"local_weight": 1.0, "team_weight": 0.0}},
            "prefer_orekit": False,
        }
    )
    observations, _ = bridge.reset(seed=4)
    assert set(observations) == {"sat_agent_0", "sat_agent_1", "sat_agent_2"}
    for agent, vector in observations.items():
        assert bridge.observation_spaces[agent].contains(vector)
    actions = {agent: np.asarray([0]) for agent in observations}
    _, rewards, terminated, _, _ = bridge.step(actions)
    expected = SSARewardFunction({"local_weight": 1.0}).satellite_rewards(
        bridge._environment,
        {satellite: "charging" for satellite in bridge._environment.satellite_ids},
        {satellite: {} for satellite in bridge._environment.satellite_ids},
    )
    assert rewards["sat_agent_1"] == pytest.approx(expected["sat_1"])
    assert not terminated["__all__"]


@pytest.mark.rl
@pytest.mark.slow
def test_unequal_hmas_clusters_train_independent_policies(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    from autops.rl.training import RLlibPPOTrainer, load_recipe

    overrides = {"organisation": {"branching_factor": 2}}
    recipe = load_recipe(
        "ssa",
        overrides={
            "timesteps": 32,
            "num_env_runners": 0,
            "num_gpus": 0,
            "rollout_fragment": 16,
            "train_batch_size": 32,
            "minibatch_size": 16,
            "ppo_epochs": 1,
            "policy_sharing": "independent_per_agent",
        },
    )
    checkpoint = RLlibPPOTrainer(
        _spec("hmas", steps=16, **overrides), recipe, tmp_path / "hmas", prefer_orekit=False
    ).train()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy_observation_shapes"] == {
        "policy_cluster_agent_0": [46],
        "policy_cluster_agent_1": [23],
    }
    evaluation = _spec("hmas", steps=4, **overrides, representation={"checkpoint": str(checkpoint)})
    identity = ExperimentRunner(evaluation, save=False).run()["experiment"]["rl_policy_identity"]
    assert identity["source"] == "checkpoint"
    assert identity["policy_ids"] == ["policy_cluster_agent_0", "policy_cluster_agent_1"]
    with pytest.raises(ValueError, match="cannot share"):
        RLlibPPOTrainer(
            _spec("hmas", steps=16, **overrides),
            {**recipe, "policy_sharing": "shared_all"},
            tmp_path / "shared",
            prefer_orekit=False,
        ).train()
