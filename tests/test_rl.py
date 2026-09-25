"""The rl representation: adapters, shaping, checkpoints, bridge parity, and training."""

from __future__ import annotations

import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from autops.board.evidence import validate_result_document
from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner, eventsat_environment
from autops.core.types import DecisionContext
from autops.missions.eventsat.observation import encode_vectors
from autops.representations.rl import EventSatRL
from autops.rl.policy import MANIFEST_SCHEMA_VERSION, validate_manifest
from autops.rl.shaping import PipelineShaping
from autops.rl.spaces import EVENTSAT_RL_SPEC, EventSatSpaceAdapter, ground_eventsat_mode


def _environment(steps: int = 40):
    spec = expand_coordinate(
        "eventsat/sas/ao/rl",
        steps=steps,
        overrides={"mission": {"anomalies": {"probability_per_step": 0.0}}},
    )
    return spec, eventsat_environment(spec, prefer_orekit=False)


def test_eventsat_adapter_encodes_the_onboard_vector_within_bounds() -> None:
    spec, env = _environment()
    adapter = EventSatSpaceAdapter(["eventsat_0"], ["eventsat_0"])
    observation = env.reset(7)
    low, high = adapter.observation_bounds(spec.mission_config)
    for mode in ("payload_observe", "communication", "payload_compress", "charging"):
        vector = adapter.encode_observation(observation)
        np.testing.assert_array_equal(vector, encode_vectors(observation)[0])
        assert np.all(vector >= low) and np.all(vector <= high)
        observation = env.step({"eventsat_0": {"mode": mode}}).observation
    assert adapter.decode_action([99]) == {"eventsat_0": {"mode": "safe"}}


@pytest.mark.parametrize(
    "health,soc,requested,expected",
    [
        ("thermal_warning", 0.9, "payload_observe", "safe"),
        ("nominal", 0.1, "communication", "charging"),
        ("nominal", 0.9, "communication", "communication"),
    ],
)
def test_shield_protects_health_and_battery_but_never_vetoes_contact(
    health: str, soc: float, requested: str, expected: str
) -> None:
    grounded = ground_eventsat_mode(
        requested, battery_soc=soc, health_status=health, battery_min_soc=0.2
    )
    assert grounded == expected


def test_delivery_potential_credits_stages_at_one_third_steps() -> None:
    shaping = PipelineShaping(discount=0.9, scale=2.0)
    ratio, target = 5.0, 10.0
    kwargs = {"compression_ratio": ratio, "downlink_target_mb": target}
    compressed = {"jetson_compressed_mb": target}
    obc = {"obc_raw_equivalent_mb": target * ratio}
    ground = {"downlink_raw_equivalent_mb": target * ratio}
    assert shaping.value(compressed, **kwargs) == pytest.approx(1 / 3)
    assert shaping.value(obc, **kwargs) == pytest.approx(2 / 3)
    assert shaping.value(ground, **kwargs) == pytest.approx(1.0)
    assert shaping.value({**ground, **obc}, **kwargs) == pytest.approx(1.0)
    step = shaping.reward(compressed, obc, is_final_step=False, **kwargs)
    assert step == pytest.approx(2.0 * (0.9 * 2 / 3 - 1 / 3))
    assert shaping.reward(compressed, obc, is_final_step=True, **kwargs) == pytest.approx(-2 / 3)
    raw = PipelineShaping(potential="raw_progress")
    product = {"jetson_raw_mb": 9.41, "observation_size_mb": 9.41, "uncompressed_observations": 1}
    assert raw.value(product, **kwargs) == pytest.approx(0.25 * 9.41 / (target * ratio))
    with pytest.raises(ValueError, match="potential"):
        PipelineShaping(potential="unknown")


def _manifest(**updates: Any) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "observation_schema_id": EVENTSAT_RL_SPEC.schema_id,
        "observation_names": list(EVENTSAT_RL_SPEC.observation_names),
        "policy_observation_shapes": {"shared_policy": [EVENTSAT_RL_SPEC.obs_dim]},
        "policy_action_nvec": {"shared_policy": [7]},
        **updates,
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"observation_schema_id": "eventsat_log_pipeline_attitude_pass_v4"},
        {"observation_names": ["battery_soc"]},
        {"policy_observation_shapes": {"shared_policy": [33]}},
        {"policy_action_nvec": {"shared_policy": [7, 2, 2]}},
    ],
)
def test_checkpoints_from_another_contract_are_rejected(updates: dict[str, Any]) -> None:
    validate_manifest(_manifest(), EVENTSAT_RL_SPEC, "shared_policy", 45, [7])
    with pytest.raises(ValueError, match="RL checkpoint"):
        validate_manifest(_manifest(**updates), EVENTSAT_RL_SPEC, "shared_policy", 45, [7])


def test_checkpoint_identity_binds_the_policy_weights(tmp_path: Path) -> None:
    from autops.rl.policy import checkpoint_identity, policy_sha256

    weights = tmp_path / "policies" / "shared_policy"
    weights.mkdir(parents=True)
    (weights / "policy_state.pkl").write_bytes(b"trained")
    manifest = _manifest(policy_sha256={"shared_policy": policy_sha256(tmp_path, "shared_policy")})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = checkpoint_identity(tmp_path, manifest, "shared_policy")
    assert identity["policy_sha256"] == manifest["policy_sha256"]["shared_policy"]
    (weights / "policy_state.pkl").write_bytes(b"replaced")
    with pytest.raises(ValueError, match="weights differ"):
        checkpoint_identity(tmp_path, manifest, "shared_policy")


def test_rl_representation_requires_a_checkpoint_or_explicit_mock() -> None:
    with pytest.raises(ValueError, match=r"representation\.checkpoint"):
        EventSatRL({})
    with pytest.raises(FileNotFoundError, match="manifest"):
        EventSatRL({"checkpoint": "missing-checkpoint"})


def test_mock_results_record_their_source_and_are_never_boardable(tmp_path: Path) -> None:
    spec = expand_coordinate(
        "eventsat/sas/ao/rl", steps=6, overrides={"representation": {"rl_mock": True}}
    )
    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    assert result["experiment"]["rl_policy_identity"] == {"source": "mock"}
    step = result["episodes"][0]["decision_diagnostics"]["onboard"]
    assert step["decisions"] == 6
    assert "jetson" not in json.dumps(result["episodes"][0]["decision_diagnostics"])
    assert result["episodes"][0]["planner_compute_energy_wh"] == 0.0
    with pytest.raises(ValueError, match="trained checkpoint identity"):
        validate_result_document(result, tmp_path / "result.json")


class _Scripted:
    """A policy replaying fixed actions, to drive the runner like RLlib drives the bridge."""

    def __init__(self, actions: list[int]) -> None:
        self.actions = list(actions)

    def seed(self, seed: int) -> None:
        del seed

    def act(self, obs: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
        del obs, deterministic
        return np.asarray([self.actions.pop(0)]), np.full(7, 1 / 7)


@pytest.mark.rl
def test_training_bridge_and_runner_share_one_mdp() -> None:
    pytest.importorskip("ray")
    from autops.memory.fixed import FixedMemory
    from autops.paradigms.ao import AutonomousOnboard
    from autops.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

    spec, env = _environment(steps=30)
    bridge = AUTOPSRLLibMultiAgentEnv(
        {"spec": spec.model_dump(mode="json"), "recipe": {}, "prefer_orekit": False}
    )
    encoded, _ = bridge.reset(seed=3)
    episode_seed = bridge._environment._seed
    actions = [2, 2, 2, 3, 3, 3, 3, 4, 1, 1, 0, 6, 5, 2, 2, 2]
    representation = EventSatRL({"rl_mock": True})
    representation._policy = _Scripted(actions)
    paradigm = AutonomousOnboard(representation, FixedMemory())
    observation = env.reset(episode_seed)
    paradigm.reset(episode_seed, observation)
    assert bridge.observation_spaces["central_agent"].contains(encoded["central_agent"])
    for action in actions:
        state = representation.encode_observation(observation)
        np.testing.assert_array_equal(encoded["central_agent"], state["vector"])
        encoded, rewards, _, _, _ = bridge.step({"central_agent": np.asarray([action])})
        decision = paradigm.act(observation, physical_contact=env.physical_contact_active())
        step = env.step(decision.actions)
        paradigm.after_step(step.info, step.observation)
        observation = step.observation
        assert rewards["central_agent"] == pytest.approx(step.reward)
    diagnostics = bridge.episode_diagnostics()
    assert diagnostics["observations"] >= 1 and diagnostics["settling_steps"] >= 1


@pytest.mark.rl
def test_shaping_is_added_only_by_the_training_bridge() -> None:
    pytest.importorskip("ray")
    from autops.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

    spec, _ = _environment(steps=10)
    rewards = {}
    for enabled in (False, True):
        bridge = AUTOPSRLLibMultiAgentEnv(
            {
                "spec": spec.model_dump(mode="json"),
                "recipe": {"gamma": 1.0, "pipeline_shaping": {"enabled": enabled}},
                "prefer_orekit": False,
            }
        )
        bridge.reset(seed=1)
        bridge._environment.state.jetson_compressed_mb = 1.0
        rewards[enabled] = bridge.step({"central_agent": np.asarray([5])})[1]["central_agent"]
    assert rewards[True] > rewards[False]


@pytest.mark.rl
def test_policy_sharing_rejects_agents_with_different_spaces() -> None:
    gymnasium = pytest.importorskip("gymnasium")
    pytest.importorskip("ray")
    from autops.rl.policy_mapping import PolicySharingConfig, build_policy_specs

    small = gymnasium.spaces.Box(0.0, 1.0, (3,))
    large = gymnasium.spaces.Box(0.0, 1.0, (4,))
    action = gymnasium.spaces.MultiDiscrete([6])
    shared = PolicySharingConfig("shared_all")
    with pytest.raises(ValueError, match="cannot share"):
        build_policy_specs(["a", "b"], {"a": small, "b": large}, {"a": action, "b": action}, shared)
    independent = PolicySharingConfig("independent_per_agent")
    specs = build_policy_specs(
        ["a", "b"], {"a": small, "b": large}, {"a": action, "b": action}, independent
    )
    assert set(specs) == {"policy_a", "policy_b"}
    assert PolicySharingConfig("shared_by_role").policy_id_for("sat_agent_3") == "satellite_policy"


@pytest.mark.rl
@pytest.mark.slow
def test_trained_checkpoint_is_evaluated_through_the_runner(tmp_path: Path) -> None:
    pytest.importorskip("ray")
    from autops.rl.training import RLlibPPOTrainer, load_recipe

    spec = expand_coordinate("eventsat/sas/ao/rl", steps=32)
    recipe = load_recipe(
        "eventsat",
        overrides={
            "timesteps": 128,
            "num_env_runners": 0,
            "num_gpus": 0,
            "rollout_fragment": 32,
            "train_batch_size": 64,
            "minibatch_size": 32,
            "ppo_epochs": 1,
            "checkpoint_every_timesteps": 64,
        },
    )
    checkpoint = RLlibPPOTrainer(spec, recipe, tmp_path / "ppo", prefer_orekit=False).train()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["sampled_steps"] == 128
    # Manifests enter model artifacts: no machine identity may leak from RLlib results.
    serialized = json.dumps(manifest)
    assert "hostname" not in serialized and "node_ip" not in serialized
    assert manifest["last_result"]["training_iteration"] >= 1
    assert (checkpoint / "step_000000064" / "manifest.json").is_file()
    assert (checkpoint / "tensorboard").is_dir()
    evaluation = expand_coordinate(
        "eventsat/sas/ao/rl",
        steps=8,
        overrides={"representation": {"checkpoint": str(checkpoint)}},
    )
    result = ExperimentRunner(evaluation, save=False, prefer_orekit=False).run()
    identity = result["experiment"]["rl_policy_identity"]
    assert identity["source"] == "checkpoint" and identity["sampled_steps"] == 128
    assert identity["policy_sha256"] == manifest["policy_sha256"]["shared_policy"]
    representation = EventSatRL({"checkpoint": str(checkpoint), "deterministic": True})
    observation = deepcopy(eventsat_environment(evaluation, prefer_orekit=False).reset(5))
    context = DecisionContext(representation.encode_observation(observation), observation, None, 0)
    first = representation.select_action(context)
    assert first == representation.select_action(context)
    # Weights replaced at one path are restored again, not served from the policy cache.
    served = tmp_path / "served"
    shutil.copytree(checkpoint / "step_000000064", served)
    early = EventSatRL({"checkpoint": str(served)})
    shutil.rmtree(served)
    shutil.copytree(checkpoint, served)
    late = EventSatRL({"checkpoint": str(served)})
    assert late.identity["policy_sha256"] == identity["policy_sha256"]
    assert early.identity["policy_sha256"] != late.identity["policy_sha256"]
    vector = context.state["vector"]
    early_probabilities = early._policy.act(vector, deterministic=True)[1]
    assert not np.array_equal(early_probabilities, late._policy.act(vector, deterministic=True)[1])


@pytest.mark.rl
def test_hybrid_training_bridge_arbitrates_like_the_runner() -> None:
    pytest.importorskip("ray")
    from autops.memory.fixed import FixedMemory
    from autops.paradigms.ah import AutonomousHybrid
    from autops.representations.symb import EventSatSymbolicScheduler
    from autops.rl.rllib_env import AUTOPSRLLibMultiAgentEnv

    spec = expand_coordinate(
        "eventsat/sas/ah/rl/symb",
        steps=720,
        overrides={"mission": {"anomalies": {"probability_per_step": 0.0}}},
    )
    env = eventsat_environment(spec, prefer_orekit=False)
    bridge = AUTOPSRLLibMultiAgentEnv(
        {"spec": spec.model_dump(mode="json"), "recipe": {}, "prefer_orekit": False}
    )
    encoded, _ = bridge.reset(seed=11)
    episode_seed = bridge._environment._seed
    rng = np.random.default_rng(5)
    representation = EventSatRL({"rl_mock": True})
    scripted = _Scripted([])
    representation._policy = scripted
    paradigm = AutonomousHybrid(
        representation, EventSatSymbolicScheduler({"conventional": False}), FixedMemory()
    )
    observation = env.reset(episode_seed)
    paradigm.reset(episode_seed, observation)
    overridden = 0
    for _ in range(spec.steps):
        # Communicate at contact so the plan is uplinked; elsewhere request productive work.
        action = 1 if env.physical_contact_active() else int(rng.choice([2, 3, 4, 5]))
        scripted.actions.append(action)
        np.testing.assert_array_equal(
            encoded["central_agent"], representation.encode_observation(observation)["vector"]
        )
        encoded, rewards, terminated, _, _ = bridge.step({"central_agent": np.asarray([action])})
        adapter = representation.adapter
        onboard = adapter.ground_decoded_action(adapter.decode_action([action]), observation)
        decision = paradigm.act(observation, physical_contact=env.physical_contact_active())
        overridden += decision.actions["eventsat_0"]["mode"] != onboard["eventsat_0"]["mode"]
        step = env.step(decision.actions)
        paradigm.after_step(step.info, step.observation)
        observation = step.observation
        assert rewards["central_agent"] == pytest.approx(step.reward)
        if terminated["__all__"]:
            break
    assert paradigm.ground.last_rationale is not None
    assert overridden > 0
