"""RLlib multi-agent bridge over the canonical mission, organisation, and paradigm.

Ported from the agentic framework: one RLlib agent per organisation agent, whose
adapter encodes its scoped view and decodes its commands through the same
controller-visible shield used at evaluation. The organisation's
``distribute_observation`` and ``collect_actions`` are the ones the runner uses,
so training and evaluation share one MDP. Autonomous-onboard paradigms observe
fresh telemetry and act immediately, so no paradigm hook intervenes.

Training episodes draw launch seeds from ``TRAINING_SEED_FLOOR`` upwards, which
keeps the small paired evaluation seeds unseen during training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from autops.config import ExperimentSpec
from autops.core.runner import eventsat_environment
from autops.organisations import AgentAction, bind_communication_topology
from autops.organisations.base import validate_agent_satellite_mapping
from autops.organisations.topologies import ORGANISATIONS
from autops.rl.diagnostics import accumulate, empty_diagnostics
from autops.rl.shaping import PipelineShaping, pipeline_state
from autops.rl.spaces import RLSpaceAdapter, make_space_adapter

TRAINING_SEED_FLOOR = 1_000_000
SUPPORTED_PARADIGMS = frozenset({"ao"})


def organisation_config(spec: ExperimentSpec) -> dict[str, Any]:
    """Organisation options from the mission defaults and the coordinate overrides."""

    defaults = spec.mission_config.get("organisation_defaults", {}).get(spec.organisation, {})
    return {**defaults, **spec.organisation_config}


class AUTOPSRLLibMultiAgentEnv(MultiAgentEnv):
    """Expose one AUTOPS coordinate as an RLlib multi-agent environment."""

    def __init__(self, env_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        config = dict(env_config or {})
        self.spec = ExperimentSpec(**config["spec"])
        if self.spec.paradigm not in SUPPORTED_PARADIGMS:
            raise ValueError(
                f"RL training supports the {sorted(SUPPORTED_PARADIGMS)} paradigms; "
                f"{self.spec.paradigm!r} needs its paradigm hooks in the bridge first"
            )
        recipe = dict(config.get("recipe", {}))
        self._environment = _mission_environment(self.spec, bool(config.get("prefer_orekit", True)))
        self._organisation = ORGANISATIONS[self.spec.organisation](organisation_config(self.spec))
        self._organisation.initialize(_satellite_ids(self._environment))
        validate_agent_satellite_mapping(self._organisation)
        self.possible_agents = self._organisation.get_agents()
        self.agents: list[str] = []
        self._adapters: dict[str, RLSpaceAdapter] = {
            agent_id: make_space_adapter(
                self.spec.mission,
                self._organisation.satellites_for_agent(agent_id),
                self._organisation.observed_satellites_for_agent(agent_id),
            )
            for agent_id in self.possible_agents
        }
        spaces = {
            agent_id: adapter.spaces(self.spec.mission_config)
            for agent_id, adapter in self._adapters.items()
        }
        self.observation_spaces = {agent_id: pair[0] for agent_id, pair in spaces.items()}
        self.action_spaces = {agent_id: pair[1] for agent_id, pair in spaces.items()}
        self.observation_space = self.observation_spaces[self.possible_agents[0]]
        self.action_space = self.action_spaces[self.possible_agents[0]]
        shaping = dict(recipe.get("pipeline_shaping", {}))
        self._shaping = (
            PipelineShaping(
                potential=str(shaping.get("potential", "delivery")),
                scale=float(shaping.get("scale", 1.0)),
                discount=float(recipe.get("gamma", 1.0)),
            )
            if shaping.get("enabled", False)
            else None
        )
        self._seed_rng = np.random.default_rng(TRAINING_SEED_FLOOR)
        self._observation: dict[str, Any] = {}
        self._diagnostics: dict[str, float] = {}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        del options
        if seed is not None:
            self._seed_rng = np.random.default_rng(TRAINING_SEED_FLOOR + int(seed))
        episode_seed = int(self._seed_rng.integers(TRAINING_SEED_FLOOR, 2**31 - 1))
        self._observation = self._environment.reset(episode_seed)
        self._organisation.initialize(_satellite_ids(self._environment))
        bind_communication_topology(self._organisation, self._environment)
        self.agents = list(self.possible_agents)
        self._diagnostics = empty_diagnostics() if self.spec.mission == "eventsat" else {}
        return self._encode(self._observation), {agent_id: {} for agent_id in self.agents}

    def step(self, action_dict: dict[str, Any]) -> tuple[dict, dict, dict, dict, dict]:
        organisation, observation = self._organisation, self._observation
        channel = organisation.channel(observation)
        views = organisation.distribute_observation(observation, channel)
        plans = {}
        for agent_id in self.agents:
            if agent_id in action_dict:
                adapter = self._adapters[agent_id]
                decoded = adapter.decode_action(action_dict[agent_id])
                action = adapter.ground_decoded_action(decoded, views[agent_id])
                plans[agent_id] = AgentAction(agent_id, action)
        commands = organisation.collect_actions(plans, channel)
        before = pipeline_state(self._environment) if self._shaping else None
        result = self._environment.step(commands)
        satellite_rewards = self._satellite_rewards(result, before)
        if self._diagnostics:
            satellite_id = self._environment.satellite_id
            metadata = result.observation["satellites"][satellite_id]["metadata"]
            accumulate(self._diagnostics, result.info, metadata)
        self._observation = result.observation
        active, done = list(self.agents), bool(result.done)
        rewards = {
            agent_id: float(
                sum(
                    satellite_rewards.get(satellite_id, 0.0)
                    for satellite_id in organisation.satellites_for_agent(agent_id)
                )
            )
            for agent_id in active
        }
        if done:
            self.agents = []
        observations = {} if done else self._encode(self._observation)
        terminateds = {**dict.fromkeys(active, done), "__all__": done}
        truncateds = {**dict.fromkeys(active, False), "__all__": False}
        return observations, rewards, terminateds, truncateds, {agent: {} for agent in observations}

    def episode_diagnostics(self) -> dict[str, float]:
        return dict(self._diagnostics)

    def _encode(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        views = self._organisation.distribute_observation(
            observation, self._organisation.channel(observation)
        )
        return {
            agent_id: self._adapters[agent_id].encode_observation(views[agent_id])
            for agent_id in self.agents
        }

    def _satellite_rewards(self, result: Any, before: dict[str, Any] | None) -> dict[str, float]:
        """Per-satellite mission reward, plus optional shaping for training only."""

        environment = self._environment
        reward = float(result.reward)
        if before is not None and self._shaping is not None:
            reward += environment.reward_function.reward_scale * self._shaping.reward(
                before,
                pipeline_state(environment),
                compression_ratio=float(environment.config["storage"]["compression_ratio"]),
                downlink_target_mb=environment.mission_targets[1],
                is_final_step=bool(result.done),
            )
        return {environment.satellite_id: reward}


def _mission_environment(spec: ExperimentSpec, prefer_orekit: bool) -> Any:
    if spec.mission == "eventsat":
        return eventsat_environment(spec, prefer_orekit=prefer_orekit)
    raise ValueError(f"no RL environment for mission {spec.mission!r}")


def _satellite_ids(environment: Any) -> list[str]:
    return list(getattr(environment, "satellite_ids", [environment.satellite_id]))


__all__ = [
    "SUPPORTED_PARADIGMS",
    "TRAINING_SEED_FLOOR",
    "AUTOPSRLLibMultiAgentEnv",
    "organisation_config",
]
