"""RLlib PPO training of the ``rl`` representation (agentic framework trainer).

PPO (Schulman et al. 2017, arXiv:1707.06347) on RLlib's old API stack with the
AUTOPS actor-critic. Hyperparameters come from ``configs/rl/<mission>.yaml``;
``--recipe-set`` overrides keys strictly. A manifest next to every checkpoint,
including intermediate ``step_<sampled steps>`` snapshots, records the observation
contract, policy spaces, recipe, and public-safe provenance. RLlib writes
TensorBoard logs; W&B receives the same per-iteration metrics.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import register_env

from autops.config import ExperimentSpec, asset_root, load_yaml, strict_deep_merge
from autops.core.provenance import collect_provenance
from autops.rl.diagnostics import EpisodeDiagnostics
from autops.rl.models import MODEL_ARCHITECTURE, register_models
from autops.rl.policy import MANIFEST_NAME, MANIFEST_SCHEMA_VERSION, policy_sha256
from autops.rl.policy_mapping import PolicySharingConfig, build_policy_specs
from autops.rl.rllib_env import AUTOPSRLLibMultiAgentEnv
from autops.rl.spaces import rl_spec

logger = logging.getLogger(__name__)
_LOGGED_EVENTSAT_METRICS = ("downlinked_mb", "failed_action_penalty", "observations")


def load_recipe(
    mission: str, path: Path | None = None, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Canonical training recipe of a mission with strictly validated overrides."""

    recipe = load_yaml(path or asset_root() / "configs" / "rl" / f"{mission}.yaml")
    return strict_deep_merge(recipe, dict(overrides or {}), path="recipe")


class RLlibPPOTrainer:
    """Train one coordinate's ``rl`` policy and write manifest-bearing checkpoints."""

    def __init__(
        self,
        spec: ExperimentSpec,
        recipe: Mapping[str, Any],
        output: Path,
        *,
        prefer_orekit: bool = True,
        tracker: Any | None = None,
    ) -> None:
        self.spec = spec
        self.recipe = dict(recipe)
        self.output = output
        self.prefer_orekit = prefer_orekit
        self.tracker = tracker
        self._every = int(self.recipe["checkpoint_every_timesteps"])
        self._next_checkpoint = self._every
        self._last_result: dict[str, Any] = {}
        self._spaces: dict[str, dict[str, list[int]]] = {}

    def train(self) -> Path:
        """Run PPO until the recipe's sampled-step budget; return the final checkpoint."""

        started_ray = not ray.is_initialized()
        if started_ray:
            ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
        env_name = f"autops_{self.spec.name}_rllib"
        env_config = {
            "spec": self.spec.model_dump(mode="json"),
            "recipe": self.recipe,
            "prefer_orekit": self.prefer_orekit,
        }
        register_env(env_name, AUTOPSRLLibMultiAgentEnv)
        probe = AUTOPSRLLibMultiAgentEnv(env_config)
        sharing = PolicySharingConfig(str(self.recipe["policy_sharing"]))
        policies = build_policy_specs(
            probe.possible_agents, probe.observation_spaces, probe.action_spaces, sharing
        )
        self._spaces = {
            "policy_observation_shapes": {
                pid: list(spec.observation_space.shape) for pid, spec in policies.items()
            },
            "policy_action_nvec": {
                pid: [int(n) for n in spec.action_space.nvec] for pid, spec in policies.items()
            },
            "agent_policies": {
                agent: sharing.policy_id_for(agent) for agent in probe.possible_agents
            },
        }
        algorithm = self._config(env_name, env_config, policies, sharing).build_algo()
        try:
            sampled, iterations = 0, 0
            target = int(self.recipe["timesteps"])
            while sampled < target and iterations < int(self.recipe["max_iterations"]):
                self._last_result = algorithm.train()
                iterations += 1
                sampled = _sampled_steps(self._last_result)
                self._log_iteration(iterations, sampled, target)
                if self._every > 0 and self._next_checkpoint <= sampled < target:
                    self._next_checkpoint = (sampled // self._every + 1) * self._every
                    self._save(algorithm, self.output / f"step_{sampled:09d}", sampled)
            return self._save(algorithm, self.output, sampled)
        finally:
            algorithm.stop()
            if started_ray:
                ray.shutdown()

    def _config(
        self, env_name: str, env_config: dict[str, Any], policies: dict[str, Any], sharing: Any
    ) -> Any:
        register_models()
        recipe = self.recipe
        minibatch = min(int(recipe["minibatch_size"]), int(recipe["train_batch_size"]))
        config = (
            PPOConfig()
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .environment(env=env_name, env_config=env_config)
            .framework("torch")
            .env_runners(
                num_env_runners=int(recipe["num_env_runners"]),
                rollout_fragment_length=int(recipe["rollout_fragment"]),
                batch_mode="truncate_episodes",
            )
            .resources(num_gpus=float(recipe["num_gpus"]))
            .callbacks(EpisodeDiagnostics)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=sharing.mapping_fn(),
                policies_to_train=list(policies),
            )
            .debugging(
                seed=int(recipe["seed"]),
                # TensorBoard event files stay with the run, not in the user's ray_results.
                logger_config={"type": UnifiedLogger, "logdir": str(self.output / "tensorboard")},
            )
        )
        config.model = {
            **dict(config.model),
            "custom_model": MODEL_ARCHITECTURE,
            "custom_model_config": {"hidden_size": int(recipe["hidden_size"])},
        }
        training = {
            "lr": recipe["lr"],
            "gamma": recipe["gamma"],
            "lambda_": recipe["gae_lambda"],
            "clip_param": recipe["clip_ratio"],
            "entropy_coeff": recipe["entropy_coef"],
            "vf_loss_coeff": recipe["value_coef"],
            "grad_clip": recipe["max_grad_norm"],
            "train_batch_size": int(recipe["train_batch_size"]),
            "minibatch_size": minibatch,
            "num_epochs": int(recipe["ppo_epochs"]),
            "lr_schedule": recipe["lr_schedule"],
        }
        # PPOConfig.training forwards the generic keys to AlgorithmConfig.training.
        config = config.training(**training)
        unapplied = sorted(key for key, value in training.items() if getattr(config, key) != value)
        if unapplied:
            raise RuntimeError(f"RLlib ignored PPO settings {unapplied}")
        return config

    def _save(self, algorithm: Any, directory: Path, sampled: int) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        algorithm.save(str(directory))
        experiment = self.spec.model_dump(mode="json")
        spec = rl_spec(self.spec.mission)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "mission": self.spec.mission,
            "coordinate": self.spec.coordinate,
            "mechanism": "ppo",
            "implementation": "rllib",
            "model_architecture": MODEL_ARCHITECTURE,
            "sampled_steps": sampled,
            "policy_sharing": self.recipe["policy_sharing"],
            "observation_schema_id": spec.schema_id,
            "observation_names": list(spec.observation_names),
            **self._spaces,
            "recipe": self.recipe,
            "provenance": collect_provenance(experiment, asset_root()),
            # An allowlist: RLlib results also carry host names, node addresses and pids.
            "last_result": {
                "training_iteration": int(self._last_result.get("training_iteration", 0)),
                **_iteration_metrics(self._last_result, self.spec.mission),
            },
        }
        manifest["policy_sha256"] = {
            policy.name: policy_sha256(directory, policy.name)
            for policy in sorted((directory / "policies").iterdir())
            if policy.is_dir()
        }
        path = directory / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), "utf-8")
        logger.info("Saved RLlib checkpoint at %d sampled steps: %s", sampled, directory)
        if self.tracker is not None:
            self.tracker.log_model_directory(
                directory,
                name=f"{self.spec.name}-ppo-{sampled}",
                metadata={"sampled_steps": sampled, "observation_schema_id": spec.schema_id},
            )
        return directory

    def _log_iteration(self, iteration: int, sampled: int, target: int) -> None:
        metrics = _iteration_metrics(self._last_result, self.spec.mission)
        logger.info(
            "PPO iteration %d: sampled_steps=%d/%d %s",
            iteration,
            sampled,
            target,
            " ".join(f"{name}={value:.3f}" for name, value in sorted(metrics.items())),
        )
        if self.tracker is not None:
            self.tracker.log_validation(sampled, metrics)


def _sampled_steps(result: Mapping[str, Any]) -> int:
    for scope in (result, result.get("env_runners", {})):
        for key in ("num_env_steps_sampled_lifetime", "timesteps_total"):
            if isinstance(scope, Mapping) and key in scope:
                return int(scope[key])
    return 0


def _iteration_metrics(result: Mapping[str, Any], mission: str) -> dict[str, float]:
    """Last-episode return and counters plus per-policy PPO learner statistics."""

    runners = result.get("env_runners", {})
    history = runners.get("hist_stats", {}) if isinstance(runners, Mapping) else {}
    metrics: dict[str, float] = {}
    returns = history.get("episode_reward", [])
    if returns:
        metrics["episode_return"] = float(returns[-1])
        if mission == "eventsat":
            for name in _LOGGED_EVENTSAT_METRICS:
                values = history.get(f"eventsat_{name}", [])
                if len(values) == len(returns):
                    metrics[f"eventsat_{name}"] = float(values[-1])
    learner = result.get("info", {}).get("learner", {})
    for policy_id, stats in learner.items():
        for name in ("kl", "entropy", "vf_explained_var", "policy_loss", "cur_lr"):
            value = stats.get("learner_stats", {}).get(name)
            if isinstance(value, (int, float)):
                metrics[f"{policy_id}/{name}"] = float(value)
    return metrics


__all__ = ["RLlibPPOTrainer", "load_recipe"]
