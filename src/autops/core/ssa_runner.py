"""SSA lifecycle composed from one environment and five organisation controllers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autops.config import ExperimentSpec, deep_merge, repository_root
from autops.core.provenance import collect_provenance
from autops.missions.eventsat.metrics import experiment_statistics
from autops.missions.ssa.env import SSAEnvironment
from autops.organisations.ssa import create_organisation


def _episode_config(spec: ExperimentSpec) -> dict[str, Any]:
    config = deepcopy(spec.mission_config)
    config.setdefault("simulation", {})["max_steps"] = spec.steps
    config.setdefault("simulation", {})["timestep_s"] = spec.timestep_s
    config.setdefault("constellation", {})["size"] = spec.constellation_size
    return config


def _organisation_config(spec: ExperimentSpec) -> dict[str, Any]:
    defaults = (
        spec.mission_config.get("organisation_defaults", {}).get(spec.organisation, {})
    )
    config = deep_merge(defaults, spec.organisation_config)
    policy = dict(config.get("policy", {}))
    custody = spec.mission_config.get("ssa", {}).get("custody_tau_steps", 4320)
    relay = spec.mission_config.get("ssa", {}).get("relay_preemption_age_steps", custody // 8)
    policy.setdefault("custody_tau_steps", custody)
    policy.setdefault("isl_aoi_threshold_steps", relay)
    config["policy"] = policy
    return config


def _run_episode(spec: ExperimentSpec, episode_id: int, seed: int) -> dict[str, Any]:
    env = SSAEnvironment(_episode_config(spec))
    controller = create_organisation(spec.organisation, _organisation_config(spec))
    observation = env.reset(seed)
    controller.reset(seed, observation)
    total_reward = 0.0
    while int(observation["step"]) < spec.steps:
        actions = controller.act(observation)
        transition = env.step(actions)
        total_reward += transition.reward
        controller.after_step(transition.info, transition.observation)
        observation = transition.observation
        if transition.done:
            break
    metrics = {**env.episode_metrics(), **controller.metrics()}
    return {
        "episode_id": episode_id,
        "seed": seed,
        "steps": int(observation["step"]),
        "total_reward": total_reward,
        "metrics": metrics,
        "provenance": {
            "target_count": len(env.target_ids),
            "support_cut_count": env.support_cut_count,
        },
    }


def run_ssa_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    """Run a paired-seed SSA experiment and return the shared result envelope."""

    episodes = [
        _run_episode(spec, episode_id, seed)
        for episode_id, seed in enumerate(spec.seeds)
    ]
    statistics = experiment_statistics([episode["metrics"] for episode in episodes])
    mean_metrics = statistics["mean"]
    return {
        "schema_version": 1,
        "experiment": spec.model_dump(mode="json"),
        "metric_registry": {name: name for name in sorted(mean_metrics)},
        "metrics": mean_metrics,
        "statistics": statistics,
        "episodes": episodes,
        "provenance": collect_provenance(spec.model_dump(mode="json"), repository_root()),
    }


__all__ = ["run_ssa_experiment"]
