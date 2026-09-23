"""SSA lifecycle composed from one environment and five organisation controllers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from autops.config import ExperimentSpec, asset_root
from autops.core.provenance import collect_provenance
from autops.core.runner import policy_identity
from autops.missions.eventsat.metrics import experiment_statistics
from autops.missions.ssa.env import SSAEnvironment
from autops.organisations import bind_communication_topology, create_organisation
from autops.organisations.loops import organisation_options
from autops.rl.policy import merge_identities


def _episode_config(spec: ExperimentSpec) -> dict[str, Any]:
    config = deepcopy(spec.mission_config)
    config.setdefault("simulation", {})["max_steps"] = spec.steps
    config.setdefault("simulation", {})["timestep_s"] = spec.timestep_s
    config.setdefault("constellation", {})["size"] = spec.constellation_size
    return config


def ssa_environment(spec: ExperimentSpec) -> SSAEnvironment:
    """The SSA truth environment of a coordinate; countdowns are published to RL only."""

    return SSAEnvironment(_episode_config(spec), event_countdowns=spec.onboard_token == "rl")


def _organisation_config(spec: ExperimentSpec) -> dict[str, Any]:
    config = organisation_options(spec)
    # Representation overrides reach every agent's plugin, as on the EventSat runner.
    policy = {**config.get("policy", {}), **spec.representation_config}
    custody = spec.mission_config.get("ssa", {}).get("custody_tau_steps", 4320)
    relay = spec.mission_config.get("ssa", {}).get("relay_preemption_age_steps", custody // 8)
    policy.setdefault("custody_tau_steps", custody)
    policy.setdefault("isl_aoi_threshold_steps", relay)
    config["policy"] = policy
    config["representation"] = spec.onboard_token
    return config


def _run_episode(spec: ExperimentSpec, episode_id: int, seed: int) -> dict[str, Any]:
    env = ssa_environment(spec)
    controller = create_organisation(spec.organisation, _organisation_config(spec))
    observation = env.reset(seed)
    controller.reset(seed, observation)
    bind_communication_topology(controller.organisation, env)
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
    diagnostics = (
        {
            "onboard": {
                "policy_identity": merge_identities(
                    [
                        policy.diagnostics()["policy_identity"]
                        for policy in controller.policies.values()
                    ]
                )
            }
        }
        if spec.onboard_token == "rl"
        else {}
    )
    return {
        "episode_id": episode_id,
        "decision_diagnostics": diagnostics,
        "seed": seed,
        "steps": int(observation["step"]),
        "total_reward": total_reward,
        "metrics": metrics,
        "provenance": {
            "target_count": len(env.target_ids),
            "orbital_backend": env.config["orbit"]["propagator"],
            "support_cut_count": env.support_cut_count,
        },
    }


def run_ssa_experiment(spec: ExperimentSpec) -> dict[str, Any]:
    """Run a paired-seed SSA experiment and return the shared result envelope."""

    episodes = [_run_episode(spec, episode_id, seed) for episode_id, seed in enumerate(spec.seeds)]
    statistics = experiment_statistics(
        [episode["metrics"] for episode in episodes], include_robustness=False
    )
    mean_metrics = statistics["mean"]
    experiment = spec.model_dump(mode="json")
    if spec.onboard_token == "rl":
        experiment["rl_policy_identity"] = policy_identity(episodes)
    return {
        "schema_version": 1,
        "experiment": experiment,
        "metric_registry": {name: name for name in sorted(mean_metrics)},
        "metrics": mean_metrics,
        "statistics": statistics,
        "episodes": episodes,
        "provenance": collect_provenance(spec.model_dump(mode="json"), asset_root()),
    }


__all__ = ["run_ssa_experiment", "ssa_environment"]
