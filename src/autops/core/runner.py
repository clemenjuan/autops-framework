"""Compact experiment lifecycle for every mission and matrix coordinate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autops.config import ExperimentSpec, asset_root, runtime_root
from autops.core.plugin import create_representation
from autops.core.provenance import collect_provenance
from autops.memory.fixed import FixedMemory
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.metrics import METRIC_IDS, EventSatMetrics, experiment_statistics
from autops.paradigms.ag import AutonomousGround
from autops.paradigms.ah import AutonomousHybrid
from autops.paradigms.ao import AutonomousOnboard
from autops.paradigms.cg import ConventionalGround


@dataclass
class ExperimentRunner:
    spec: ExperimentSpec
    save: bool = True
    prefer_orekit: bool = True

    def run(self) -> dict[str, Any]:
        if self.spec.mission != "eventsat":
            return self._run_ssa()
        episodes = [
            self._run_eventsat_episode(index, seed) for index, seed in enumerate(self.spec.seeds)
        ]
        statistics = experiment_statistics([item["metrics"] for item in episodes])
        result = self._result_document(episodes, statistics)
        if self.save:
            self._write_result(result)
        return result

    def _run_eventsat_episode(self, episode_id: int, seed: int) -> dict[str, Any]:
        env = EventSatEnvironment(
            self.spec.mission_config,
            max_steps=self.spec.steps,
            onboard_compute_active=self.spec.onboard_uses_jetson,
            anomaly_requires_ground_pass=self.spec.paradigm in {"ag", "conventional"},
            prefer_orekit=self.prefer_orekit,
        )
        paradigm = self._build_paradigm()
        observation = env.reset(seed)
        paradigm.reset(seed, observation)
        collector = EventSatMetrics(
            self.spec.mission_config,
            self.spec.steps,
            self.spec.timestep_s,
        )
        total_reward = 0.0
        planner_compute_energy_wh = 0.0
        while int(observation["step"]) < self.spec.steps:
            decision = paradigm.act(observation, physical_contact=env.physical_contact_active())
            step = env.step(decision.actions)
            total_reward += step.reward
            planner_compute_energy_wh += float(step.info.get("planner_compute_energy_wh", 0.0))
            collector.record(
                step.info,
                decision_latency_s=decision.latency_s,
                inference_allowed=decision.inference_allowed,
                has_rationale=bool(decision.rationale),
                ground_latency_s=decision.ground_latency_s,
            )
            paradigm.after_step(step.info, step.observation)
            observation = step.observation
            if step.done:
                break
        return {
            "episode_id": episode_id,
            "seed": seed,
            "steps": int(observation["step"]),
            "total_reward": total_reward,
            "planner_compute_energy_wh": planner_compute_energy_wh,
            "metrics": collector.aggregate(),
            "provenance": env.episode_provenance(),
            "decision_diagnostics": self._decision_diagnostics(paradigm),
        }

    def _decision_diagnostics(self, paradigm: Any) -> dict[str, Any]:
        diagnostics: dict[str, Any] = {}
        for role in ("onboard", "ground"):
            representation = getattr(paradigm, role, None)
            collect = getattr(representation, "diagnostics", None)
            if callable(collect):
                diagnostics[role] = collect()
        return diagnostics

    def _build_paradigm(self):
        memory = FixedMemory()
        rep_config = dict(self.spec.representation_config)
        if self.spec.paradigm == "ao":
            onboard = create_representation(
                self.spec.mission, self.spec.onboard_token or "", "onboard", rep_config
            )
            return AutonomousOnboard(onboard, memory)
        if self.spec.paradigm in {"ag", "conventional"}:
            rep_config["conventional"] = self.spec.paradigm == "conventional"
            ground = create_representation(
                self.spec.mission, self.spec.ground_token or "", "ground", rep_config
            )
            return (
                ConventionalGround(ground, memory)
                if self.spec.paradigm == "conventional"
                else AutonomousGround(ground, memory)
            )
        onboard = create_representation(
            self.spec.mission, self.spec.onboard_token or "", "onboard", rep_config
        )
        ground = create_representation(
            self.spec.mission, self.spec.ground_token or "", "ground", rep_config
        )
        return AutonomousHybrid(onboard, ground, memory)

    def _run_ssa(self) -> dict[str, Any]:
        from autops.core.ssa_runner import run_ssa_experiment

        result = run_ssa_experiment(self.spec)
        if self.save:
            self._write_result(result)
        return result

    def _result_document(
        self, episodes: list[dict[str, Any]], statistics: dict[str, dict[str, float]]
    ) -> dict[str, Any]:
        mean_metrics = statistics["mean"]
        experiment = self.spec.model_dump(mode="json")
        if self.spec.onboard_token == "lewm-cem":
            identities = []
            for episode in episodes:
                diagnostics = episode.get("decision_diagnostics", {}).get("onboard", {})
                artifact = diagnostics.get("artifact_identity")
                checkpoint = diagnostics.get("checkpoint_identity")
                if not isinstance(artifact, dict) or not isinstance(checkpoint, dict):
                    raise ValueError("LeWM result lacks planner artifact identity")
                identities.append(
                    {
                        "schema_version": artifact.get("schema_version"),
                        "artifact_sha256": artifact.get("sha256"),
                        "trace_sha256": artifact.get("trace_sha256"),
                        "checkpoint_sha256": checkpoint.get("sha256"),
                    }
                )
            if any(identity != identities[0] for identity in identities[1:]):
                raise ValueError("LeWM episodes used inconsistent planner artifacts")
            experiment["planner_artifact_identity"] = identities[0]
        return {
            "schema_version": 1,
            "experiment": experiment,
            "metric_registry": METRIC_IDS,
            "metrics": {
                metric_id: mean_metrics.get(name, 0.0) for metric_id, name in METRIC_IDS.items()
            },
            "statistics": statistics,
            "episodes": episodes,
            "provenance": collect_provenance(experiment, asset_root()),
        }

    def _write_result(self, result: dict[str, Any]) -> Path:
        digest = str(result.get("provenance", {}).get("config_sha256", ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("result provenance must contain a lowercase config SHA-256")
        coordinate = Path(*self.spec.coordinate.split("/"))
        root = runtime_root() / self.spec.output_root / coordinate / digest[:12]
        root.mkdir(parents=True, exist_ok=True)
        destination = root / "results.json"
        temporary = root / ".results.json.tmp"
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(destination)
        return destination
