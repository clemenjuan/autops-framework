"""Resettable compute evidence for deployed world-model planners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autops.wm.artifact import PlannerArtifact, artifact_sha256
from autops.wm.cem import CEMConfig


@dataclass
class PlannerComputeEvidence:
    planning_events: int = 0
    held_action_steps: int = 0
    held_action_repairs: int = 0
    reflex_overrides: int = 0
    cem_latency_total_s: float = 0.0
    evaluated_rollouts: int = 0

    def reset(self) -> None:
        self.planning_events = 0
        self.held_action_steps = 0
        self.held_action_repairs = 0
        self.reflex_overrides = 0
        self.cem_latency_total_s = 0.0
        self.evaluated_rollouts = 0

    def record_plan(self, latency_s: float, *, samples: int, iterations: int) -> None:
        self.planning_events += 1
        self.cem_latency_total_s += max(0.0, float(latency_s))
        self.evaluated_rollouts += int(samples) * int(iterations)

    def to_dict(self, cem: CEMConfig, artifact: PlannerArtifact) -> dict[str, Any]:
        probe_errors = dict(artifact.probe_evidence.rmse_over_std)
        finite_probe_errors = [value for value in probe_errors.values() if value is not None]
        mean_probe_error = (
            sum(finite_probe_errors) / len(finite_probe_errors) if finite_probe_errors else None
        )
        mean_latency = (
            self.cem_latency_total_s / self.planning_events if self.planning_events else 0.0
        )
        throughput = (
            self.evaluated_rollouts / self.cem_latency_total_s
            if self.cem_latency_total_s > 0.0
            else 0.0
        )
        return {
            "planning_events": self.planning_events,
            "held_action_steps": self.held_action_steps,
            "held_action_repairs": self.held_action_repairs,
            "future_action_repair_rate": (
                self.held_action_repairs / self.held_action_steps if self.held_action_steps else 0.0
            ),
            "reflex_overrides": self.reflex_overrides,
            "cem_latency_total_s": self.cem_latency_total_s,
            "cem_latency_mean_s": mean_latency,
            "evaluated_rollouts": self.evaluated_rollouts,
            "rollouts_per_second": throughput,
            "plan_hold": cem.plan_hold,
            "horizon": cem.horizon,
            "samples": cem.samples,
            "elites": cem.elites,
            "iterations": cem.iterations,
            "checkpoint_size_bytes": artifact.probe_evidence.checkpoint_size_bytes,
            "probe_rmse_over_std_mean": mean_probe_error,
            "probe_rmse_over_std": probe_errors,
            "artifact_identity": {
                "schema_version": artifact.schema_version,
                "trace_sha256": artifact.model.trace_sha256,
                "sha256": artifact_sha256(artifact),
            },
            "checkpoint_identity": {
                "relative_path": artifact.model.checkpoint,
                "sha256": artifact.model.checkpoint_sha256,
            },
        }


__all__ = ["PlannerComputeEvidence"]
