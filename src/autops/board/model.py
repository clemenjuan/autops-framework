"""Validated, presentation-independent board evidence models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeWMTreatment:
    mission_mode: str
    plan_hold: int
    horizon: int
    samples: int
    elites: int
    iterations: int
    onboard_compute_w: float
    artifact_sha256: str
    trace_sha256: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class LeWMEvidence:
    treatment: LeWMTreatment
    mean_final_battery_soc: float
    mean_total_energy_consumed_wh: float
    mean_planner_compute_energy_wh: float
    planning_duty_cycle: float
    mean_planning_events: float
    mean_cem_latency_s: float
    rollouts_per_second: float
    model_checkpoint_mb: float
    mean_normalized_probe_error: float


__all__ = ["LeWMEvidence", "LeWMTreatment"]
