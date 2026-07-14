"""Custody ceilings and episode statistics for SSA."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


def discovery_ceiling(
    pass_windows: Iterable[Mapping[str, Any]],
    visibility: Iterable[Mapping[str, Any]],
    target_count: int,
) -> float:
    """Optimistic discoverable fraction that can be observed before the final pass."""

    if target_count <= 0:
        return 0.0
    final_communication: int | None = None
    for window in pass_windows:
        try:
            start = int(window["start_step"])
            end = int(window.get("end_step", start))
        except (KeyError, TypeError, ValueError):
            continue
        if end >= start:
            final_communication = (
                end if final_communication is None else max(final_communication, end)
            )
    if final_communication is None:
        return 0.0
    deliverable = {
        str(object_id)
        for item in visibility
        if int(item.get("step", 0)) < final_communication
        for object_id in item.get("visible_target_ids", [])
    }
    return min(target_count, len(deliverable)) / target_count


def custody_ceiling(
    pass_windows: Iterable[Mapping[str, Any]],
    visibility: Iterable[Mapping[str, Any]],
    target_count: int,
    custody_tau_steps: int,
    max_steps: int,
) -> float:
    """Optimistic time-averaged delivered-custody upper bound."""

    if target_count <= 0 or max_steps <= 0:
        return 0.0
    visible_by_step: dict[int, set[str]] = {}
    for item in visibility:
        step = int(item.get("step", 0))
        visible_by_step.setdefault(step, set()).update(
            str(object_id) for object_id in item.get("visible_target_ids", [])
        )
    pass_starts = {
        int(window["start_step"])
        for window in pass_windows
        if "start_step" in window
        and int(window.get("end_step", window["start_step"])) >= int(window["start_step"])
    }
    latest_visible: dict[str, int] = {}
    latest_ground: dict[str, int] = {}
    cumulative = 0.0
    tau = max(0, int(custody_tau_steps))
    for step in range(max_steps):
        for object_id in visible_by_step.get(step, set()):
            latest_visible[object_id] = step
        if step in pass_starts:
            latest_ground.update(latest_visible)
        fresh = sum(1 for observed in latest_ground.values() if step - observed <= tau)
        cumulative += min(fresh, target_count) / target_count
    return cumulative / max_steps


@dataclass
class EpisodeStats:
    successful_objects: set[str] = field(default_factory=set)
    first_detected_step: dict[str, int] = field(default_factory=dict)
    delivered_step: dict[str, int] = field(default_factory=dict)
    latest_observed_step: dict[str, int] = field(default_factory=dict)
    observation_steps: dict[str, set[int]] = field(default_factory=dict)
    duplicate_observations: int = 0
    total_observation_records: int = 0
    cued_detections: int = 0
    relayed_first_deliveries: int = 0
    isl_attempts: int = 0
    isl_successes: int = 0
    isl_records_relayed: int = 0
    isl_bytes_transferred: float = 0.0
    coverage_auc_sum: float = 0.0
    custody_auc_sum: float = 0.0
    previous_custody: set[str] = field(default_factory=set)
    custody_losses: int = 0

    def record_detection(
        self,
        object_id: str,
        observation_step: int,
        *,
        duplicate: bool,
        cued: bool,
    ) -> None:
        self.successful_objects.add(object_id)
        self.first_detected_step[object_id] = min(
            observation_step,
            self.first_detected_step.get(object_id, observation_step),
        )
        self.latest_observed_step[object_id] = max(
            observation_step,
            self.latest_observed_step.get(object_id, observation_step),
        )
        self.observation_steps.setdefault(object_id, set()).add(observation_step)
        self.total_observation_records += 1
        self.duplicate_observations += int(duplicate)
        self.cued_detections += int(cued)

    def record_first_delivery(self, object_id: str, step: int, relay_hops: int) -> None:
        if object_id in self.delivered_step:
            return
        self.delivered_step[object_id] = step
        self.relayed_first_deliveries += int(relay_hops > 0)

    def update_step(self, delivered: set[str], custody: set[str], target_count: int) -> None:
        denominator = max(1, target_count)
        self.coverage_auc_sum += len(delivered) / denominator
        self.custody_auc_sum += len(custody) / denominator
        self.custody_losses += len(self.previous_custody - custody)
        self.previous_custody = set(custody)

    def snapshot(
        self,
        *,
        current_step: int,
        target_count: int,
        delivered: set[str],
        custody: set[str],
        freshest_ground_steps: Mapping[str, int],
        known_count: int,
        mean_knowledge_latency_steps: float,
    ) -> dict[str, float]:
        elapsed = max(1, current_step)
        denominator = max(1, target_count)
        revisit_intervals = [
            later - earlier
            for steps in self.observation_steps.values()
            for earlier, later in zip(sorted(steps), sorted(steps)[1:], strict=False)
            if later - earlier > 1
        ]
        delivery_latencies = [
            self.delivered_step[object_id] - self.first_detected_step[object_id]
            for object_id in delivered
            if object_id in self.first_detected_step
        ]
        custody_ages = [current_step - freshest_ground_steps[obj] for obj in custody]
        return {
            "ssa_catalog_size": float(target_count),
            "ssa_known_objects": float(known_count),
            "ssa_delivered_objects": float(len(delivered)),
            "ssa_onboard_coverage": known_count / denominator,
            "ssa_delivered_coverage": len(delivered) / denominator,
            "ssa_delivered_coverage_auc": self.coverage_auc_sum / elapsed,
            "ssa_custody_utility": self.custody_auc_sum / elapsed,
            "ssa_custody_fraction_final": len(custody) / denominator,
            "mean_custody_age_steps": _mean(custody_ages),
            "custody_losses": float(self.custody_losses),
            "successful_observations": float(len(self.successful_objects)),
            "duplicate_observations": float(self.duplicate_observations),
            "duplicate_observation_rate": _ratio(
                self.duplicate_observations, self.total_observation_records
            ),
            "cued_observation_fraction": _ratio(
                self.cued_detections, self.total_observation_records
            ),
            "mean_revisit_steps": _mean(revisit_intervals),
            "mean_staleness_steps": _mean(
                current_step - step for step in self.latest_observed_step.values()
            ),
            "mean_delivery_latency_steps": _mean(delivery_latencies),
            "mean_knowledge_latency_steps": mean_knowledge_latency_steps,
            "relayed_delivery_fraction": _ratio(self.relayed_first_deliveries, len(delivered)),
            "isl_connectivity": _ratio(self.isl_successes, self.isl_attempts),
            "isl_records_relayed": float(self.isl_records_relayed),
            "isl_bytes_transferred": self.isl_bytes_transferred,
        }


def _mean(values: Iterable[float | int]) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
