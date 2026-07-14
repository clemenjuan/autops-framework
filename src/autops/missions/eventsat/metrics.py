"""Canonical M-01…M-14 aggregation for EventSat and compatible missions."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any

METRIC_IDS = {
    "M-01": "utility",
    "M-02": "mean_aoi_s",
    "M-03": "peak_aoi_s",
    "M-04": "mean_recovery_steps",
    "M-05": "safety_override_rate",
    "M-06": "resource_efficiency",
    "M-07": "mean_decision_latency_s",
    "M-08": "explainability_coverage",
    "M-09": "robustness_cv",
    "M-10": "scale_efficiency",
    "M-11": "downlink_efficiency",
    "M-12": "value_of_information",
    "M-13": "constraint_violation_rate",
    "M-14": "commanding_effort",
}


@dataclass
class EventSatMetrics:
    config: dict[str, Any]
    max_steps: int
    timestep_s: float
    constellation_size: int = 1
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        info: dict[str, Any],
        *,
        decision_latency_s: float = 0.0,
        inference_allowed: bool = True,
        has_rationale: bool = False,
        ground_latency_s: float = 0.0,
    ) -> None:
        self.rows.append(
            {
                **info,
                "decision_latency_s": decision_latency_s,
                "inference_allowed": inference_allowed,
                "has_rationale": has_rationale,
                "ground_latency_s": ground_latency_s,
            }
        )

    def aggregate(self) -> dict[str, float]:
        if not self.rows:
            return {name: 0.0 for name in METRIC_IDS.values()}
        last = self.rows[-1]
        count = len(self.rows) * self.constellation_size
        metrics_cfg = self.config.get("metrics", {})
        weights = metrics_cfg.get("utility_weights", {})
        objectives = self.config.get("objectives", {})
        episode_days = len(self.rows) * self.timestep_s / 86400.0
        target_scale = episode_days / max(float(objectives.get("mission_duration_days", 90)), 1e-12)
        obs_target = float(objectives.get("total_observation_hours", 2)) * target_scale
        dl_target = float(objectives.get("min_downlinked_data_mb", 221)) * target_scale
        anomaly_events = sum(float(bool(row.get("anomaly_event"))) for row in self.rows)
        obs_ratio = float(last.get("observation_hours", 0.0)) / max(obs_target, 1e-12)
        dl_ratio = float(last.get("data_downlinked_mb", 0.0)) / max(dl_target, 1e-12)
        utility = (
            float(weights.get("observation", 0.0)) * obs_ratio
            + float(weights.get("downlink", 1.0)) * dl_ratio
            - float(weights.get("anomaly_penalty", 0.1)) * anomaly_events / count
        )
        mean_aoi, peak_aoi = self._age_of_information()
        recovery = self._recovery_steps()
        safety = sum(float(row.get("safety_safe", 0.0)) for row in self.rows)
        violations = sum(
            float(bool(row.get("forced", False)) and not row.get("safety_safe", False))
            for row in self.rows
        )
        total_energy = sum(float(row.get("gross_energy_consumed_wh", 0.0)) for row in self.rows)
        decision_rows = [row for row in self.rows if row.get("inference_allowed", True)]
        explained = sum(float(bool(row.get("has_rationale"))) for row in decision_rows)
        max_downlink = float(last.get("max_achievable_downlink_mb", 0.0))
        raw_captured = float(last.get("total_raw_captured_mb", 0.0))
        raw_delivered = float(last.get("downlink_raw_equivalent_mb", 0.0))
        command_count = self._command_count()
        manual_weight = float(metrics_cfg.get("manual_command_weight", 10.0))
        baseline = float(metrics_cfg.get("baseline_utility_n1", 0.0))
        scale = utility / self.constellation_size / baseline if baseline > 0 else 0.0
        result = {
            "utility": utility,
            "mean_aoi_s": mean_aoi,
            "peak_aoi_s": peak_aoi,
            "mean_recovery_steps": mean(recovery) if recovery else 0.0,
            "safety_override_rate": safety / count,
            "resource_efficiency": utility / total_energy if total_energy else 0.0,
            "mean_decision_latency_s": mean(
                [float(row.get("decision_latency_s", 0.0)) for row in decision_rows]
            )
            if decision_rows
            else 0.0,
            "explainability_coverage": explained / len(decision_rows) if decision_rows else 0.0,
            "robustness_cv": 0.0,
            "scale_efficiency": scale,
            "downlink_efficiency": float(last.get("data_downlinked_mb", 0.0)) / max_downlink
            if max_downlink
            else 0.0,
            "value_of_information": raw_delivered / raw_captured if raw_captured else 0.0,
            "constraint_violation_rate": violations / count,
            "commanding_effort": (command_count + manual_weight * anomaly_events) / episode_days
            if episode_days
            else 0.0,
            "observation_hours": float(last.get("observation_hours", 0.0)),
            "downlinked_mb": float(last.get("data_downlinked_mb", 0.0)),
            "final_battery_soc": float(last.get("battery_soc", 0.0)),
            "total_energy_consumed_wh": total_energy,
            "safety_overrides": safety,
            "anomaly_events": anomaly_events,
            "constraint_violations": violations,
            "command_count": float(command_count),
            "total_raw_captured_mb": raw_captured,
            "downlink_raw_equivalent_mb": raw_delivered,
            "max_achievable_downlink_mb": max_downlink,
        }
        result.update(
            {
                metric_id.lower().replace("-", "_"): result[name]
                for metric_id, name in METRIC_IDS.items()
            }
        )
        return result

    def _age_of_information(self) -> tuple[float, float]:
        age = 0.0
        values: list[float] = []
        for row in self.rows:
            age = 0.0 if float(row.get("step_downlinked_mb", 0.0)) > 0 else age + self.timestep_s
            values.append(age)
        return mean(values), max(values)

    def _recovery_steps(self) -> list[int]:
        start: int | None = None
        durations: list[int] = []
        for index, row in enumerate(self.rows):
            if row.get("anomaly_event") and start is None:
                start = index
            if (
                start is not None
                and index > start
                and not row.get("anomaly_active")
                and not row.get("safety_safe")
            ):
                durations.append(index - start)
                start = None
        if start is not None:
            durations.append(len(self.rows) - start)
        return durations

    def _command_count(self) -> int:
        count = 0
        previous: str | None = None
        for row in self.rows:
            requested = str(row.get("requested_mode", ""))
            if requested and requested != previous:
                count += 1
                previous = requested
        return count


def experiment_statistics(episodes: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = sorted({key for episode in episodes for key in episode})
    means: dict[str, float] = {}
    deviations: dict[str, float] = {}
    for key in keys:
        values = [float(episode.get(key, 0.0)) for episode in episodes]
        means[key] = mean(values)
        deviations[key] = stdev(values) if len(values) > 1 else 0.0
    utility_mean = means.get("utility", 0.0)
    means["robustness_cv"] = (
        deviations.get("utility", 0.0) / utility_mean if utility_mean > 0 else 0.0
    )
    means["m_09"] = means["robustness_cv"]
    return {"mean": means, "std": deviations}
