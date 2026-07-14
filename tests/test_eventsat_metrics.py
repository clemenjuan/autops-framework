"""Direct arithmetic tests for the canonical EventSat M-01…M-14 registry."""

from __future__ import annotations

import math

import pytest

from autops.missions.eventsat.metrics import (
    METRIC_IDS,
    EventSatMetrics,
    experiment_statistics,
)

CONFIG = {
    "objectives": {
        "total_observation_hours": 2.0,
        "min_downlinked_data_mb": 10.0,
        "mission_duration_days": 1.0,
    },
    "metrics": {
        "utility_weights": {
            "observation": 0.25,
            "downlink": 0.75,
            "anomaly_penalty": 0.1,
        },
        "manual_command_weight": 10.0,
        "baseline_utility_n1": 0.975,
    },
}


def _collector() -> EventSatMetrics:
    collector = EventSatMetrics(CONFIG, max_steps=4, timestep_s=21_600.0)
    rows = [
        (
            {
                "requested_mode": "charging",
                "anomaly_event": "thermal_warning",
                "anomaly_active": True,
                "safety_safe": 1.0,
                "forced": True,
                "gross_energy_consumed_wh": 1.0,
                "step_downlinked_mb": 0.0,
            },
            0.1,
            True,
            True,
        ),
        (
            {
                "requested_mode": "charging",
                "anomaly_active": True,
                "safety_safe": 1.0,
                "forced": True,
                "gross_energy_consumed_wh": 2.0,
                "step_downlinked_mb": 1.0,
            },
            0.2,
            False,
            False,
        ),
        (
            {
                "requested_mode": "payload_observe",
                "anomaly_active": False,
                "safety_safe": 0.0,
                "forced": True,
                "gross_energy_consumed_wh": 3.0,
                "step_downlinked_mb": 0.0,
            },
            0.3,
            True,
            False,
        ),
        (
            {
                "requested_mode": "communication",
                "anomaly_active": False,
                "safety_safe": 0.0,
                "forced": False,
                "gross_energy_consumed_wh": 4.0,
                "step_downlinked_mb": 0.0,
                "observation_hours": 2.0,
                "data_downlinked_mb": 10.0,
                "total_raw_captured_mb": 20.0,
                "downlink_raw_equivalent_mb": 5.0,
                "max_achievable_downlink_mb": 20.0,
                "battery_soc": 0.8,
            },
            0.5,
            True,
            True,
        ),
    ]
    for info, latency, inference_allowed, rationale in rows:
        collector.record(
            info,
            decision_latency_s=latency,
            inference_allowed=inference_allowed,
            has_rationale=rationale,
        )
    return collector


def test_metric_registry_and_direct_m01_to_m14_formulas() -> None:
    metrics = _collector().aggregate()
    assert set(METRIC_IDS) == {f"M-{index:02d}" for index in range(1, 15)}
    assert set(METRIC_IDS.values()).issubset(metrics)
    expected = {
        "M-01": 0.975,
        "M-02": 21_600.0,
        "M-03": 43_200.0,
        "M-04": 2.0,
        "M-05": 0.5,
        "M-06": 0.0975,
        "M-07": 0.3,
        "M-08": 2.0 / 3.0,
        "M-09": 0.0,
        "M-10": 1.0,
        "M-11": 0.5,
        "M-12": 0.25,
        "M-13": 0.25,
        "M-14": 13.0,
    }
    for metric_id, value in expected.items():
        name = METRIC_IDS[metric_id]
        assert metrics[name] == pytest.approx(value)
        assert metrics[metric_id.lower().replace("-", "_")] == pytest.approx(value)


def test_m09_uses_sample_utility_coefficient_of_variation() -> None:
    statistics = experiment_statistics([{"utility": 1.0}, {"utility": 3.0}])
    expected = math.sqrt(2.0) / 2.0
    assert statistics["mean"]["robustness_cv"] == pytest.approx(expected)
    assert statistics["mean"]["m_09"] == pytest.approx(expected)
