"""Fallback-physics and first runnable EventSat coordinate smoke tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner
from autops.core.types import DecisionContext
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.metrics import METRIC_IDS
from autops.missions.eventsat.physics import planner_event_energy_wh
from autops.orbital import GroundPass
from autops.paradigms.base import ParadigmDecision, refresh_almanac
from autops.representations.symb import EventSatSymbolicScheduler


def test_fallback_environment_uses_configured_physics() -> None:
    config = deepcopy(expand_coordinate("eventsat/sas/ag/symb").mission_config)
    config["anomalies"]["probability_per_step"] = 0.0
    env = EventSatEnvironment(config, max_steps=4, prefer_orekit=False)
    observation = env.reset(42)
    metadata = observation["satellites"]["eventsat_0"]["metadata"]
    assert observation["global"]["orbital_backend"] == "simplified"
    assert metadata["downlink_rate_kbps"] == 50.0
    assert metadata["orbital_period_steps"] == 92
    assert metadata["settling_time_steps"] == 2
    assert not metadata["in_sunlight"]
    assert all(0.0 <= value <= 360.0 for value in env.state.orbit_elements.values())

    step = env.step({"eventsat_0": {"mode": "charging", "jetson_planned": False}})
    expected_gross_wh = 4.32 * 60.0 / 3_600.0
    assert step.info["gross_energy_consumed_wh"] == pytest.approx(expected_gross_wh)
    assert step.info["solar_generation_wh"] == 0.0
    assert step.info["battery_soc"] == pytest.approx(0.8 - expected_gross_wh / 70.0)


def test_planner_energy_uses_measured_event_time_and_never_double_counts_jetson() -> None:
    config = deepcopy(expand_coordinate("eventsat/sas/ao/symb").mission_config)
    config["power"]["onboard_compute_w"] = 21.0
    config["power"]["planner_compute"] = {
        "boot_energy_wh": 0.01,
        "idle_power_w": 2.0,
        "idle_time_s": 3.0,
        "evidence_source": "assumed",
    }

    energy = planner_event_energy_wh(config, "charging", active_time_s=1.2)

    assert energy == pytest.approx(21.0 * 1.2 / 3600.0 + 0.01 + 2.0 * 3.0 / 3600.0)
    assert planner_event_energy_wh(config, "payload_compress", active_time_s=1.2) == 0.0


def test_refresh_almanac_updates_clock_without_truth_or_resource_leaks() -> None:
    stale = {
        "step": 4,
        "epoch_s": 240.0,
        "satellites": {
            "eventsat_0": {
                "status": "charging",
                "resources": {"battery_soc": 0.4},
                "metadata": {
                    "time_to_next_pass": 88.0,
                    "following_gap_steps": 90.0,
                    "future_pass_capacity_mb": 1.0,
                    "health_status": "thermal_warning",
                    "physical_ground_pass_active": False,
                },
            }
        },
    }
    current = {
        "step": 20,
        "epoch_s": 1_200.0,
        "satellites": {
            "eventsat_0": {
                "status": "safe",
                "resources": {"battery_soc": 0.9},
                "metadata": {
                    "time_to_next_pass": 0.0,
                    "following_gap_steps": 71.0,
                    "future_pass_capacity_mb": 2.5,
                    "health_status": "nominal",
                    "physical_ground_pass_active": True,
                },
            }
        },
    }
    refreshed = refresh_almanac(stale, current)
    satellite = refreshed["satellites"]["eventsat_0"]
    assert refreshed["step"] == 20
    assert refreshed["epoch_s"] == 1_200.0
    assert satellite["metadata"]["time_to_next_pass"] == 0.0
    assert satellite["metadata"]["following_gap_steps"] == 71.0
    assert satellite["metadata"]["future_pass_capacity_mb"] == 2.5
    assert satellite["resources"]["battery_soc"] == 0.4
    assert satellite["status"] == "charging"
    assert satellite["metadata"]["health_status"] == "thermal_warning"
    assert not satellite["metadata"]["physical_ground_pass_active"]


def test_conventional_scheduler_targets_the_following_gap() -> None:
    scheduler = EventSatSymbolicScheduler({"conventional": True})
    output = scheduler.select_action(
        DecisionContext(
            state={
                "planning_gap_steps": 5,
                "following_gap_steps": 17,
                "future_pass_capacity_mb": 0.0,
                "settling_time_steps": 2,
            },
            observation={},
            memory=None,
            step=0,
            role="ground",
        )
    )
    assert sum(block["steps"] for block in output["schedule"]) == 17


def test_eventsat_ag_symbolic_runner_exposes_all_fourteen_metrics() -> None:
    spec = expand_coordinate(
        "eventsat/sas/ag/symb",
        episodes=1,
        steps=24,
        seeds=[17],
        overrides={"mission": {"anomalies": {"probability_per_step": 0.0}}},
    )
    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    assert result["metric_registry"] == METRIC_IDS
    assert set(result["metrics"]) == set(METRIC_IDS)
    assert result["episodes"][0]["steps"] == 24
    assert result["episodes"][0]["provenance"]["orbital_backend"] == "simplified"
    assert all(isinstance(value, float) for value in result["metrics"].values())


def test_runner_persists_planner_diagnostics_and_existing_compute_energy(
    monkeypatch,
) -> None:
    class FakeOnboard:
        def diagnostics(self) -> dict[str, int]:
            return {"planning_events": 3}

    class FakeParadigm:
        onboard = FakeOnboard()

        def reset(self, seed: int, observation: dict) -> None:
            del seed, observation

        def act(self, observation: dict, *, physical_contact: bool) -> ParadigmDecision:
            del observation, physical_contact
            return ParadigmDecision(
                {
                    "eventsat_0": {
                        "mode": "charging",
                        "jetson_planned": True,
                        "planner_active_s": 2.0,
                    }
                }
            )

        def after_step(self, info: dict, observation: dict) -> None:
            del info, observation

    monkeypatch.setattr(ExperimentRunner, "_build_paradigm", lambda self: FakeParadigm())
    spec = expand_coordinate(
        "eventsat/sas/ag/symb",
        episodes=1,
        steps=3,
        seeds=[17],
        overrides={"mission": {"anomalies": {"probability_per_step": 0.0}}},
    )

    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    episode = result["episodes"][0]

    assert episode["planner_compute_energy_wh"] == pytest.approx(7.0 * 6.0 / 3600.0)
    assert episode["decision_diagnostics"]["onboard"] == {"planning_events": 3}
    assert "planner_compute_energy_wh" not in result["metric_registry"].values()


def test_pass_entry_horizon_spans_the_gap_to_the_next_pass() -> None:
    """A step straddling a pass start must not collapse the planning horizon.

    Ground paradigms plan exactly once per pass, on its first step. Real pass
    boundaries do not align to step edges, so that step sees the current pass
    as both current and still-upcoming; measuring the gap against it yields a
    negative span that clamps to one step and leaves ag/ah commanding a single
    step per pass. The simplified backend aligns passes to step edges and never
    exposes this, so the boundary is asserted directly.
    """

    config = deepcopy(expand_coordinate("eventsat/sas/ag/symb").mission_config)
    config["anomalies"]["probability_per_step"] = 0.0
    env = EventSatEnvironment(config, max_steps=200, prefer_orekit=False)
    env.reset(42)

    entry_step = 10
    step_s = env.timestep_s
    # The pass opens 12 s into step 10 and the next two follow 60 and 30 steps later.
    current = GroundPass(entry_step * step_s + 12.0, 900.0, 40.0, 5.0)
    following = GroundPass(4_500.0, 4_800.0, 40.0, 5.0)
    third = GroundPass(6_600.0, 6_900.0, 40.0, 5.0)
    env.orbit = replace(env.orbit, ground_passes=(current, following, third))
    env.state.step = entry_step

    assert env.physical_contact_active()
    metadata = env.observe()["satellites"]["eventsat_0"]["metadata"]
    assert metadata["planning_gap_steps"] == pytest.approx((4_500.0 - 900.0) / step_s)
    assert metadata["following_gap_steps"] == pytest.approx((6_600.0 - 4_800.0) / step_s)
