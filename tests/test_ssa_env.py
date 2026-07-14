"""Closed-loop SSA environment contracts and trace semantics."""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from typing import Any

import pytest

from autops.missions.ssa.env import SSAEnvironment
from autops.missions.ssa.transport import published_isl_pairs


def _config(
    *,
    satellites: int = 1,
    steps: int = 6,
    always_visible: bool = True,
) -> dict[str, Any]:
    fixed_satellites = {f"sat_{index}": [7_000.0, float(index), 0.0] for index in range(satellites)}
    return {
        "simulation": {"timestep_s": 60.0, "max_steps": steps},
        "constellation": {
            "size": satellites,
            "fixed_positions_km": fixed_satellites,
        },
        "orbit": {"prefer_orekit": False},
        "targets": {
            "fixed_positions_km": {"visible": [7_000.0, -10.0, 0.0]},
            "fov_half_angle_deg": 2.0,
            "boresight_pitch_deg": 0.0,
            "range_cap_km": 100.0,
            "magnitude_limit": 100.0,
        },
        "payload": {"detection_time_s": 60.0},
        "transitions": {"settling_time_s": 0.0},
        "ground_station": {"always_visible": always_visible},
    }


@pytest.fixture
def sunlit_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("autops.missions.ssa.env")
    monkeypatch.setattr(module, "sun_unit_eci", lambda _epoch_s: (1.0, 0.0, 0.0))


def test_public_ssa_package_exports_environment_and_policy() -> None:
    package = import_module("autops.missions.ssa")
    assert package.SSAEnvironment is SSAEnvironment
    assert package.RuleBasedSSA.__name__ == "RuleBasedSSA"


def test_observe_detect_downlink_pipeline_preserves_record_time(
    sunlit_geometry: None,
) -> None:
    env = SSAEnvironment(_config())
    initial = env.reset(seed=4)
    assert initial["satellites"]["sat_0"]["detection_row"] == [0]

    observed = env.step({"sat_0": {"mode": "payload_observe"}})
    runtime = env.satellites["sat_0"]
    assert runtime.jetson_raw_mb == pytest.approx(2_016.0)
    assert len(runtime.pending_batches) == 1
    assert not runtime.estimates
    assert observed.info["per_satellite"]["sat_0"]["observation_accepted"]

    detected = env.step({"sat_0": {"mode": "payload_detect"}})
    record = runtime.undelivered["visible"]
    assert runtime.jetson_raw_mb == 0.0
    assert runtime.detection_row == [1]
    assert record["obs_step"] == 0
    assert record["known_step"] == 1
    assert env.record_size_bytes == pytest.approx(10.0 * 1_024.0)
    assert detected.info["per_satellite"]["sat_0"]["detections_revealed"] == ["visible"]

    delivered = env.step({"sat_0": {"mode": "communication"}})
    assert not runtime.undelivered
    assert env.ground_archive["visible"][0]["obs_step"] == 0
    assert delivered.info["per_satellite"]["sat_0"]["downlinked_records"] == 1
    assert delivered.observation["global"]["ssa_delivered_objects"] == 1.0


def test_requested_and_resolved_trace_preserve_logical_isl_share(
    sunlit_geometry: None,
) -> None:
    env = SSAEnvironment(_config(satellites=2, steps=2))
    env.reset(seed=3)
    result = env.step(
        {
            "sat_0": {"mode": "isl_share"},
            "sat_1": {"mode": "charging"},
        }
    )
    assert result.info["requested_modes"]["sat_0"] == "isl_share"
    assert result.info["resolved_modes"]["sat_0"] == "isl_share"
    trace = result.info["per_satellite"]["sat_0"]
    assert trace["requested_mode"] == "isl_share"
    assert trace["resolved_mode"] == "isl_share"
    assert trace["logical_mode"] == "isl_share"


def test_support_cut_removes_targets_with_no_episode_access(
    sunlit_geometry: None,
) -> None:
    config = _config(steps=2)
    config["targets"]["fixed_positions_km"] = {
        "visible": [7_000.0, -10.0, 0.0],
        "outside_fov": [7_000.0, 10.0, 0.0],
    }
    env = SSAEnvironment(config)
    observation = env.reset(seed=2)
    assert env.target_ids == ["visible"]
    assert observation["global"]["ssa_support_cut_count"] == 1.0
    assert observation["satellites"]["sat_0"]["detection_row"] == [0]


def test_isl_pair_publication_caches_each_substep_position_once() -> None:
    env = SSAEnvironment(_config(satellites=5, steps=1))
    calls: Counter[tuple[str, float]] = Counter()

    def position(satellite_id: str, epoch_s: float) -> tuple[float, float, float]:
        calls[(satellite_id, epoch_s)] += 1
        index = int(satellite_id.removeprefix("sat_"))
        return 7_000.0, float(index), 0.0

    env._position_provider = position
    pairs = published_isl_pairs(env)
    assert len(pairs) == 10
    assert len(calls) == 5 * 6
    assert set(calls.values()) == {1}


def test_communication_can_prepoint_but_cannot_deliver_without_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SSAEnvironment(_config(always_visible=False))
    env.reset(seed=8)
    runtime = env.satellites["sat_0"]
    runtime.undelivered["visible"] = {
        "object_id": "visible",
        "obs_step": 0,
        "known_step": 0,
        "quality": 1.0,
        "relay_hops": 0,
    }
    dynamics = import_module("autops.missions.ssa.dynamics")
    monkeypatch.setattr(dynamics, "contact_seconds", lambda *_args: 0.0)
    result = env.step({"sat_0": {"mode": "communication"}})
    assert result.info["requested_modes"]["sat_0"] == "communication"
    assert result.info["resolved_modes"]["sat_0"] == "communication"
    assert result.info["per_satellite"]["sat_0"]["contact_seconds"] == 0.0
    assert "visible" in runtime.undelivered
    assert not env.ground_archive["visible"]


def test_custody_uses_record_age_not_delivery_age(sunlit_geometry: None) -> None:
    config = _config(steps=3)
    config["ssa"] = {"custody_tau_steps": 1}
    env = SSAEnvironment(config)
    env.reset(seed=1)
    env.ground_archive["visible"].append({"object_id": "visible", "obs_step": 0})
    assert env.custody_object_ids == {"visible"}
    env.current_step = 2
    assert env.delivered_object_ids == {"visible"}
    assert env.custody_object_ids == set()
