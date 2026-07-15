"""Closed-loop SSA environment contracts and trace semantics."""

from __future__ import annotations

import json
from collections import Counter
from importlib import import_module
from typing import Any

import pytest

from autops.missions.ssa.env import SSAEnvironment
from autops.missions.ssa.transport import apply_isl, published_isl_pairs


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


def test_dynamic_isl_pair_publication_recomputes_each_step() -> None:
    env = SSAEnvironment(_config(satellites=5, steps=1))
    calls: Counter[tuple[str, float]] = Counter()

    def position(satellite_id: str, epoch_s: float) -> tuple[float, float, float]:
        calls[(satellite_id, epoch_s)] += 1
        index = int(satellite_id.removeprefix("sat_"))
        return 7_000.0, float(index), 0.0

    env._position_provider = position
    initial_pairs = published_isl_pairs(env)
    env.current_step = 1
    later_pairs = published_isl_pairs(env)
    assert initial_pairs == later_pairs
    assert len(initial_pairs) == 10
    assert len(calls) == 2 * 5 * 6
    assert set(calls.values()) == {1}
    assert env._isl_capacity_cache is None


def test_shared_plane_isl_pair_publication_reuses_episode_capacities() -> None:
    config = _config(satellites=12, steps=1)
    config["constellation"]["share_plane"] = True
    env = SSAEnvironment(config)
    calls: Counter[tuple[str, float]] = Counter()

    def position(satellite_id: str, epoch_s: float) -> tuple[float, float, float]:
        calls[(satellite_id, epoch_s)] += 1
        index = int(satellite_id.removeprefix("sat_"))
        return 7_000.0, float(index), 0.0

    env._position_provider = position
    initial_pairs = published_isl_pairs(env)
    env.current_step = 10_000
    assert published_isl_pairs(env) == initial_pairs
    assert len(initial_pairs) == 66
    assert ["sat_2", "sat_10"] in initial_pairs
    assert len(calls) == 12 * 6
    assert set(calls.values()) == {1}
    assert env._isl_capacity_cache is not None
    assert ("sat_10", "sat_2") in env._isl_capacity_cache


def test_shared_plane_isl_capacity_cache_is_invalidated_by_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("autops.missions.ssa.env")
    original = module.link_capacity_bytes
    calls = 0

    def counted_capacity(*args: Any, **kwargs: Any) -> float:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "link_capacity_bytes", counted_capacity)
    config = _config(satellites=4, steps=1)
    config["constellation"]["share_plane"] = True
    env = SSAEnvironment(config)
    env.reset(seed=1)
    first_cache = env._isl_capacity_cache
    assert calls == 6

    env.reset(seed=2)
    assert calls == 12
    assert env._isl_capacity_cache is not first_cache


def test_apply_isl_uses_cached_capacities_with_mode_gating_and_unicast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(satellites=4, steps=1)
    config["constellation"]["share_plane"] = True
    env = SSAEnvironment(config)
    env.reset(seed=1)
    env._isl_capacity_cache = {
        ("sat_0", "sat_1"): env.record_size_bytes,
        ("sat_0", "sat_2"): 2.0 * env.record_size_bytes,
        ("sat_0", "sat_3"): 3.0 * env.record_size_bytes,
        ("sat_1", "sat_2"): 0.0,
        ("sat_1", "sat_3"): 0.0,
        ("sat_2", "sat_3"): 0.0,
    }
    env.satellites["sat_0"].undelivered["visible"] = {
        "object_id": "visible",
        "obs_step": 0,
        "quality": 1.0,
    }
    transport = import_module("autops.missions.ssa.transport")

    def unexpected_recompute(*_args: Any, **_kwargs: Any) -> float:
        raise AssertionError("shared-plane ISL capacity should come from the episode cache")

    monkeypatch.setattr(transport, "link_capacity_bytes", unexpected_recompute)
    modes = {
        "sat_0": "isl_share",
        "sat_1": "charging",
        "sat_2": "safe",
        "sat_3": "payload_detect",
    }
    per_satellite = {satellite_id: {} for satellite_id in env.satellite_ids}
    apply_isl(env, modes, 0.0, per_satellite)

    assert env.stats.isl_attempts == 3
    assert env.stats.isl_successes == 2
    assert per_satellite["sat_0"]["isl_feasible_receivers"] == ["sat_1", "sat_2"]
    assert "visible" not in env.satellites["sat_0"].undelivered
    assert "visible" not in env.satellites["sat_1"].undelivered
    assert "visible" in env.satellites["sat_2"].undelivered
    assert "visible" not in env.satellites["sat_3"].undelivered


def test_apply_isl_recomputes_capacity_without_shared_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = SSAEnvironment(_config(satellites=2, steps=1))
    env.reset(seed=1)
    calls: list[tuple[str, str, float, float]] = []
    transport = import_module("autops.missions.ssa.transport")

    def dynamic_capacity(
        _position_at: Any,
        left: str,
        right: str,
        start_s: float,
        end_s: float,
        _budget: Any,
        **_kwargs: Any,
    ) -> float:
        calls.append((left, right, start_s, end_s))
        return env.record_size_bytes

    monkeypatch.setattr(transport, "link_capacity_bytes", dynamic_capacity)
    per_satellite = {satellite_id: {} for satellite_id in env.satellite_ids}
    apply_isl(env, {"sat_0": "isl_share", "sat_1": "charging"}, 600.0, per_satellite)

    assert calls == [("sat_0", "sat_1", 600.0, 660.0)]
    assert env.stats.isl_attempts == 1
    assert env.stats.isl_successes == 1
    assert env._isl_capacity_cache is None


def test_shared_plane_cache_is_trace_equivalent_to_dynamic_path() -> None:
    cached_config = _config(satellites=3, steps=3)
    cached_config["constellation"]["share_plane"] = True
    dynamic_config = _config(satellites=3, steps=3)
    dynamic_config["constellation"]["share_plane"] = False
    cached = SSAEnvironment(cached_config)
    dynamic = SSAEnvironment(dynamic_config)

    cached_observation = cached.reset(seed=9)
    dynamic_observation = dynamic.reset(seed=9)
    assert json.dumps(cached_observation).encode() == json.dumps(dynamic_observation).encode()

    actions = [
        {"sat_0": {"mode": "isl_share"}, "sat_1": {"mode": "charging"}},
        {"sat_1": {"mode": "isl_share"}, "sat_2": {"mode": "safe"}},
        {"sat_0": {"mode": "safe"}, "sat_2": {"mode": "isl_share"}},
    ]
    for action in actions:
        cached_result = cached.step(action)
        dynamic_result = dynamic.step(action)
        assert cached_result.reward == dynamic_result.reward
        assert cached_result.done == dynamic_result.done
        assert json.dumps(cached_result.info).encode() == json.dumps(dynamic_result.info).encode()
        assert json.dumps(cached_result.observation).encode() == json.dumps(
            dynamic_result.observation
        ).encode()
    assert cached.episode_metrics() == dynamic.episode_metrics()


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
