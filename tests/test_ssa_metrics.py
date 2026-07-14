"""SSA discovery, custody, and time-averaged metric tests."""

from __future__ import annotations

import pytest

from autops.missions.ssa.metrics import EpisodeStats, custody_ceiling, discovery_ceiling


def test_discovery_ceiling_requires_visibility_before_the_final_pass() -> None:
    passes = [
        {"satellite_id": "sat_0", "start_step": 2, "end_step": 2},
        {"satellite_id": "sat_0", "start_step": 6, "end_step": 7},
    ]
    visibility = [
        {"step": 1, "visible_target_ids": ["a"]},
        {"step": 6, "visible_target_ids": ["b"]},
        {"step": 7, "visible_target_ids": ["c"]},
    ]
    assert discovery_ceiling(passes, visibility, 3) == pytest.approx(2.0 / 3.0)
    assert discovery_ceiling([], visibility, 3) == 0.0
    assert discovery_ceiling(passes, visibility, 0) == 0.0


def test_custody_ceiling_is_delivered_to_ground_and_time_averaged() -> None:
    passes = [{"satellite_id": "sat_0", "start_step": 2, "end_step": 2}]
    visibility = [{"step": 1, "visible_target_ids": ["a"]}]
    # At step two, the latest visible record reaches ground. It is fresh for
    # step 2 only when tau is one, so the six-step utility is 1/6.
    assert custody_ceiling(passes, visibility, 1, 1, 6) == pytest.approx(1.0 / 6.0)
    assert custody_ceiling(passes, visibility, 0, 1, 6) == 0.0


def test_episode_stats_separates_coverage_from_custody() -> None:
    stats = EpisodeStats()
    stats.record_detection("a", 0, duplicate=False, cued=False)
    stats.record_first_delivery("a", 2, relay_hops=1)
    stats.update_step({"a"}, {"a"}, 2)
    stats.update_step({"a"}, set(), 2)
    snapshot = stats.snapshot(
        current_step=2,
        target_count=2,
        delivered={"a"},
        custody=set(),
        freshest_ground_steps={"a": 0},
        known_count=1,
        mean_knowledge_latency_steps=2.0,
    )
    assert snapshot["ssa_delivered_coverage"] == pytest.approx(0.5)
    assert snapshot["ssa_delivered_coverage_auc"] == pytest.approx(0.5)
    assert snapshot["ssa_custody_utility"] == pytest.approx(0.25)
    assert snapshot["ssa_custody_fraction_final"] == 0.0
    assert snapshot["custody_losses"] == 1.0
    assert snapshot["relayed_delivery_fraction"] == 1.0
