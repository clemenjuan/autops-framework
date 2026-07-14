"""Determinism and physical-access tests for the SSA target layer."""

from __future__ import annotations

import math
import random

import pytest

from autops.missions.ssa.targets import (
    Target,
    apparent_magnitude,
    detection_draw,
    detection_probability,
    generate_catalog,
    optical_accesses,
    phase_function,
)


def test_photometry_and_detection_probability_have_physical_anchors() -> None:
    assert phase_function(0.0) == pytest.approx(2.0 / (3.0 * math.pi))
    assert phase_function(math.pi) == 0.0
    assert apparent_magnitude(0.1, 10.0, 0.0) < apparent_magnitude(0.01, 10.0, 0.0)
    assert apparent_magnitude(0.1, 10.0, 0.0) < apparent_magnitude(0.1, 100.0, 0.0)
    assert detection_probability(15.0, 15.0, 0.5) == pytest.approx(0.5)
    assert detection_probability(14.0, 15.0, 0.5) > 0.5
    with pytest.raises(ValueError, match="sigma"):
        detection_probability(15.0, sigma=0.0)


def test_detection_draw_is_pure_and_sensitive_to_the_paired_tuple() -> None:
    baseline = detection_draw(7, "rso_0", "sat_0", 12)
    assert baseline == detection_draw(7, "rso_0", "sat_0", 12)
    alternatives = {
        detection_draw(8, "rso_0", "sat_0", 12),
        detection_draw(7, "rso_1", "sat_0", 12),
        detection_draw(7, "rso_0", "sat_1", 12),
        detection_draw(7, "rso_0", "sat_0", 13),
    }
    assert baseline not in alternatives
    assert all(0.0 <= draw < 1.0 for draw in alternatives | {baseline})


def test_fragmentation_catalog_preserves_the_settled_rng_draw_order() -> None:
    seed = 19
    catalog = generate_catalog(1, seed, raan_center_deg=120.0)
    target = catalog[0]

    rng = random.Random(seed)
    rng.gauss(0.0, 13.0)
    rng.gauss(0.0, 26.0)
    expected_raan = (120.0 + rng.uniform(-0.3, 0.3)) % 360.0
    expected_eccentricity = rng.uniform(0.0, 0.001)
    expected_argument = rng.uniform(0.0, 360.0)
    expected_anomaly = rng.uniform(0.0, 360.0)
    size_u = rng.random()
    size_span = 1.0 - (0.10 / 0.01) ** -2.5
    expected_size = 0.01 * (1.0 - size_u * size_span) ** (-1.0 / 2.5)

    assert target.raan_deg == pytest.approx(expected_raan)
    assert target.eccentricity == pytest.approx(expected_eccentricity)
    assert target.arg_perigee_deg == pytest.approx(expected_argument)
    assert target.true_anomaly_deg == pytest.approx(expected_anomaly)
    assert target.size_m == pytest.approx(expected_size)
    assert catalog == generate_catalog(1, seed, raan_center_deg=120.0)
    assert 0.01 <= target.size_m <= 0.10


def test_optical_access_enforces_boresight_range_and_sunlight() -> None:
    target = Target("rso_0", 7_100.0, 0.0, 0.0, 0.0, 0.0, 0.0, size_m=0.1)
    kwargs = {
        "observer_position_km": (7_000.0, 0.0, 0.0),
        "observer_velocity_hat": (0.0, 1.0, 0.0),
        "targets": [target],
        "target_positions_km": {"rso_0": (7_000.0, 10.0, 0.0)},
        "sun_hat": (1.0, 0.0, 0.0),
        "fov_half_angle_deg": 1.0,
        "boresight_pitch_deg": 0.0,
        "range_cap_km": 20.0,
        "magnitude_limit": 100.0,
    }
    access = optical_accesses(**kwargs)
    assert len(access) == 1
    assert access[0].range_km == pytest.approx(10.0)
    assert access[0].angle_deg == pytest.approx(0.0)

    assert not optical_accesses(**{**kwargs, "range_cap_km": 5.0})
    assert not optical_accesses(
        **{**kwargs, "target_positions_km": {"rso_0": (7_000.0, -10.0, 0.0)}}
    )
    assert not optical_accesses(**{**kwargs, "sun_hat": (-1.0, 0.0, 0.0)})
