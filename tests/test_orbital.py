"""Fast tests for orbital events and pure RF helpers."""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime
from pathlib import Path

import pytest

from autops.orbital import (
    EclipseInterval,
    GroundLinkConfig,
    GroundPass,
    GroundStation,
    ISLConfig,
    OrbitalContext,
    OrbitElements,
    SimplifiedModel,
    apply_launch_lottery,
    build_orbital_context,
    ground_link_budget,
    isl_link_budget,
    orekit,
)


@pytest.fixture
def orbit() -> OrbitElements:
    return OrbitElements(
        altitude_km=400.0,
        eccentricity=0.001,
        inclination_deg=97.4,
        raan_deg=0.0,
        arg_perigee_deg=0.0,
        true_anomaly_deg=0.0,
        epoch=datetime(2026, 6, 1, tzinfo=UTC),
        propagator="j2",
    )


@pytest.fixture
def fallback() -> SimplifiedModel:
    return SimplifiedModel(
        orbital_period_s=5_554.0,
        eclipse_fraction=0.36,
        passes_min_per_day=2,
        passes_max_per_day=3,
        pass_min_duration_s=22.0,
        pass_max_duration_s=422.0,
    )


@pytest.fixture
def station() -> GroundStation:
    return GroundStation(
        latitude_deg=48.0483,
        longitude_deg=11.6567,
        altitude_m=0.0,
        min_elevation_deg=10.0,
    )


def test_launch_lottery_is_seeded_and_does_not_mutate_input(orbit: OrbitElements) -> None:
    first = apply_launch_lottery(orbit, 42)
    second = apply_launch_lottery(orbit, 42)
    assert first == second
    assert orbit.raan_deg == orbit.arg_perigee_deg == orbit.true_anomaly_deg == 0.0
    assert (first.raan_deg, first.arg_perigee_deg, first.true_anomaly_deg) == pytest.approx(
        (230.19364744483815, 9.003871880160098, 99.01055461288293)
    )


def test_seeded_fallback_is_reproducible_and_local(
    orbit: OrbitElements,
    fallback: SimplifiedModel,
    station: GroundStation,
) -> None:
    random.seed(999)
    before = random.getstate()
    kwargs = dict(
        orbit=orbit,
        fallback=fallback,
        ground_station=station,
        downlink_rate_kbps=50.0,
        step_s=60.0,
        total_steps=1_440,
        seed=7,
        prefer_orekit=False,
    )
    first = build_orbital_context(**kwargs)
    second = build_orbital_context(**kwargs)
    assert first == second
    assert random.getstate() == before
    assert first.backend == "simplified"
    assert 2 <= len(first.ground_passes) <= 3


def test_orekit_failure_uses_honestly_labelled_seeded_fallback(
    orbit: OrbitElements,
    fallback: SimplifiedModel,
    station: GroundStation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_orbit: OrbitElements) -> None:
        raise RuntimeError("Eckstein-Hechler construction failed")

    monkeypatch.setattr(orekit, "create_propagator", fail)
    context = build_orbital_context(
        orbit,
        fallback,
        station,
        downlink_rate_kbps=50.0,
        step_s=60.0,
        total_steps=10,
        seed=7,
    )
    assert context.backend == "simplified"
    assert context.propagator_kind == "phase-and-seeded-passes"

    with pytest.raises(RuntimeError, match="required Orekit propagation failed"):
        build_orbital_context(
            orbit,
            fallback,
            station,
            downlink_rate_kbps=50.0,
            step_s=60.0,
            total_steps=10,
            seed=7,
            require_orekit=True,
        )


def test_fallback_uses_exact_seconds_and_supplied_rate(
    orbit: OrbitElements,
    station: GroundStation,
) -> None:
    model = SimplifiedModel(6_000.0, 0.3, 1, 1, 22.0, 22.0)
    context = build_orbital_context(
        orbit,
        model,
        station,
        downlink_rate_kbps=50.0,
        step_s=60.0,
        total_steps=1_440,
        seed=3,
        prefer_orekit=False,
    )
    ground_pass = context.ground_passes[0]
    assert ground_pass.duration_s == pytest.approx(22.0)
    assert ground_pass.data_budget_mb == pytest.approx(0.1375)
    assert context.eclipses[0] == EclipseInterval(0.0, 1_800.0)


def test_context_credits_substep_contact_and_explicit_future_passes() -> None:
    current = GroundPass(610.0, 632.0, 20.0, 0.1375)
    future = GroundPass(700.0, 760.0, 30.0, 0.375)
    second = GroundPass(1_000.0, 1_120.0, 40.0, 0.75)
    context = OrbitalContext(
        eclipses=(),
        ground_passes=(current, future, second),
        backend="simplified",
        propagator_kind="test",
        step_s=60.0,
        duration_s=1_200.0,
    )
    assert context.contact_seconds(10) == pytest.approx(22.0)
    assert context.contact_seconds(9) == 0.0
    assert context.next_pass_contact_s(10) == pytest.approx(22.0)
    assert context.future_pass_contact_s(10, 1) == pytest.approx(60.0)
    assert context.future_pass_contact_s(10, 2) == pytest.approx(120.0)
    assert context.remaining_contact_s(10) == pytest.approx(202.0)


def test_half_open_eclipse_state_and_overlap() -> None:
    context = OrbitalContext(
        eclipses=(EclipseInterval(0.0, 90.0),),
        ground_passes=(),
        backend="simplified",
        propagator_kind="test",
        step_s=60.0,
        duration_s=180.0,
    )
    assert not context.is_in_sunlight(0)
    assert not context.is_in_sunlight(1)
    assert context.eclipse_seconds(1) == pytest.approx(30.0)
    assert context.is_in_sunlight(2)


def test_ground_link_budget_is_pure_and_geometry_sensitive() -> None:
    config = GroundLinkConfig.from_mapping(
        {
            "downlink": {
                "frequency_mhz": 2_245.0,
                "sat_tx_power_dbm": 30.0,
                "sat_antenna_gain_dbi": 5.0,
                "sat_cable_loss_db": 2.0,
                "gs_antenna_gain_dbi": 20.0,
                "gs_cable_loss_db": 2.0,
                "gs_sensitivity_dbm": -100.0,
            },
            "uplink": {
                "frequency_mhz": 2_067.5,
                "gs_tx_power_dbm": 30.0,
                "gs_pa_gain_db": 0.0,
                "gs_antenna_gain_dbi": 5.0,
                "gs_cable_loss_db": 2.0,
                "sat_antenna_gain_dbi": 20.0,
                "sat_cable_loss_db": 2.0,
                "sat_sensitivity_dbm": -100.0,
            },
            "losses": {"atmosphere_db": 2.0, "pointing_error_db": 2.0},
        }
    )
    horizon = ground_link_budget(config, altitude_km=400.0, elevation_deg=10.0)
    overhead = ground_link_budget(config, altitude_km=400.0, elevation_deg=90.0)
    assert horizon.slant_range_km > overhead.slant_range_km
    assert horizon.downlink.margin_db < overhead.downlink.margin_db
    assert horizon.downlink.received_power_dbm - (-100.0) == pytest.approx(
        horizon.downlink.margin_db
    )


def _isl_config() -> ISLConfig:
    return ISLConfig.from_mapping(
        {
            "tx_power_w": 2.0,
            "rx_gain_db": 1.0,
            "rx_loss_db": 0.5,
            "tx_gain_db": 1.0,
            "tx_loss_db": 3.0,
            "frequency_hz": 437e6,
            "bandwidth_hz": 9_600.0,
            "symbol_rate_hz": 9_600.0,
            "modulation_order": 4,
            "sensitivity_dbw": -151.0,
            "noise_temperature_k": 290.0,
            "operational_field_is_ignored": True,
        }
    )


def test_isl_budget_degrades_with_range() -> None:
    near = isl_link_budget(100_000.0, _isl_config())
    far = isl_link_budget(1_000_000.0, _isl_config())
    assert near["margin_db"] > far["margin_db"]
    assert near["effective_data_rate_bps"] >= far["effective_data_rate_bps"]
    assert 0.0 <= far["bit_error_rate"] <= 1.0


def test_orekit_data_is_repository_relative() -> None:
    assert orekit.orekit_data_path() == Path(__file__).resolve().parents[1] / "orekit-data.zip"


@pytest.mark.orekit
def test_actual_orekit_eckstein_hechler_backend(orbit: OrbitElements) -> None:
    if not orekit.is_available():
        pytest.skip("Orekit, Java 17, or orekit-data.zip is unavailable")
    propagator = orekit.create_propagator(orbit)
    initial = orekit.position_km(propagator, 0.0)
    later = orekit.position_km(propagator, 60.0)
    assert propagator.kind == "eckstein-hechler-j2"
    assert 6_700.0 < math.dist((0.0, 0.0, 0.0), initial) < 6_900.0
    assert math.dist(initial, later) > 100.0
