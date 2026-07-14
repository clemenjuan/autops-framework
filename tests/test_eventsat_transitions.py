"""Atomic-boundary tests for the authoritative EventSat pipeline."""

from __future__ import annotations

from dataclasses import replace

import pytest

from autops.missions.eventsat.transitions import (
    EPSILON_MB,
    PipelineParameters,
    apply_can_transfer,
    apply_compress,
    apply_detect,
    apply_downlink,
    apply_observe,
    jetson_occupancy_mb,
)

PARAMETERS = PipelineParameters(
    observation_size_mb=9.41,
    compression_ratio=5.11,
    jetson_capacity_mb=249_036.8,
    obc_capacity_mb=4_096.0,
    detection_metadata_mb=0.01,
    jetson_to_obc_rate_kbps=8_000.0,
    downlink_rate_kbps=50.0,
    step_duration_s=60.0,
)


def test_observe_accepts_exact_fit_and_rejects_overflow_atomically() -> None:
    parameters = replace(PARAMETERS, jetson_capacity_mb=PARAMETERS.observation_size_mb)
    exact = apply_observe({}, parameters)
    assert exact.accepted
    assert exact.state["jetson_raw_mb"] == pytest.approx(9.41)
    assert exact.state["uncompressed_observations"] == 1

    original = {
        "jetson_raw_mb": 2.0 * EPSILON_MB,
        "uncompressed_observations": 0,
        "total_raw_captured_mb": 0.0,
    }
    rejected = apply_observe(original, parameters)
    assert not rejected.accepted
    assert rejected.reason == "jetson_capacity"
    assert rejected.state == original


def test_compression_conserves_one_atomic_product() -> None:
    state = {
        "jetson_raw_mb": 9.41,
        "jetson_compressed_mb": 2.0,
        "uncompressed_observations": 1,
        "undetected_observations": 0,
    }
    transition = apply_compress(state, PARAMETERS)
    assert transition.accepted
    assert transition.state["jetson_raw_mb"] == pytest.approx(0.0)
    assert transition.state["jetson_compressed_mb"] == pytest.approx(
        2.0 + PARAMETERS.compressed_observation_mb
    )
    assert transition.state["uncompressed_observations"] == 0
    assert transition.state["undetected_observations"] == 1
    assert jetson_occupancy_mb(transition.state) < jetson_occupancy_mb(state)


def test_detection_obc_boundary_is_atomic() -> None:
    exact_state = {
        "obc_data_mb": PARAMETERS.obc_capacity_mb - PARAMETERS.detection_metadata_mb,
        "undetected_observations": 1,
        "total_detections": 0,
    }
    exact = apply_detect(exact_state, PARAMETERS)
    assert exact.accepted
    assert exact.state["obc_data_mb"] == pytest.approx(PARAMETERS.obc_capacity_mb)
    assert exact.state["undetected_observations"] == 0
    assert exact.state["total_detections"] == 1

    overflow_state = {
        **exact_state,
        "obc_data_mb": exact_state["obc_data_mb"] + 1e-8,
    }
    rejected = apply_detect(overflow_state, PARAMETERS)
    assert not rejected.accepted
    assert rejected.reason == "obc_capacity"
    assert rejected.state == overflow_state


def test_can_transfer_respects_headroom_and_raw_equivalent() -> None:
    state = {
        "jetson_compressed_mb": 10.0,
        "obc_data_mb": PARAMETERS.obc_capacity_mb - 0.5,
        "obc_raw_equivalent_mb": 3.0,
    }
    transition = apply_can_transfer(state, PARAMETERS)
    assert transition.accepted
    assert transition.transferred_mb == pytest.approx(0.5)
    assert transition.raw_equivalent_mb == pytest.approx(0.5 * PARAMETERS.compression_ratio)
    assert transition.state["obc_data_mb"] == pytest.approx(PARAMETERS.obc_capacity_mb)
    assert transition.state["jetson_compressed_mb"] == pytest.approx(9.5)


def test_downlink_uses_actual_contact_seconds_and_conserves_backlog() -> None:
    state = {
        "obc_data_mb": 1.0,
        "obc_raw_equivalent_mb": 5.11,
        "data_downlinked_mb": 0.0,
        "downlink_raw_equivalent_mb": 0.0,
    }
    transition = apply_downlink(state, PARAMETERS, contact_seconds=22.0)
    assert transition.accepted
    assert transition.transferred_mb == pytest.approx(0.1375)
    assert transition.state["obc_data_mb"] + transition.state[
        "data_downlinked_mb"
    ] == pytest.approx(1.0)
    assert (
        transition.state["obc_raw_equivalent_mb"] + transition.state["downlink_raw_equivalent_mb"]
    ) == pytest.approx(5.11)

    no_contact = apply_downlink(state, PARAMETERS, contact_seconds=0.0)
    assert not no_contact.accepted
    assert no_contact.reason == "no_contact"
    assert no_contact.state == state
