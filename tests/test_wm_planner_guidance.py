from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from autops.core.types import DecisionContext
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.artifact import (
    ModelContract,
    NormalizationContract,
    PlannerArtifact,
    ProbeContract,
    ProbeEvidenceContract,
)
from autops.wm.cem import CEMConfig
from autops.wm.guidance import pipeline_scores, project_executable_candidates
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS
from autops.wm.scoring import analytical_candidate_attributes


def _evidence(attributes: tuple[str, ...]) -> ProbeEvidenceContract:
    zeros = {name: 0.0 for name in attributes}
    return ProbeEvidenceContract(attributes, zeros, zeros, zeros, (0,), (1,), 1e-3, 1)


def _artifact() -> PlannerArtifact:
    attributes = ("science_progress", "downlink_progress")
    return PlannerArtifact(
        model=ModelContract(
            checkpoint="weights/lewm.ckpt",
            mission="eventsat",
            obs_dim=25,
            action_dim=7,
            embed_dim=4,
            history=3,
            observation_names=EVENTSAT_OBSERVATIONS,
            action_names=EVENTSAT_ACTIONS,
            trace_sha256="0" * 64,
            checkpoint_sha256="0" * 64,
        ),
        normalization=NormalizationContract(
            obs_mean=(0.0,) * 25,
            obs_std=(1.0,) * 25,
            action_mean=(0.0,) * 7,
            action_std=(1.0,) * 7,
        ),
        probe=ProbeContract(
            W=((0.0,) * 4, (0.0,) * 4),
            b=(0.0, 0.0),
            attribute_names=attributes,
            target_mean=(0.0, 0.0),
            target_std=(1.0, 1.0),
        ),
        probe_evidence=_evidence(attributes),
        cem=CEMConfig(
            horizon=4,
            samples=16,
            elites=4,
            iterations=1,
            plan_hold=1,
            seed=17,
        ),
        mode_weight_presets={"science": {"science_progress": 0.0, "downlink_progress": 0.25}},
    )


def _planner(**config: Any) -> EventSatLeWMCEM:
    return EventSatLeWMCEM(
        {
            "artifact": _artifact(),
            "rollout_scorer": lambda history, sequences: np.zeros(sequences.shape[0]),
            **config,
        }
    )


def _state(**updates: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "step_duration_s": 60.0,
        "downlink_rate_kbps": 8000.0,
        "obc_data_mb": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "uncompressed_observations": 0,
        "compression_progress": 0,
        "compression_time_factor": 2.0,
        "observation_size_mb": 9.41,
        "compression_ratio": 5.11,
        "jetson_to_obc_rate_kbps": 8000.0,
        "storage_capacity_mb": 4096.0,
        "settling_time_steps": 0,
        "current_mode": "charging",
        "orbital_period_steps": 94.0,
        "time_to_next_pass": 94.0,
    }
    state.update(updates)
    return state


def test_contact_guidance_biases_staging_pointing_and_contact_offsets() -> None:
    planner = _planner(contact_guidance_strength=0.75)
    probabilities = np.full((4, 7), 1.0 / 7.0)
    state = _state(
        timestep=10,
        settling_time_steps=2,
        _analytic_orbit_cache={13: {"ground_pass_active": True}},
    )

    guided = planner._contact_guided_probabilities(state, probabilities)
    send = EVENTSAT_ACTIONS.index("payload_send")
    communication = EVENTSAT_ACTIONS.index("communication")

    assert guided[0, send] == pytest.approx(0.75 + 0.25 / 7.0)
    assert np.allclose(guided[1:, communication], 0.75 + 0.25 / 7.0)
    assert np.allclose(guided.sum(axis=1), 1.0)
    assert np.array_equal(
        _planner(contact_guidance=False)._contact_guided_probabilities(state, probabilities),
        probabilities,
    )


def test_lightweight_score_credits_only_pipeline_feasible_downlink() -> None:
    planner = _planner()
    send = EVENTSAT_ACTIONS.index("payload_send")
    communication = EVENTSAT_ACTIONS.index("communication")
    state = _state(
        jetson_compressed_mb=5.0,
        timestep=20,
        _analytic_orbit_cache={21: {"contact_window_seconds": 60.0}},
    )
    sequences = np.asarray([[send, communication], [communication, communication]])

    scores = planner._lightweight_pipeline_scores(state, sequences)

    assert scores[0] > 0.0
    assert scores[1] == pytest.approx(0.0)


def test_lightweight_flag_and_undeliverable_penalty_are_independent() -> None:
    charging = EVENTSAT_ACTIONS.index("charging")
    sequences = np.asarray([[charging, charging]])
    state = _state(
        jetson_raw_mb=9.41,
        remaining_achievable_downlink_mb=0.0,
    )
    history = {"state": state}

    enabled = _planner(undeliverable_capacity_penalty=1.0, pass_stage_reward=0.0)
    disabled = _planner(
        lightweight_shaping=False,
        undeliverable_capacity_penalty=1.0,
        pass_stage_reward=0.0,
    )

    assert enabled._score_candidates(history, sequences)[0] < 0.0
    assert disabled._score_candidates(history, sequences)[0] == pytest.approx(0.0)


def test_shaping_configuration_is_bounded_and_rejects_exact_surrogate() -> None:
    planner = _planner(
        contact_guidance_strength=2.0,
        undeliverable_capacity_penalty=-1.0,
    )
    assert planner._contact_guidance_strength == 1.0
    assert planner._undeliverable_capacity_penalty == 0.0

    with pytest.raises(ValueError, match="intentionally unsupported"):
        _planner(exact_analytic_shaping=True)


def test_cold_start_prior_favors_charging_and_suppresses_safe() -> None:
    planner = _planner()
    probabilities = planner._proposal_probabilities()
    expected = np.full((4, 7), 1.0 / 7.0)
    expected[:, EVENTSAT_ACTIONS.index("charging")] += 0.08
    expected[:, EVENTSAT_ACTIONS.index("safe")] *= 0.20
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(probabilities, expected)


def test_seeded_candidate_completes_observe_compress_send_contact_pipeline() -> None:
    planner = _planner()
    charging = EVENTSAT_ACTIONS.index("charging")
    sequences = np.full((4, 4), charging, dtype=np.int64)
    state = _state(
        compression_time_factor=1.0,
        remaining_achievable_downlink_mb=10.0,
        timestep=0,
        time_to_next_pass=3.0,
        _analytic_orbit_cache={3: {"contact_window_seconds": 60.0}},
    )

    seeded = planner._seed_pipeline_candidate(state, sequences, np.ones(7, dtype=bool))

    np.testing.assert_array_equal(
        seeded[0],
        [
            EVENTSAT_ACTIONS.index("payload_observe"),
            EVENTSAT_ACTIONS.index("payload_compress"),
            EVENTSAT_ACTIONS.index("payload_send"),
            EVENTSAT_ACTIONS.index("communication"),
        ],
    )
    np.testing.assert_array_equal(seeded[1:], sequences[1:])


def test_planner_injects_pipeline_candidate_at_every_cem_iteration() -> None:
    scored_first_candidates: list[np.ndarray] = []

    def scorer(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        del history
        scored_first_candidates.append(sequences[0].copy())
        return np.zeros(sequences.shape[0])

    planner = _planner(iterations=3, rollout_scorer=scorer)
    state = _state(
        obs25=np.zeros(25, dtype=np.float32),
        battery_soc=0.8,
        health_status="nominal",
        physical_ground_pass_active=False,
        contact_window_active=False,
        data_stored_mb=0.0,
        undetected_observations=0,
        compression_time_factor=1.0,
        remaining_achievable_downlink_mb=10.0,
        timestep=0,
        time_to_next_pass=3.0,
        _analytic_orbit_cache={3: {"contact_window_seconds": 60.0}},
    )
    context = DecisionContext(state=state, observation={}, memory=None, step=0, role="onboard")

    planner.select_action(context)

    expected = np.asarray(
        [
            EVENTSAT_ACTIONS.index("payload_observe"),
            EVENTSAT_ACTIONS.index("payload_compress"),
            EVENTSAT_ACTIONS.index("payload_send"),
            EVENTSAT_ACTIONS.index("communication"),
        ]
    )
    assert len(scored_first_candidates) == 3
    assert all(np.array_equal(candidate, expected) for candidate in scored_first_candidates)


def test_projection_propagates_complete_pipeline_through_future_actions() -> None:
    actions = {name: EVENTSAT_ACTIONS.index(name) for name in EVENTSAT_ACTIONS}
    state = _state(
        jetson_raw_mb=9.41,
        uncompressed_observations=1,
        detection_time_steps=5,
        planning_contact_seconds=[0.0] * 8 + [60.0],
    )
    requested = np.asarray(
        [
            [
                actions["payload_compress"],
                actions["payload_compress"],
                *([actions["payload_detect"]] * 5),
                actions["payload_send"],
                actions["communication"],
            ]
        ]
    )

    projection = project_executable_candidates(
        state, requested, reserve_soc=0.5, comms_soc_floor=0.25
    )
    terminal = projection.terminal_states[0]

    np.testing.assert_array_equal(projection.sequences, requested)
    assert projection.repair_counts[0] == 0
    assert terminal["total_detections"] == 1
    assert terminal["jetson_raw_mb"] == pytest.approx(0.0)
    assert terminal["jetson_compressed_mb"] == pytest.approx(0.0)
    assert terminal["obc_data_mb"] == pytest.approx(0.0)
    assert terminal["data_downlinked_mb"] > 1.8
    attributes = analytical_candidate_attributes(
        projection,
        ("science_progress", "detection_progress", "downlink_progress"),
    )
    assert attributes[0, 0] == 0.0
    assert attributes[0, 1] == 1.0
    assert attributes[0, 2] > 1.8


def test_projection_repairs_invalid_future_actions_and_propagates_battery() -> None:
    observe = EVENTSAT_ACTIONS.index("payload_observe")
    send = EVENTSAT_ACTIONS.index("payload_send")
    charging = EVENTSAT_ACTIONS.index("charging")
    state = _state(
        battery_soc=0.51,
        planning_sunlight=[False, False],
        planning_power={
            "consumption": {
                name: {"sun_w": 100.0, "eclipse_w": 100.0} for name in EVENTSAT_ACTIONS
            },
            "generation_peak_w": 0.0,
            "panel_efficiency_factor": 0.0,
            "battery_capacity_wh": 70.0,
            "charge_efficiency": 0.9,
        },
    )
    requested = np.asarray([[observe, observe], [send, send]])

    projection = project_executable_candidates(
        state, requested, reserve_soc=0.5, comms_soc_floor=0.25
    )

    np.testing.assert_array_equal(projection.sequences[0], [observe, charging])
    np.testing.assert_array_equal(projection.sequences[1], [charging, charging])
    np.testing.assert_array_equal(projection.repair_counts, [1, 2])


def test_pipeline_score_rejects_a_projection_from_another_candidate_bank() -> None:
    charging = EVENTSAT_ACTIONS.index("charging")
    observe = EVENTSAT_ACTIONS.index("payload_observe")
    state = _state()
    projection = project_executable_candidates(
        state,
        np.asarray([[charging]], dtype=np.int64),
        reserve_soc=0.5,
        comms_soc_floor=0.25,
    )

    with pytest.raises(ValueError, match="must match"):
        pipeline_scores(
            state,
            np.asarray([[observe]], dtype=np.int64),
            downlink_weight=1.0,
            downlink_reward=1.0,
            pass_stage_reward=0.0,
            reference_weight=1.0,
            undeliverable_penalty=0.0,
            projection=projection,
        )
