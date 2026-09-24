"""Semantic acceptance tests for the EventSat onboard information boundary."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from test_wm_planner import _artifact

from autops.config import expand_coordinate
from autops.core.types import DecisionContext
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.observation import (
    FORECAST_KEYS,
    encode_vectors,
    observation_space,
)
from autops.orbital import orekit
from autops.representations.analytical_planner import EventSatAnalyticalCEM
from autops.representations.symb import EventSatSymbolic
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.dataset import split_episodes
from autops.wm.guidance import admissible_action_mask
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS, EVENTSAT_STATES


def _environment(max_steps: int = 6, *, prefer_orekit: bool = False) -> EventSatEnvironment:
    config = deepcopy(expand_coordinate("eventsat/sas/ao/symb").mission_config)
    config["anomalies"]["probability_per_step"] = 0.0
    return EventSatEnvironment(config, max_steps=max_steps, prefer_orekit=prefer_orekit)


def _metadata(observation: dict[str, Any]) -> dict[str, Any]:
    return observation["satellites"]["eventsat_0"]["metadata"]


def _perturb_future(observation: dict[str, Any]) -> dict[str, Any]:
    changed = deepcopy(observation)
    metadata = _metadata(changed)
    metadata.update(
        time_to_next_pass=1.0,
        time_to_next_eclipse=3.0,
        remaining_pass_duration=9.0,
        future_pass_capacity_mb=99.0,
        remaining_achievable_downlink_mb=999.0,
        planning_contact_seconds=[60.0] * len(metadata["planning_contact_seconds"]),
        planning_sunlight=[False] * len(metadata["planning_sunlight"]),
    )
    return changed


def _vector_value(observation: dict[str, Any], name: str) -> float:
    return float(encode_vectors(observation)[0][EVENTSAT_OBSERVATIONS.index(name)])


def test_hidden_future_tables_cannot_change_the_encoded_observation() -> None:
    observation = _environment().reset(7)
    original, labels, _ = encode_vectors(observation)
    changed, changed_labels, _ = encode_vectors(_perturb_future(observation))
    np.testing.assert_array_equal(original, changed)
    assert not np.array_equal(labels, changed_labels)


def test_hidden_future_tables_cannot_change_the_learned_decision() -> None:
    observation = _environment().reset(7)

    def score(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        assert not set(FORECAST_KEYS) & set(history["state"])
        return (sequences == 1).sum(axis=1).astype(np.float32)

    decisions = []
    for record in (observation, _perturb_future(observation)):
        planner = EventSatLeWMCEM({"artifact": _artifact(), "rollout_scorer": score})
        planner.reset(3)
        state = planner.encode_observation(record)
        context = DecisionContext(state, record, None, 0, "onboard")
        action = planner.select_action(context)["eventsat_0"]
        decisions.append((action["mode"], planner.last_plan))
    assert decisions[0] == decisions[1]


def test_only_the_analytical_oracle_receives_the_almanac() -> None:
    observation = _environment().reset(7)
    oracle = EventSatAnalyticalCEM({"artifact": _artifact()}).encode_observation(observation)
    learned = EventSatLeWMCEM({"artifact": _artifact()}).encode_observation(observation)
    assert "planning_contact_seconds" in oracle
    assert not set(FORECAST_KEYS) & set(learned)


def test_onboard_view_permits_prepointing_while_truth_gates_transfer() -> None:
    onboard = {"battery_soc": 0.8, "obc_data_mb": 5.0, "settling_time_steps": 2}
    oracle = {**onboard, "time_to_next_pass": 40.0}
    communication = 1
    assert admissible_action_mask(onboard, reserve_soc=0.5, comms_soc_floor=0.25)[communication]
    assert not admissible_action_mask(oracle, reserve_soc=0.5, comms_soc_floor=0.25)[communication]

    env = _environment(200)
    env.reset(7)
    while env.physical_contact_active():
        env.step({"eventsat_0": {"mode": "charging"}})
    env.state.obc_data_mb = 5.0
    step = env.step({"eventsat_0": {"mode": "communication"}})
    assert step.info["step_downlinked_mb"] == 0.0


def test_interval_feedback_is_causal_and_encoded_as_increments() -> None:
    env = _environment()
    initial = env.reset(7)
    assert _vector_value(initial, "last_captured") == 0.0
    assert _vector_value(initial, "last_action_accepted") == 1.0
    env.step({"eventsat_0": {"mode": "payload_observe"}})
    env.step({"eventsat_0": {"mode": "payload_observe"}})
    captured = env.step({"eventsat_0": {"mode": "payload_observe"}}).observation
    assert _metadata(captured)["last_interval"]["captured_mb"] == pytest.approx(9.41)
    assert _vector_value(captured, "last_captured") == pytest.approx(1.0)
    assert _vector_value(captured, "attitude_target_payload_observe") == 1.0
    compressing = env.step({"eventsat_0": {"mode": "payload_compress"}}).observation
    assert _vector_value(compressing, "last_captured") == 0.0


def test_in_progress_jobs_are_accepted_and_only_missing_products_fail() -> None:
    env = _environment(max_steps=40)
    env.reset(7)
    for _ in range(env.settling_steps + 1):
        env.step({"eventsat_0": {"mode": "payload_observe"}})
    compress = [
        env.step({"eventsat_0": {"mode": "payload_compress"}}).info
        for _ in range(env.settling_steps + env.compression_steps)
    ]
    assert [info["action_accepted"] for info in compress if not info["in_transition"]] == [
        True
    ] * env.compression_steps
    assert env.state.undetected_observations == 1
    detect = [
        env.step({"eventsat_0": {"mode": "payload_detect"}}).info["action_accepted"]
        for _ in range(env.detection_steps)
    ]
    assert all(detect)
    assert env.state.total_detections == 1
    assert not env.step({"eventsat_0": {"mode": "payload_compress"}}).info["action_accepted"]


def test_each_seed_starts_at_its_own_paired_epoch() -> None:
    starts = {}
    for seed in (5, 5, 6):
        env = _environment()
        env.reset(seed)
        starts.setdefault(seed, set()).add(env.episode_provenance()["orbit"]["epoch"])
    assert len(starts[5]) == 1
    assert starts[5] != starts[6]


def test_planning_charge_reaches_the_battery_but_not_the_encoded_platform_energy() -> None:
    observations = {}
    for planned in (False, True):
        env = _environment()
        env.reset(7)
        observations[planned] = env.step(
            {"eventsat_0": {"mode": "charging", "jetson_planned": planned}}
        ).observation
    assert _metadata(observations[True])["last_interval"]["planner_energy_wh"] > 0.0
    assert _vector_value(observations[True], "battery_soc") < _vector_value(
        observations[False], "battery_soc"
    )
    assert _vector_value(observations[True], "last_platform_energy_norm") == pytest.approx(
        _vector_value(observations[False], "last_platform_energy_norm")
    )


def test_storage_fills_resolve_one_product_and_reach_one_only_at_capacity() -> None:
    observation = _environment().reset(7)
    metadata = _metadata(observation)
    compressed_mb = metadata["observation_size_mb"] / metadata["compression_ratio"]
    metadata.update(
        jetson_raw_mb=metadata["observation_size_mb"],
        jetson_compressed_mb=0.0,
        uncompressed_observations=1,
    )
    one_raw = _vector_value(observation, "jetson_fill_log")
    assert one_raw > 0.05
    assert one_raw == pytest.approx(_vector_value(observation, "uncompressed_observations_log"))
    metadata.update(jetson_compressed_mb=compressed_mb)
    assert _vector_value(observation, "jetson_fill_log") > one_raw
    assert _vector_value(observation, "jetson_compressed_fill_log") > 0.05
    metadata.update(obc_data_mb=compressed_mb)
    assert _vector_value(observation, "obc_fill_log") > 0.05
    metadata.update(jetson_raw_mb=metadata["jetson_capacity_mb"], jetson_compressed_mb=0.0)
    assert _vector_value(observation, "jetson_fill_log") == pytest.approx(1.0)


def test_every_encoded_input_stays_within_the_declared_bounds() -> None:
    env = _environment(max_steps=240)
    space = observation_space(env.config["power"])
    low, high = np.asarray(space.low), np.asarray(space.high)
    assert low[EVENTSAT_OBSERVATIONS.index("last_platform_energy_norm")] < -1.9
    observation = env.reset(7)
    energies = []
    for step in range(env.max_steps):
        vector = encode_vectors(observation)[0]
        assert np.all(vector >= low - 1e-6) and np.all(vector <= high + 1e-6)
        energies.append(vector[EVENTSAT_OBSERVATIONS.index("last_platform_energy_norm")])
        mode = EVENTSAT_ACTIONS[(step // 6) % len(EVENTSAT_ACTIONS)]
        observation = env.step({"eventsat_0": {"mode": mode}}).observation
    assert min(energies) < -1.0


def test_countdowns_without_a_later_event_are_censored_labels() -> None:
    observation = _environment().reset(7)
    _metadata(observation).update(next_pass_known=False)
    labels = encode_vectors(observation)[1]
    assert labels[EVENTSAT_STATES.index("time_to_next_pass")] == -1.0


def test_repeated_launch_seeds_never_cross_the_split() -> None:
    seeds = (100, 101, 102, 103, 100, 101, 102, 103, 104, 104)
    split = split_episodes(seeds, train_fraction=0.6, seed=19)
    train = {seeds[index] for index in split.train}
    validation = {seeds[index] for index in split.validation}
    assert train and validation and train.isdisjoint(validation)
    assert sorted(split.train + split.validation) == list(range(len(seeds)))
    assert split_episodes(range(5), seed=4).train == split_episodes((9, 8, 7, 6, 5), seed=4).train


@pytest.mark.orekit
def test_navigation_distinguishes_launch_seeds_with_equal_elapsed_time() -> None:
    if not orekit.is_available():
        pytest.skip("Orekit, Java 17, or orekit-data.zip is unavailable")
    first = encode_vectors(_environment(2, prefer_orekit=True).reset(42))[0]
    second = encode_vectors(_environment(2, prefer_orekit=True).reset(43))[0]
    position = [EVENTSAT_OBSERVATIONS.index(f"position_itrf_{axis}_norm") for axis in "xyz"]
    assert np.linalg.norm(first[position] - second[position]) > 0.5


def test_onboard_rules_react_to_present_visibility_not_forecasts() -> None:
    observation = _environment().reset(7)
    metadata = _metadata(observation)
    metadata["obc_data_mb"] = 5.0
    rules = EventSatSymbolic()

    def decide(**updates: Any) -> str:
        record = deepcopy(observation)
        _metadata(record).update(updates)
        state = rules.encode_observation(record)
        assert not set(FORECAST_KEYS) & set(state)
        return rules.select_action(DecisionContext(state, record, None, 0, "onboard"))[
            "eventsat_0"
        ]["mode"]

    assert metadata["planning_contact_seconds"]
    assert decide(station_visible=False, contact_window_active=True) != "communication"
    assert decide(station_visible=True, contact_window_active=False) == "communication"
