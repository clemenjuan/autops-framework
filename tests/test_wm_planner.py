from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from autops.core.plugin import create_representation, registered_plugins
from autops.core.types import DecisionContext
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.artifact import (
    ModelContract,
    NormalizationContract,
    PlannerArtifact,
    ProbeContract,
    ProbeEvidenceContract,
    artifact_sha256,
)
from autops.wm.cem import CEMConfig
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS


def _evidence(attributes: tuple[str, ...]) -> ProbeEvidenceContract:
    zeros = {name: 0.0 for name in attributes}
    normalized = {name: (0.25 if index == 0 else None) for index, name in enumerate(attributes)}
    return ProbeEvidenceContract(attributes, zeros, normalized, normalized, (0,), (1,), 1e-3, 1)


def _artifact(*, plan_hold: int = 3) -> PlannerArtifact:
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
            target_std=(1.0, 100.0),
            degenerate=("downlink_progress",),
        ),
        probe_evidence=_evidence(attributes),
        cem=CEMConfig(
            horizon=4,
            samples=128,
            elites=16,
            iterations=4,
            plan_hold=plan_hold,
            seed=17,
        ),
        mode_weight_presets={"science": dict.fromkeys(attributes, 1.0)},
    )


def _state(**updates: Any) -> dict[str, Any]:
    state: dict[str, Any] = {
        "obs25": np.zeros(25, dtype=np.float32),
        "battery_soc": 0.8,
        "health_status": "nominal",
        "physical_ground_pass_active": False,
        "contact_window_active": False,
        "obc_data_mb": 0.0,
        "jetson_raw_mb": 0.0,
        "jetson_compressed_mb": 0.0,
        "data_stored_mb": 0.0,
        "storage_capacity_mb": 100.0,
        "uncompressed_observations": 0,
        "undetected_observations": 0,
    }
    state.update(updates)
    return state


def _context(state: dict[str, Any], step: int = 0) -> DecisionContext:
    return DecisionContext(
        state=state,
        observation={},
        memory=None,
        step=step,
        role="onboard",
    )


def _mode(action: dict[str, Any]) -> str:
    return str(action["eventsat_0"]["mode"])


def _planner(
    scorer: Callable[[dict[str, Any], np.ndarray], np.ndarray],
    **config: Any,
) -> EventSatLeWMCEM:
    return EventSatLeWMCEM({"artifact": _artifact(), "rollout_scorer": scorer, **config})


def test_plugin_registers_for_eventsat_onboard() -> None:
    plugins = registered_plugins("eventsat")
    assert plugins[("eventsat", "lewm-cem", "onboard")] is EventSatLeWMCEM

    representation = create_representation(
        "eventsat",
        "lewm-cem",
        "onboard",
        {
            "artifact": _artifact(),
            "rollout_scorer": lambda history, sequences: np.zeros(sequences.shape[0]),
        },
    )
    assert isinstance(representation, EventSatLeWMCEM)


def test_plan_hold_reuses_actions_without_calling_rollout_scorer() -> None:
    calls: list[tuple[int, ...]] = []

    def scorer(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        assert history["obs"].shape == (3, 25)
        assert history["action"].shape == (3, 7)
        calls.append(tuple(sequences[:, 0]))
        return (sequences == 2).sum(axis=1).astype(np.float32)

    planner = _planner(scorer)
    planner.reset(9)
    actions = [planner.select_action(_context(_state(), step)) for step in range(4)]

    assert [action["eventsat_0"]["jetson_planned"] for action in actions] == [
        True,
        False,
        False,
        True,
    ]
    assert len(calls) == 8
    first_plan = planner.last_plan
    planner.reset(9)
    planner.select_action(_context(_state()))
    assert planner.last_plan == first_plan


def test_probe_target_scale_normalization_is_enabled_by_default() -> None:
    def attributes(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        del history
        result = np.zeros((sequences.shape[0], 2), dtype=np.float32)
        result[:, 0] = np.where(sequences[:, 0] == 2, 2.0, 0.0)
        result[:, 1] = np.where(sequences[:, 0] == 0, 100.0, 0.0)
        return result

    normalized = _planner(attributes)
    unnormalized = _planner(attributes, normalize_attribute_scale=False)

    assert _mode(normalized.select_action(_context(_state()))) == "payload_observe"
    assert _mode(unnormalized.select_action(_context(_state()))) == "charging"


def test_downlink_reflex_requires_physical_contact_and_obc_data() -> None:
    calls = 0

    def charging_score(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        nonlocal calls
        del history
        calls += 1
        return -(sequences != 0).sum(axis=1).astype(np.float32)

    predicted = _planner(charging_score)
    predicted_action = predicted.select_action(
        _context(_state(contact_window_active=True, obc_data_mb=5.0))
    )
    assert _mode(predicted_action) == "charging"
    assert calls == 4

    physical = _planner(charging_score)
    physical_action = physical.select_action(
        _context(_state(physical_ground_pass_active=True, obc_data_mb=5.0))
    )
    assert _mode(physical_action) == "communication"
    assert physical_action["eventsat_0"]["jetson_planned"] is False
    assert calls == 4


def test_mission_mask_applies_to_planning_and_repairs_held_actions() -> None:
    observed_first_actions: list[np.ndarray] = []

    def observe_mask(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        del history
        observed_first_actions.append(sequences[:, 0].copy())
        return (sequences == 2).sum(axis=1).astype(np.float32)

    planner = _planner(observe_mask)
    baseline = _state()
    mask = planner.mission_action_mask(baseline)
    assert set(np.flatnonzero(mask)) == {0, 2}
    planner.select_action(_context(baseline))
    assert all(set(values.tolist()) <= {0, 2} for values in observed_first_actions)

    low_power = _state(battery_soc=0.3)
    held = planner.select_action(_context(low_power, 1))
    assert _mode(held) == "charging"
    assert held["eventsat_0"]["jetson_planned"] is False
    assert "repaired" in str(planner.last_rationale)

    anomaly_mask = planner.mission_action_mask(_state(health_status="payload_fault"))
    assert set(np.flatnonzero(anomaly_mask)) == {0, 6}


def test_reflex_overrides_and_consumes_a_held_action() -> None:
    def payload_score(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        del history
        return (sequences == 2).sum(axis=1).astype(np.float32)

    planner = _planner(payload_score)
    planner.select_action(_context(_state()))
    reflex = planner.select_action(
        _context(_state(physical_ground_pass_active=True, obc_data_mb=4.0), 1)
    )
    after_reflex = planner.select_action(_context(_state(), 2))

    assert _mode(reflex) == "communication"
    assert reflex["eventsat_0"]["jetson_planned"] is False
    assert after_reflex["eventsat_0"]["jetson_planned"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [("mission", "ssa"), ("action_names", tuple(reversed(EVENTSAT_ACTIONS)))],
)
def test_checkpoint_load_rejects_artifact_semantic_mismatch(
    monkeypatch, tmp_path, field: str, value: Any
) -> None:
    pytest.importorskip("torch")
    artifact = _artifact()
    semantics = {
        "mission": artifact.model.mission,
        "observation_names": artifact.model.observation_names,
        "action_names": artifact.model.action_names,
        "trace_sha256": artifact.model.trace_sha256,
    }
    semantics[field] = value
    contract = SimpleNamespace(
        model_config=SimpleNamespace(
            obs_dim=25,
            action_dim=7,
            embed_dim=4,
            history=3,
        ),
        normalizer=SimpleNamespace(
            obs_mean=np.zeros(25, dtype=np.float32),
            obs_std=np.ones(25, dtype=np.float32),
            action_mean=np.zeros(7, dtype=np.float32),
            action_std=np.ones(7, dtype=np.float32),
        ),
        **semantics,
    )
    monkeypatch.setattr(
        "autops.wm.training.load_checkpoint",
        lambda checkpoint, device: (object(), contract),
    )
    monkeypatch.setattr(
        "autops.representations.wm_planner.checkpoint_sha256",
        lambda checkpoint: artifact.model.checkpoint_sha256,
    )
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"x")
    planner = EventSatLeWMCEM({"artifact": artifact, "checkpoint_path": str(checkpoint)})

    with pytest.raises(ValueError, match="mission/data semantics"):
        planner._torch_model()


def test_compute_diagnostics_are_timed_resettable_and_identity_bound(monkeypatch) -> None:
    ticks = iter([10.0, 10.25, 20.0, 20.75])
    monkeypatch.setattr("autops.representations.wm_planner.perf_counter", lambda: next(ticks))

    def payload_score(history: dict[str, Any], sequences: np.ndarray) -> np.ndarray:
        del history
        return (sequences == 2).sum(axis=1).astype(np.float32)

    planner = _planner(payload_score)
    planner.reset(9)
    planner.select_action(_context(_state(), 0))
    planner.select_action(_context(_state(), 1))
    planner.select_action(_context(_state(physical_ground_pass_active=True, obc_data_mb=4.0), 2))
    planner.select_action(_context(_state(), 3))

    diagnostics = planner.diagnostics()
    assert diagnostics["planning_events"] == 2
    assert diagnostics["held_action_steps"] == 1
    assert diagnostics["reflex_overrides"] == 1
    assert diagnostics["cem_latency_total_s"] == pytest.approx(1.0)
    assert diagnostics["cem_latency_mean_s"] == pytest.approx(0.5)
    assert diagnostics["evaluated_rollouts"] == 2 * 128 * 4
    assert diagnostics["rollouts_per_second"] == pytest.approx(1024.0)
    assert {
        key: diagnostics[key] for key in ("plan_hold", "horizon", "samples", "elites", "iterations")
    } == {
        "plan_hold": 3,
        "horizon": 4,
        "samples": 128,
        "elites": 16,
        "iterations": 4,
    }
    assert diagnostics["artifact_identity"] == {
        "schema_version": planner.artifact.schema_version,
        "trace_sha256": "0" * 64,
        "sha256": artifact_sha256(planner.artifact),
    }
    assert diagnostics["checkpoint_identity"] == {
        "relative_path": "weights/lewm.ckpt",
        "sha256": "0" * 64,
    }
    assert diagnostics["checkpoint_size_bytes"] == 1
    assert diagnostics["probe_rmse_over_std_mean"] == pytest.approx(0.25)
    assert diagnostics["probe_rmse_over_std"] == {
        "science_progress": 0.25,
        "downlink_progress": None,
    }

    planner.reset(9)
    reset = planner.diagnostics()
    assert reset["planning_events"] == 0
    assert reset["held_action_steps"] == 0
    assert reset["reflex_overrides"] == 0
    assert reset["cem_latency_total_s"] == 0.0
    assert reset["evaluated_rollouts"] == 0
