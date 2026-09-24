from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.replay import replay_environments
from autops.core.selection_audit import audit_candidate_selection
from autops.core.workflows import fit_planner_artifact
from autops.wm.schema import load_trace, write_trace
from autops.wm.training import save_checkpoint


def _export(path: Path, seeds: list[int]) -> Path:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=len(seeds), steps=8, seeds=seeds)
    return export_trace(spec, path, prefer_orekit=False)


def test_replay_rebuilds_logged_records_and_detects_divergence(tmp_path) -> None:
    trace = load_trace(_export(tmp_path / "trace.npz", [5, 6]))
    snapshots = replay_environments(trace, 1, [0, 7])
    assert sorted(snapshots) == [0, 7]
    assert snapshots[7].observe()["step"] == 7
    snapshots[0].step({"eventsat_0": {"mode": "payload_observe"}})
    assert snapshots[7].observe()["step"] == 7
    trace.mode[1, 2] = (trace.mode[1, 2] + 3) % 7
    with pytest.raises(ValueError, match="diverged from the trace at step 3"):
        replay_environments(trace, 1, [7])


def test_selection_audit_scores_one_bank_per_context(tmp_path, tiny_lewm) -> None:
    trace_path = _export(tmp_path / "trace.npz", [11, 12, 13, 14])
    checkpoint = save_checkpoint(tmp_path / "model.pt", tiny_lewm(load_trace(trace_path)))
    artifact = fit_planner_artifact(trace_path, checkpoint, tmp_path / "planner.json")["artifact"]
    test_path = _export(tmp_path / "test.npz", [21, 22])

    audit = audit_candidate_selection(
        trace_path,
        artifact,
        test_trace_path=test_path,
        output=tmp_path / "selection.json",
        contexts=6,
        candidates=32,
    )
    assert audit["evaluation"]["episodes"] == "test-trace"
    assert len(audit["contexts"]) == 6
    assert {context["seed"] for context in audit["contexts"]} <= {21, 22}
    assert all(context["lewm_cem_regret"] >= 0.0 for context in audit["contexts"])
    for scorer in ("lewm-cem", "random"):
        overlap = audit["metrics"]["all"][scorer]["top_elite_overlap"]
        assert 0.0 <= overlap <= 1.0
    assert (tmp_path / "selection.json").exists()

    with pytest.raises(ValueError, match="test seeds also occur"):
        audit_candidate_selection(trace_path, artifact, test_trace_path=trace_path, contexts=2)
    changed = load_trace(trace_path)
    changed.reward[0, 0] += 1.0
    with pytest.raises(ValueError, match="SHA-256"):
        audit_candidate_selection(write_trace(tmp_path / "changed.npz", changed), artifact)


def test_selection_contexts_are_stratified_by_contact(tmp_path) -> None:
    from autops.core.offline import sample_contexts

    near = np.zeros((2, 10), dtype=bool)
    near[0, 3:5] = True
    chosen = sample_contexts(near, (0, 1), 6, 0.5, np.random.default_rng(0))
    flags = [flag for _, _, flag in chosen]
    assert flags.count(True) == 2 and flags.count(False) == 4
    assert all(near[episode, step] == flag for episode, step, flag in chosen)


def test_forecast_audit_compares_rollouts_with_references(tmp_path, tiny_lewm) -> None:
    from autops.core.forecast_audit import audit_recursive_forecasts

    trace_path = _export(tmp_path / "trace.npz", [11, 12, 13, 14])
    checkpoint = save_checkpoint(tmp_path / "model.pt", tiny_lewm(load_trace(trace_path)))
    artifact = fit_planner_artifact(trace_path, checkpoint, tmp_path / "planner.json")["artifact"]

    audit = audit_recursive_forecasts(
        trace_path,
        artifact,
        test_trace_path=_export(tmp_path / "test.npz", [21, 22]),
        contexts=4,
        steps=3,
    )
    assert audit["context_count"]["all"] == 4
    assert audit["attributes"]["downlink_progress"] == "flow"
    assert audit["attributes"]["battery_margin"] == "stock"
    methods = audit["metrics"]["all"]
    assert set(methods) == {
        "lewm-rollout",
        "encoded-truth-readout",
        "persistence",
        "analytical-projection",
    }
    storage = methods["analytical-projection"]["storage_margin"]
    assert len(storage["rmse"]) == 3
    assert max(storage["rmse"]) < 1e-5
    with pytest.raises(ValueError, match="shorter than an episode"):
        audit_recursive_forecasts(trace_path, artifact, steps=8)


def test_counterfactual_audit_compares_model_and_simulator_responses(tmp_path, tiny_lewm) -> None:
    from autops.core.counterfactual_audit import audit_action_conditioning

    trace_path = _export(tmp_path / "trace.npz", [11, 12, 13, 14])
    checkpoint = save_checkpoint(tmp_path / "model.pt", tiny_lewm(load_trace(trace_path)))
    artifact = fit_planner_artifact(trace_path, checkpoint, tmp_path / "planner.json")["artifact"]

    audit = audit_action_conditioning(
        trace_path,
        artifact,
        test_trace_path=_export(tmp_path / "test.npz", [21, 22]),
        contexts=3,
        steps=3,
        random_sequences=1,
    )
    assert audit["context_count"] == 3
    assert len(audit["sequences"]) == 8
    assert audit["sequences"][0] == ["charging"] * 3
    metrics = audit["metrics"]
    assert metrics["exogenous_spread"]["simulator"] == 0.0
    assert len(metrics["response"]["science_progress"]["rmse"]) == 3
    assert 0.0 <= metrics["altered_step_fraction"] <= 1.0
    json.dumps(audit, allow_nan=False)


def test_event_labels_measure_pass_start_duration_and_eclipse_edges(tmp_path) -> None:
    from autops.core.event_audit import EVENTS, event_labels
    from autops.wm.schema import EVENTSAT_STATES

    trace = load_trace(_export(tmp_path / "trace.npz", [5, 6]))
    column = {name: EVENTSAT_STATES.index(name) for name in EVENTSAT_STATES}
    state = trace.state[0]
    state[:] = 0.0
    state[:, column["time_to_next_pass"]] = [2, 1, 0, 4, -1, -1, -1, -1]
    state[:, column["physical_contact_seconds"]] = [0, 0, 30, 60, 20, 0, 0, 40]
    state[:, column["in_sunlight"]] = [1, 1, 0, 0, 1, 1, 1, 1]
    state[:, column["time_to_next_eclipse"]] = [2, 1, -1, -1, -1, -1, -1, -1]
    labels = event_labels(trace)[0]
    start, duration, entry, exit_ = (EVENTS.index(name) for name in EVENTS)
    np.testing.assert_allclose(labels[:3, start], [2, 1, 0])
    np.testing.assert_allclose(labels[:3, duration], [110, 110, 110])
    assert np.isnan(labels[3, duration]) and np.isnan(labels[4, start])
    np.testing.assert_allclose(labels[:2, entry], [2, 1])
    np.testing.assert_allclose(labels[2:4, exit_], [2, 1])
    assert np.isnan(labels[0, exit_])


def test_event_audit_scores_every_method_on_shared_contexts(tmp_path, tiny_lewm) -> None:
    from autops.core.event_audit import audit_event_timing

    trace_path = _export(tmp_path / "trace.npz", [11, 12, 13, 14])
    checkpoint = save_checkpoint(tmp_path / "model.pt", tiny_lewm(load_trace(trace_path)))
    artifact = fit_planner_artifact(trace_path, checkpoint, tmp_path / "planner.json")["artifact"]
    audit = audit_event_timing(
        trace_path,
        artifact,
        test_trace_path=_export(tmp_path / "test.npz", [21, 22]),
        contexts=4,
        steps=3,
        lookahead_steps=6,
    )
    assert set(audit["metrics"]) == {
        "recurrence",
        "physics",
        "record-readout",
        "latent-readout",
        "lewm-rollout",
    }
    assert audit["context_count"] == 4
    json.dumps(audit, allow_nan=False)
