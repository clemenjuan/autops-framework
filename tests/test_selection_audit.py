from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.replay import replay_records
from autops.core.selection_audit import audit_candidate_selection
from autops.core.workflows import fit_planner_artifact
from autops.wm.schema import load_trace, write_trace
from autops.wm.training import save_checkpoint


def _export(path: Path, seeds: list[int]) -> Path:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=len(seeds), steps=8, seeds=seeds)
    return export_trace(spec, path, prefer_orekit=False)


def test_replay_rebuilds_logged_records_and_detects_divergence(tmp_path) -> None:
    trace = load_trace(_export(tmp_path / "trace.npz", [5, 6]))
    records = replay_records(trace, 1, [0, 7])
    assert sorted(records) == [0, 7]
    assert records[7]["step"] == 7
    trace.mode[1, 2] = (trace.mode[1, 2] + 3) % 7
    with pytest.raises(ValueError, match="diverged from the trace at step 3"):
        replay_records(trace, 1, [7])


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
    from autops.core.selection_audit import _contexts

    near = np.zeros((2, 10), dtype=bool)
    near[0, 3:5] = True
    chosen = _contexts(near, (0, 1), 6, 0.5, np.random.default_rng(0))
    flags = [flag for _, _, flag in chosen]
    assert flags.count(True) == 2 and flags.count(False) == 4
    assert all(near[episode, step] == flag for episode, step, flag in chosen)
