from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from autops.board.manifest import load_result_manifest, verified_result_paths
from autops.core.provenance import result_document_sha256


def _result(root: Path, name: str, *, checkpoint: str | None) -> tuple[Path, dict]:
    payload = {
        "schema_version": 1,
        "experiment": (
            {"planner_artifact_identity": {"checkpoint_sha256": checkpoint}}
            if checkpoint is not None
            else {}
        ),
        "provenance": {
            "config_sha256": "b" * 64,
            "git_commit": "a" * 40,
        },
        "name": name,
    }
    payload["result_id"] = result_document_sha256(payload)
    path = root / "results" / f"{payload['result_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, payload


def _entry(root: Path, path: Path, payload: dict, checkpoint: str | None) -> dict:
    return {
        "result_id": payload["result_id"],
        "commit_sha": "a" * 40,
        "config_sha256": "b" * 64,
        "checkpoint_sha256": checkpoint,
        "path": str(path.relative_to(root)),
    }


def test_manifest_selects_approved_results_and_keeps_diagnostics_separate(
    tmp_path: Path,
) -> None:
    approved_path, approved = _result(tmp_path, "approved", checkpoint=None)
    diagnostic_path, diagnostic = _result(tmp_path, "diagnostic", checkpoint="c" * 64)
    manifest = {
        "schema_version": "autops.paper-results-manifest/v1",
        "paper_id": "paper-a",
        "approved_results": [_entry(tmp_path, approved_path, approved, None)],
        "diagnostic_results": [_entry(tmp_path, diagnostic_path, diagnostic, "c" * 64)],
    }
    manifest_path = tmp_path / "paper-a.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    assert verified_result_paths(manifest_path, tmp_path) == [approved_path]
    assert verified_result_paths(manifest_path, tmp_path, include_diagnostics=True) == [
        approved_path,
        diagnostic_path,
    ]


def test_manifest_rejects_tampered_result_bytes(tmp_path: Path) -> None:
    result_path, payload = _result(tmp_path, "approved", checkpoint=None)
    manifest = {
        "schema_version": "autops.paper-results-manifest/v1",
        "paper_id": "paper-a",
        "approved_results": [_entry(tmp_path, result_path, payload, None)],
        "diagnostic_results": [],
    }
    manifest_path = tmp_path / "paper-a.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    payload["name"] = "tampered"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="result bytes"):
        verified_result_paths(manifest_path, tmp_path)


def test_canonical_paper_a_manifest_starts_with_no_approved_rows() -> None:
    manifest = load_result_manifest(Path("configs/papers/paper_a.yaml"))

    assert manifest.paper_id == "paper-a-compute-aware-planning"
    assert manifest.approved == ()
