"""Strict paper-result manifest selection and identity verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from autops.core.provenance import result_document_sha256

MANIFEST_SCHEMA_VERSION = "autops.paper-results-manifest/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_ENTRY_FIELDS = {
    "result_id",
    "commit_sha",
    "config_sha256",
    "checkpoint_sha256",
    "path",
}


@dataclass(frozen=True)
class ManifestEntry:
    result_id: str
    commit_sha: str
    config_sha256: str
    checkpoint_sha256: str | None
    path: Path


@dataclass(frozen=True)
class PaperResultManifest:
    paper_id: str
    approved: tuple[ManifestEntry, ...]
    diagnostic: tuple[ManifestEntry, ...]


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if not _SHA256.fullmatch(text):
        raise ValueError(f"manifest {label} must be a lowercase SHA-256")
    return text


def _entry(value: Any) -> ManifestEntry:
    if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
        raise ValueError(f"manifest entries must contain exactly {sorted(_ENTRY_FIELDS)}")
    commit = str(value["commit_sha"])
    if not _COMMIT.fullmatch(commit):
        raise ValueError("manifest commit_sha must be a lowercase 40-character Git SHA")
    checkpoint = value["checkpoint_sha256"]
    checkpoint_digest = None if checkpoint is None else _digest(checkpoint, "checkpoint_sha256")
    path = Path(str(value["path"]))
    if path.is_absolute() or ".." in path.parts or path.suffix != ".json":
        raise ValueError("manifest result paths must be safe relative JSON paths")
    return ManifestEntry(
        result_id=_digest(value["result_id"], "result_id"),
        commit_sha=commit,
        config_sha256=_digest(value["config_sha256"], "config_sha256"),
        checkpoint_sha256=checkpoint_digest,
        path=path,
    )


def load_result_manifest(path: str | Path) -> PaperResultManifest:
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    fields = {"schema_version", "paper_id", "approved_results", "diagnostic_results"}
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError(f"paper manifest must contain exactly {sorted(fields)}")
    if payload["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported paper result manifest schema")
    paper_id = str(payload["paper_id"])
    if not paper_id:
        raise ValueError("paper manifest requires a non-empty paper_id")
    approved_values = payload["approved_results"]
    diagnostic_values = payload["diagnostic_results"]
    if not isinstance(approved_values, list) or not isinstance(diagnostic_values, list):
        raise ValueError("paper manifest result collections must be arrays")
    manifest = PaperResultManifest(
        paper_id=paper_id,
        approved=tuple(_entry(value) for value in approved_values),
        diagnostic=tuple(_entry(value) for value in diagnostic_values),
    )
    identities = [entry.result_id for entry in (*manifest.approved, *manifest.diagnostic)]
    if len(identities) != len(set(identities)):
        raise ValueError("paper manifest contains a duplicate result ID")
    return manifest


def _checkpoint_identity(payload: dict[str, Any]) -> str | None:
    experiment = payload.get("experiment")
    if not isinstance(experiment, dict):
        return None
    identity = experiment.get("planner_artifact_identity")
    if not isinstance(identity, dict):
        return None
    value = identity.get("checkpoint_sha256")
    return str(value) if value is not None else None


def _verify_entry(entry: ManifestEntry, root: Path) -> Path:
    source = (root / entry.path).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError("manifest result path escapes the runtime root")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("result_id") != entry.result_id:
        raise ValueError(f"{entry.path}: result ID disagrees with manifest")
    if result_document_sha256(payload) != entry.result_id:
        raise ValueError(f"{entry.path}: result bytes disagree with result ID")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{entry.path}: result lacks provenance")
    if provenance.get("config_sha256") != entry.config_sha256:
        raise ValueError(f"{entry.path}: config hash disagrees with manifest")
    if provenance.get("git_commit") != entry.commit_sha:
        raise ValueError(f"{entry.path}: commit SHA disagrees with manifest")
    if _checkpoint_identity(payload) != entry.checkpoint_sha256:
        raise ValueError(f"{entry.path}: checkpoint hash disagrees with manifest")
    return source


def verified_result_paths(
    manifest_path: str | Path,
    runtime_root: str | Path,
    *,
    include_diagnostics: bool = False,
) -> list[Path]:
    """Return only identity-verified manifest paths in declared order."""

    manifest = load_result_manifest(manifest_path)
    entries = manifest.approved + (manifest.diagnostic if include_diagnostics else ())
    return [_verify_entry(entry, Path(runtime_root)) for entry in entries]


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManifestEntry",
    "PaperResultManifest",
    "load_result_manifest",
    "verified_result_paths",
]
