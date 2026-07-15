"""Public-safe experiment provenance without machine identity or paths."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any


def scientific_config_sha256(config: dict[str, Any]) -> str:
    """Hash scientific inputs while deliberately ignoring output placement."""

    scientific = {key: value for key, value in config.items() if key != "output_root"}
    return hashlib.sha256(json.dumps(scientific, sort_keys=True, default=str).encode()).hexdigest()


def result_document_sha256(result: dict[str, Any]) -> str:
    """Hash immutable result content without recursively hashing its identifier."""

    payload = {key: value for key, value in result.items() if key != "result_id"}
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def collect_provenance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    source_revision = commit or _package_revision()
    return {
        "config_sha256": scientific_config_sha256(config),
        "source_revision": source_revision,
        "source_kind": "git" if commit else "installed-package",
        "git_commit": commit,
        "git_dirty": bool(_git(root, "status", "--porcelain")) if commit else False,
        "python": platform.python_version(),
        "dependencies": {
            name: _version(name)
            for name in (
                "numpy",
                "pydantic",
                "pyyaml",
                "torch",
                "orekit-jpype",
                "openai",
                "requests",
            )
        },
    }


def _package_revision() -> str | None:
    try:
        record = distribution("autops-framework").read_text("RECORD")
    except PackageNotFoundError:
        return None
    return hashlib.sha256(record.encode()).hexdigest() if record else None


def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


__all__ = ["collect_provenance", "result_document_sha256", "scientific_config_sha256"]
