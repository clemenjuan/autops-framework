"""Public-safe experiment provenance without machine identity or paths."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def collect_provenance(config: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "git_commit": _git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git(root, "status", "--porcelain")),
        "python": platform.python_version(),
        "dependencies": {name: _version(name) for name in ("numpy", "pydantic", "pyyaml")},
    }


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
