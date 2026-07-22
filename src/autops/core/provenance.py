"""Public-safe experiment provenance without machine identity or paths."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any

# Microarchitecture levels, most capable first. The level selects which kernels
# NumPy and Torch dispatch to, which changes float reduction order and therefore
# the trajectories a closed-loop planner produces; runs are only bitwise
# comparable within one level.
_X86_LEVELS: tuple[tuple[str, frozenset[str]], ...] = (
    ("x86-64-v4", frozenset({"avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl"})),
    ("x86-64-v3", frozenset({"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe"})),
    ("x86-64-v2", frozenset({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})),
)


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
        "compute": {
            "machine": platform.machine(),
            "cpu_isa": _cpu_isa(),
            "cpu_count": os.cpu_count(),
        },
        "dependencies": {
            name: _version(name)
            for name in (
                "numpy",
                "pydantic",
                "pyyaml",
                "torch",
                "wandb",
                "orekit-jpype",
                "openai",
                "requests",
            )
        },
    }


def _cpu_flags() -> frozenset[str]:
    """Read the instruction-set flags Linux reports for the first core."""

    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    for line in text.splitlines():
        name, _, values = line.partition(":")
        if name.strip() in ("flags", "Features"):
            return frozenset(values.split())
    return frozenset()


def _cpu_isa() -> str | None:
    """Name the microarchitecture level, not the machine, so results stay public-safe."""

    flags = _cpu_flags()
    if not flags:
        return None
    for level, required in _X86_LEVELS:
        if required <= flags:
            return f"{level}+amx" if "amx_tile" in flags else level
    return "x86-64-v1" if "lm" in flags else platform.machine()


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
