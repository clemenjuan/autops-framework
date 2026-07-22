from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

from autops.core import provenance

_V2 = frozenset({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})
_V3 = _V2 | {"avx", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe"}
_V4 = _V3 | {"avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl"}


def test_installed_package_revision_is_boardable(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    monkeypatch.setattr(provenance, "_package_revision", lambda: "b" * 64)
    config = {"coordinate": "eventsat/sas/ag/symb"}
    result = provenance.collect_provenance(config, tmp_path)
    assert result["source_revision"] == "b" * 64
    assert result["source_kind"] == "installed-package"
    assert result["git_commit"] is None
    assert result["git_dirty"] is False
    assert (
        result["config_sha256"]
        == hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()
    )


def test_checkout_revision_takes_precedence(monkeypatch, tmp_path: Path) -> None:
    def fake_git(root, *args):
        del root
        return "a" * 40 if args == ("rev-parse", "HEAD") else ""

    monkeypatch.setattr(provenance, "_git", fake_git)
    monkeypatch.setattr(provenance, "_package_revision", lambda: "b" * 64)
    result = provenance.collect_provenance({}, tmp_path)
    assert result["source_revision"] == "a" * 40
    assert result["source_kind"] == "git"


def test_cpu_isa_reports_the_most_capable_level(monkeypatch) -> None:
    for flags, expected in ((_V2, "x86-64-v2"), (_V3, "x86-64-v3"), (_V4, "x86-64-v4")):
        monkeypatch.setattr(provenance, "_cpu_flags", lambda flags=flags: flags)
        assert provenance._cpu_isa() == expected

    monkeypatch.setattr(provenance, "_cpu_flags", lambda: _V4 | {"amx_tile"})
    assert provenance._cpu_isa() == "x86-64-v4+amx"


def test_cpu_isa_handles_non_x86_and_unreadable_flags(monkeypatch) -> None:
    monkeypatch.setattr(provenance, "_cpu_flags", lambda: frozenset({"asimd", "neon"}))
    assert provenance._cpu_isa() == platform.machine()

    monkeypatch.setattr(provenance, "_cpu_flags", frozenset)
    assert provenance._cpu_isa() is None


def test_compute_provenance_stays_public_safe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(provenance, "_git", lambda *args: None)
    result = provenance.collect_provenance({}, tmp_path)

    assert set(result["compute"]) == {"machine", "cpu_isa", "cpu_count"}
    node = platform.node()
    assert not node or node not in json.dumps(result)


def test_scientific_config_hash_excludes_output_root() -> None:
    first = {"coordinate": "eventsat/sas/ag/symb", "output_root": "results-a"}
    second = {"coordinate": "eventsat/sas/ag/symb", "output_root": "results-b"}

    assert provenance.scientific_config_sha256(first) == provenance.scientific_config_sha256(second)
