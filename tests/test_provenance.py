from __future__ import annotations

import hashlib
import json
from pathlib import Path

from autops.core import provenance


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


def test_scientific_config_hash_excludes_output_root() -> None:
    first = {"coordinate": "eventsat/sas/ag/symb", "output_root": "results-a"}
    second = {"coordinate": "eventsat/sas/ag/symb", "output_root": "results-b"}

    assert provenance.scientific_config_sha256(first) == provenance.scientific_config_sha256(second)
