from __future__ import annotations

import json
from pathlib import Path

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner


def test_override_variants_write_distinct_coordinate_hash_directories(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("AUTOPS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    first_spec = expand_coordinate(
        "eventsat/sas/ao/symb",
        episodes=1,
        steps=1,
        seeds=[41],
        overrides={"mission": {"anomalies": {"probability_per_step": 0.0}}},
    )
    second_spec = expand_coordinate(
        "eventsat/sas/ao/symb",
        episodes=1,
        steps=1,
        seeds=[41],
        overrides={"mission": {"anomalies": {"probability_per_step": 0.002}}},
    )

    first = ExperimentRunner(first_spec, prefer_orekit=False).run()
    second = ExperimentRunner(second_spec, prefer_orekit=False).run()
    first_hash = first["provenance"]["config_sha256"]
    second_hash = second["provenance"]["config_sha256"]
    assert first_hash != second_hash

    coordinate_root = tmp_path / "results" / "eventsat" / "sas" / "ao" / "symb"
    first_path = coordinate_root / first_hash[:12] / "results.json"
    second_path = coordinate_root / second_hash[:12] / "results.json"
    assert first_path.is_file()
    assert second_path.is_file()
    assert (
        json.loads(first_path.read_text(encoding="utf-8"))["provenance"]["config_sha256"]
        == first_hash
    )
    assert (
        json.loads(second_path.read_text(encoding="utf-8"))["provenance"]["config_sha256"]
        == second_hash
    )


def test_installed_style_result_write_targets_cwd_not_package(monkeypatch, tmp_path: Path) -> None:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=1, seeds=[7])
    package_root = tmp_path / "site-packages" / "autops"
    package_root.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.delenv("AUTOPS_ROOT", raising=False)
    monkeypatch.chdir(work)

    destination = ExperimentRunner(spec)._write_result({"provenance": {"config_sha256": "a" * 64}})

    assert destination == (
        work / "results" / "eventsat" / "sas" / "ao" / "symb" / ("a" * 12) / "results.json"
    )
    assert destination.is_file()
    assert not (package_root / "results").exists()

    explicit = tmp_path / "explicit-runtime"
    monkeypatch.setenv("AUTOPS_ROOT", str(explicit))
    explicit_destination = ExperimentRunner(spec)._write_result(
        {"provenance": {"config_sha256": "b" * 64}}
    )
    assert explicit_destination.is_relative_to(explicit)
    assert explicit_destination.is_file()
