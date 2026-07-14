from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from autops.commands import main
from autops.config import expand_coordinate
from autops.core.exporter import export_trace, export_traces
from autops.core.provenance import scientific_config_sha256
from autops.wm.schema import TraceMetadata, load_trace


def test_mixed_policy_export_preserves_sources_and_canonical_episode_axis(tmp_path: Path) -> None:
    ao = expand_coordinate("eventsat/sas/ao/symb", episodes=2, steps=4, seeds=[7, 8])
    ag = expand_coordinate("eventsat/sas/ag/symb", episodes=2, steps=4, seeds=[7, 8])

    trace = load_trace(export_traces((ao, ag), tmp_path / "mixed.npz", prefer_orekit=False))

    assert trace.n_episodes == 4
    assert trace.episode_id.tolist() == [0, 1, 2, 3]
    assert trace.episode_seed.tolist() == [7, 8, 7, 8]
    assert trace.metadata.timestep_s == 60.0
    assert [source.coordinate for source in trace.metadata.sources] == [
        ao.coordinate,
        ag.coordinate,
    ]
    assert [source.episode_count for source in trace.metadata.sources] == [2, 2]
    assert [source.seeds for source in trace.metadata.sources] == [(7, 8), (7, 8)]
    assert trace.metadata.sources[0].config_sha256 == scientific_config_sha256(
        ao.model_dump(mode="json")
    )
    assert all(
        source.source_kind in {"git", "installed-package"} for source in trace.metadata.sources
    )
    assert all(len(source.source_revision) in {40, 64} for source in trace.metadata.sources)
    assert all(source.orbital_backend == "simplified" for source in trace.metadata.sources)
    assert all(isinstance(source.source_dirty, bool) for source in trace.metadata.sources)


def test_mixed_export_rejects_incompatible_episode_geometry(tmp_path: Path) -> None:
    short = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=3)
    long = expand_coordinate("eventsat/sas/ag/symb", episodes=1, steps=4)

    with pytest.raises(ValueError, match="episode steps"):
        export_traces((short, long), tmp_path / "bad.npz", prefer_orekit=False)


def test_mixed_export_rejects_incompatible_mission_and_timestep(tmp_path: Path) -> None:
    base = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=2)
    wrong_timestep = base.model_copy(update={"timestep_s": 30.0})
    wrong_mission = expand_coordinate("ssa/sas/ao/symb", episodes=1, steps=2)

    with pytest.raises(ValueError, match="timestep_s"):
        export_traces((base, wrong_timestep), tmp_path / "time.npz", prefer_orekit=False)
    with pytest.raises(ValueError, match="mission"):
        export_traces((base, wrong_mission), tmp_path / "mission.npz", prefer_orekit=False)


def test_serialized_source_provenance_rejects_tampering(tmp_path: Path) -> None:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=2, seeds=[9])
    original = export_trace(spec, tmp_path / "original.npz", prefer_orekit=False)
    with np.load(original, allow_pickle=False) as blob:
        arrays = {name: blob[name].copy() for name in blob.files}
    metadata = json.loads(str(arrays["__metadata_json__"].item()))
    metadata["sources"][0]["config_sha256"] = "not-a-digest"
    arrays["__metadata_json__"] = np.asarray(json.dumps(metadata, sort_keys=True))
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)

    with pytest.raises(ValueError, match="config_sha256"):
        load_trace(tampered)


def test_trace_metadata_rejects_unknown_source_fields(tmp_path: Path) -> None:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=2)
    path = export_trace(spec, tmp_path / "roundtrip.npz", prefer_orekit=False)
    payload = load_trace(path).metadata.to_dict()
    payload["sources"][0]["local_path"] = "forbidden"

    with pytest.raises(ValueError, match="unknown trace source fields"):
        TraceMetadata.from_dict(payload)


def test_trace_metadata_rejects_invalid_dirty_and_backend_provenance(tmp_path: Path) -> None:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=1, steps=2)
    path = export_trace(spec, tmp_path / "strict-source.npz", prefer_orekit=False)
    payload = load_trace(path).metadata.to_dict()
    payload["sources"][0]["source_dirty"] = 1
    with pytest.raises(ValueError, match="source_dirty"):
        TraceMetadata.from_dict(payload)

    payload = load_trace(path).metadata.to_dict()
    payload["sources"][0]["orbital_backend"] = "not-applicable"
    with pytest.raises(ValueError, match="orbital_backend"):
        TraceMetadata.from_dict(payload)


def test_cli_accepts_multiple_coordinates_and_reports_per_coordinate_episodes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "cli-mixed.npz"
    assert (
        main(
            [
                "export",
                "eventsat/sas/ao/symb",
                "eventsat/sas/ag/symb",
                "--episodes",
                "1",
                "--steps",
                "2",
                "--no-orekit",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["experiments"] == ["eventsat_sas_ao_symb", "eventsat_sas_ag_symb"]
    assert result["episodes_per_coordinate"] == 1
    assert load_trace(output).n_episodes == 2
