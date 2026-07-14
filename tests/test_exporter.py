from __future__ import annotations

from pathlib import Path

from autops.config import expand_coordinate
from autops.core.exporter import SSA_OBSERVATIONS, export_trace
from autops.wm.schema import EVENTSAT_OBSERVATIONS, load_trace


def test_eventsat_export_uses_shared_schema(tmp_path: Path) -> None:
    spec = expand_coordinate("eventsat/sas/ao/symb", episodes=2, steps=4, seeds=[7, 8])
    trace = load_trace(export_trace(spec, tmp_path / "eventsat.npz", prefer_orekit=False))
    assert trace.obs.shape == (2, 4, 25)
    assert trace.metadata.observation_names == EVENTSAT_OBSERVATIONS
    assert trace.episode_seed.tolist() == [7, 8]
    source = trace.metadata.sources[0]
    assert source.orbital_backend == "simplified"
    assert isinstance(source.source_dirty, bool)


def test_ssa_export_adds_satellite_axis_and_collective_fields(tmp_path: Path) -> None:
    spec = expand_coordinate(
        "ssa/imas/ao/symb",
        episodes=1,
        steps=3,
        constellation_size=2,
    )
    trace = load_trace(export_trace(spec, tmp_path / "ssa.npz"))
    assert trace.obs.shape == (1, 3, 2, len(SSA_OBSERVATIONS))
    assert trace.action.shape == (1, 3, 2, 6)
    assert trace.metadata.satellite_ids == ("sat_0", "sat_1")
    assert set(trace.collective) == {
        "delivered_coverage",
        "onboard_coverage",
        "archive_records",
    }
    assert trace.metadata.sources[0].orbital_backend == "not-applicable"
