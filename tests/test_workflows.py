from __future__ import annotations

import pytest

from autops.config import expand_coordinate
from autops.core.exporter import export_trace
from autops.core.workflows import _training_tracking_config, matrix_coordinates
from autops.wm.recipe import load_eventsat_recipe
from autops.wm.schema import load_trace, trace_sha256


def test_matrix_sweep_expands_runtime_cells_without_reserved_rl() -> None:
    coordinates = matrix_coordinates("eventsat")
    assert "eventsat/sas/ag/symb" in coordinates
    assert "eventsat/sas/ao/lewm-cem" in coordinates
    assert "eventsat/sas/ah/lewm-cem/hllm-a" in coordinates
    assert not any("/rl" in coordinate or "/hrl" in coordinate for coordinate in coordinates)


def test_matrix_sweep_filters_and_fails_when_empty() -> None:
    assert matrix_coordinates("ssa", organisation="imas") == ["ssa/imas/ao/symb"]
    with pytest.raises(ValueError, match="no runnable"):
        matrix_coordinates("ssa", representation="llm-s")


def test_wandb_tracking_config_uses_canonical_trace_digest_key(tmp_path) -> None:
    spec = expand_coordinate(
        "eventsat/sas/ao/symb", episodes=2, steps=4, seeds=[7, 8]
    )
    trace = load_trace(export_trace(spec, tmp_path / "trace.npz", prefer_orekit=False))
    recipe = load_eventsat_recipe()

    config = _training_tracking_config(trace, recipe.model, recipe.training)

    assert config["trace"]["trace_sha256"] == trace_sha256(trace)
