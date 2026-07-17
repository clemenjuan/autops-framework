from __future__ import annotations

from itertools import product

import pytest

from autops.config import asset_root, expand_coordinate, load_yaml
from autops.core.exporter import export_trace
from autops.core.workflows import _training_tracking_config, matrix_coordinates
from autops.wm.recipe import load_eventsat_recipe
from autops.wm.schema import load_trace, trace_sha256


def test_matrix_sweep_expands_runtime_cells_without_reserved_rl() -> None:
    coordinates = matrix_coordinates("eventsat")
    assert len(coordinates) == 23
    assert "eventsat/sas/ag/symb" in coordinates
    assert {
        f"eventsat/sas/ao/{token}" for token in ("llm-s", "llm-a", "hllm-s", "hllm-a")
    } <= set(coordinates)
    assert "eventsat/sas/ao/analytical-cem" in coordinates
    assert "eventsat/sas/ao/lewm-cem" in coordinates
    assert "eventsat/sas/ah/lewm-cem/hllm-a" in coordinates
    assert not any("/rl" in coordinate or "/hrl" in coordinate for coordinate in coordinates)


def test_eventsat_declared_design_extends_historical_32_to_43_cells() -> None:
    matrix = load_yaml(asset_root() / "configs" / "matrix.yaml")
    baseline: set[tuple[str, ...]] = set()
    for paradigm, rule in matrix["canonical_eventsat_32"].items():
        if paradigm == "ah":
            baseline.update(
                (paradigm, onboard, ground)
                for onboard, ground in product(rule["onboard"], rule["ground"])
            )
        else:
            slot = "onboard" if paradigm == "ao" else "ground"
            baseline.update((paradigm, token) for token in rule[slot])

    runnable = {
        tuple(coordinate.split("/")[2:]) for coordinate in matrix_coordinates("eventsat")
    }

    assert len(baseline) == 32
    assert len(runnable) == 23
    assert len(baseline | runnable) == 43


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
