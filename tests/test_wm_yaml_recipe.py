from __future__ import annotations

from dataclasses import asdict

import pytest
import yaml

from autops.config import asset_root
from autops.wm.jepa import LeWMConfig
from autops.wm.probes import DEFAULT_ATTRIBUTES
from autops.wm.recipe import load_eventsat_recipe
from autops.wm.schema import TRACE_SCHEMA_VERSION
from autops.wm.training import TrainingConfig


def test_checked_in_recipe_is_the_exact_canonical_runtime_recipe() -> None:
    recipe = load_eventsat_recipe()

    assert recipe.model == LeWMConfig()
    assert recipe.training == TrainingConfig()
    assert recipe.probes.attributes == DEFAULT_ATTRIBUTES
    assert recipe.probes.ridge == pytest.approx(1e-3)
    assert recipe.planner.cem.horizon == 12
    assert recipe.planner.cem.samples == 256
    assert recipe.planner.cem.elites == 32
    assert recipe.planner.cem.iterations == 4
    assert recipe.planner.cem.seed == 3072
    assert recipe.planner.representation_config()["comms_soc_floor"] == pytest.approx(0.25)


def test_recipe_rejects_unknown_fields_and_stale_schema(tmp_path) -> None:
    source = asset_root() / "configs" / "wm" / "eventsat.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert payload["dataset"]["schema"] == TRACE_SCHEMA_VERSION

    payload["training"]["optimizer_stepz"] = payload["training"].pop("optimizer_steps")
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid training fields"):
        load_eventsat_recipe(path)


def test_recipe_yaml_edits_materially_change_typed_execution_config(tmp_path) -> None:
    source = asset_root() / "configs" / "wm" / "eventsat.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["training"]["learning_rate"] = 1e-4
    payload["planner"]["plan_hold"] = 6
    path = tmp_path / "edited.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    recipe = load_eventsat_recipe(path)
    assert asdict(recipe.training)["learning_rate"] == pytest.approx(1e-4)
    assert recipe.planner.cem.plan_hold == 6
