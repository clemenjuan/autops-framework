"""Configuration expansion and scoped plugin-discovery tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import autops.config as config_module
import autops.core.plugin as plugins
from autops.config import expand_coordinate


def test_expand_runnable_matrix_coordinate_with_paired_seeds() -> None:
    spec = expand_coordinate(
        "eventsat/sas/ag/symb",
        episodes=2,
        steps=24,
        seeds=[101, 202],
    )
    assert spec.coordinate == "eventsat/sas/ag/symb"
    assert spec.name == "eventsat_sas_ag_symb"
    assert spec.ground_token == "symb"
    assert spec.onboard_token is None
    assert spec.seeds == [101, 202]
    assert spec.mission_config["power"]["battery"]["capacity_wh"] == 70.0
    assert spec.mission_config["communications"]["sband"]["downlink_rate_kbps"] == 50


@pytest.mark.parametrize("token", ["llm-s", "llm-a", "hllm-s", "hllm-a"])
def test_eventsat_ao_llm_coordinates_activate_jetson_compute(token: str) -> None:
    spec = expand_coordinate(f"eventsat/sas/ao/{token}")

    assert spec.onboard_token == token
    assert spec.onboard_uses_jetson


@pytest.mark.parametrize("token", ["rl", "hrl"])
def test_expand_rejects_reserved_deferred_representations(token: str) -> None:
    with pytest.raises(ValueError, match="reserved but not implemented"):
        expand_coordinate(f"eventsat/sas/ag/{token}")


def test_overrides_reject_unknown_top_level_and_nested_mission_keys() -> None:
    with pytest.raises(ValueError, match=r"Unknown top-level override keys.*misson"):
        expand_coordinate("eventsat/sas/ag/symb", overrides={"misson": {}})
    with pytest.raises(ValueError, match=r"mission\.power\.battery\.capacity_what"):
        expand_coordinate(
            "eventsat/sas/ag/symb",
            overrides={"mission": {"power": {"battery": {"capacity_what": 75.0}}}},
        )
    with pytest.raises(ValueError, match=r"Unknown top-level override keys.*metrics"):
        expand_coordinate("eventsat/sas/ag/symb", overrides={"metrics": {}})
    with pytest.raises(ValueError, match=r"Unknown top-level override keys.*paradigm"):
        expand_coordinate("eventsat/sas/ag/symb", overrides={"paradigm": {}})
    with pytest.raises(ValueError, match="overrides must be a mapping"):
        expand_coordinate("eventsat/sas/ag/symb", overrides=["mission"])
    with pytest.raises(ValueError, match=r"output_root.*path string"):
        expand_coordinate("eventsat/sas/ag/symb", overrides={"output_root": {}})

    spec = expand_coordinate(
        "eventsat/sas/ag/symb",
        overrides={"mission": {"power": {"battery": {"capacity_wh": 75.0}}}},
    )
    assert spec.mission_config["power"]["battery"]["capacity_wh"] == 75.0
    assert not hasattr(spec, "metrics_config")
    assert not hasattr(spec, "paradigm_config")


@pytest.mark.parametrize(
    "output_root",
    [Path("../escape"), Path("nested/../../escape"), Path("/absolute/results")],
)
def test_output_root_must_be_safe_and_relative(output_root: Path) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        expand_coordinate(
            "eventsat/sas/ag/symb",
            overrides={"output_root": output_root},
        )


def test_installed_assets_and_runtime_writes_have_distinct_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package_root = tmp_path / "site-packages" / "autops"
    (package_root / "configs").mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(config_module, "__file__", str(package_root / "config.py"))
    monkeypatch.delenv("AUTOPS_ROOT", raising=False)
    monkeypatch.chdir(work)

    assert config_module.asset_root() == package_root
    assert config_module.runtime_root() == work

    explicit = tmp_path / "explicit-runtime"
    monkeypatch.setenv("AUTOPS_ROOT", str(explicit))
    assert config_module.runtime_root() == explicit
    assert config_module.asset_root() == package_root


def test_discovery_scans_only_the_requested_mission(monkeypatch: pytest.MonkeyPatch) -> None:
    packages: list[tuple[str, bool]] = []

    def record(package_name: str, *, optional: bool = False) -> None:
        packages.append((package_name, optional))

    monkeypatch.setattr(plugins, "_discover_package", record)
    monkeypatch.setattr(plugins, "_discover_entry_points", lambda: None)
    plugins.discover_representations("eventsat")
    assert packages == [
        ("autops.representations", False),
        ("autops.missions.eventsat", True),
    ]
