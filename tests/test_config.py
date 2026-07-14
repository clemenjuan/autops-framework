"""Configuration expansion and scoped plugin-discovery tests."""

from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("token", ["rl", "hrl"])
def test_expand_rejects_reserved_deferred_representations(token: str) -> None:
    with pytest.raises(ValueError, match="reserved but not implemented"):
        expand_coordinate(f"eventsat/sas/ag/{token}")


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
