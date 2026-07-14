from __future__ import annotations

import pytest

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner


@pytest.mark.parametrize("organisation", ["sas", "cmas", "dmas", "hmas", "imas"])
def test_ssa_runner_smokes_every_organisation(organisation: str) -> None:
    spec = expand_coordinate(
        f"ssa/{organisation}/ao/symb",
        episodes=1,
        steps=2,
        constellation_size=1,
    )
    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    assert result["schema_version"] == 1
    assert result["episodes"][0]["steps"] == 2
    assert "ssa_custody_utility" in result["metrics"]
    assert result["provenance"]["config_sha256"]
