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


def test_ssa_rl_result_hash_covers_the_published_policy_identity(monkeypatch) -> None:
    from autops.core import ssa_runner
    from autops.core.provenance import scientific_config_sha256

    identity = {"source": "checkpoint", "checkpoint_sha256": "c" * 64}
    monkeypatch.setattr(
        ssa_runner, "_run_episode", lambda spec, episode_id, seed: {"metrics": {"utility": 1.0}}
    )
    monkeypatch.setattr(ssa_runner, "policy_identity", lambda episodes: identity)
    spec = expand_coordinate("ssa/sas/ao/rl", episodes=1, steps=2, seeds=[3])
    result = ssa_runner.run_ssa_experiment(spec)
    assert result["experiment"]["rl_policy_identity"] == identity
    assert result["provenance"]["config_sha256"] == scientific_config_sha256(result["experiment"])
