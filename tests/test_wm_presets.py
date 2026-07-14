from __future__ import annotations

import pytest

from autops.config import asset_root, load_yaml
from autops.representations.wm_planner import EventSatLeWMCEM
from autops.wm.artifact import (
    ModelContract,
    NormalizationContract,
    PlannerArtifact,
    ProbeContract,
    ProbeEvidenceContract,
)
from autops.wm.cem import CEMConfig
from autops.wm.probes import DEFAULT_ATTRIBUTES
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS


def _evidence(attributes: tuple[str, ...]) -> ProbeEvidenceContract:
    zeros = {name: 0.0 for name in attributes}
    return ProbeEvidenceContract(attributes, zeros, zeros, zeros, (0,), (1,), 1e-3, 1)


def _artifact() -> PlannerArtifact:
    attributes = DEFAULT_ATTRIBUTES
    return PlannerArtifact(
        model=ModelContract(
            checkpoint="model.pt",
            mission="eventsat",
            obs_dim=25,
            action_dim=7,
            embed_dim=8,
            history=3,
            observation_names=EVENTSAT_OBSERVATIONS,
            action_names=EVENTSAT_ACTIONS,
            trace_sha256="0" * 64,
            checkpoint_sha256="0" * 64,
        ),
        normalization=NormalizationContract(
            obs_mean=(0.0,) * 25,
            obs_std=(1.0,) * 25,
            action_mean=(0.0,) * 7,
            action_std=(1.0,) * 7,
        ),
        probe=ProbeContract(
            W=tuple((0.0,) * 8 for _ in attributes),
            b=(0.0,) * len(attributes),
            attribute_names=attributes,
            target_mean=(0.0,) * len(attributes),
            target_std=(1.0,) * len(attributes),
        ),
        probe_evidence=_evidence(attributes),
        cem=CEMConfig(horizon=2, samples=4, elites=1, iterations=1, plan_hold=1),
        mode_weight_presets={"science": {"science_progress": 1.0}},
    )


def test_canonical_config_defines_all_retargeting_presets() -> None:
    planner = load_yaml(asset_root() / "configs" / "wm" / "eventsat.yaml")["planner"]
    presets = planner["mode_weight_presets"]
    assert set(presets) == {"science", "safe", "downlink"}
    for weights in presets.values():
        assert set(weights) == set(DEFAULT_ATTRIBUTES)
        assert sum(abs(float(value)) for value in weights.values()) == pytest.approx(1.0)


def test_unknown_mission_mode_fails_instead_of_falling_back_to_science() -> None:
    with pytest.raises(ValueError, match="unknown mission_mode"):
        EventSatLeWMCEM(
            {
                "artifact": _artifact(),
                "mission_mode": "missing",
                "rollout_scorer": lambda history, sequences: sequences[:, 0] * 0.0,
            }
        )
