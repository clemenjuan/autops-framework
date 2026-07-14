from __future__ import annotations

import json

from autops.config import expand_coordinate
from autops.core.runner import ExperimentRunner


def test_runner_persists_redacted_llm_diagnostics() -> None:
    spec = expand_coordinate(
        "eventsat/sas/ag/llm-s",
        steps=5,
        overrides={"representation": {"llm_mock": True}},
    )
    result = ExperimentRunner(spec, save=False, prefer_orekit=False).run()
    diagnostics = result["episodes"][0]["decision_diagnostics"]["ground"]
    serialized = json.dumps(diagnostics)
    assert diagnostics["llm_provenance"]["mock"] is True
    assert "endpoint" not in serialized
    assert "prompt" not in serialized
