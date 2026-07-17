"""Matched analytical EventSat CEM reference.

The analytical leaf of the shared EventSat CEM adapter: it scores candidates
from the exact projected EventSat transitions over the orbit-derived almanac,
isolating propagation-model quality from the optimizer and executable
candidates. It requires no torch checkpoint at scoring time; the artifact
still binds the matched comparison's provenance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from autops.core.plugin import register
from autops.representations.cem_planner import EventSatCEMBase
from autops.wm.guidance import CandidateProjection
from autops.wm.probes import DEFAULT_ATTRIBUTES
from autops.wm.scoring import analytical_candidate_attributes


@register("analytical-cem", mission="eventsat", role="onboard")
class EventSatAnalyticalCEM(EventSatCEMBase):
    """CEM with exact EventSat propagation over the orbit-derived almanac."""

    token = "analytical-cem"
    scorer_kind = "analytical-terminal"
    propagation_model = "orbit-almanac+eventsat-physics"
    uses_checkpoint = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is not None and config.get("rollout_scorer") is not None:
            raise ValueError("analytical-cem does not accept an injected rollout scorer")
        super().__init__(config)
        unsupported = set(self.artifact.probe.attribute_names) - set(DEFAULT_ATTRIBUTES)
        if unsupported:
            raise ValueError(f"analytical-cem cannot score probe attributes: {sorted(unsupported)}")

    def _score_candidates(
        self,
        history: Mapping[str, Any],
        sequences: np.ndarray,
        projection: CandidateProjection | None = None,
    ) -> np.ndarray:
        state = history["state"]
        if projection is None:
            projection = self._project_executable(state, sequences)
        attributes = analytical_candidate_attributes(
            projection, self.artifact.probe.attribute_names
        )
        scores = attributes.astype(np.float64) @ self._weights
        if not np.isfinite(scores).all():
            raise ValueError("analytical candidate attributes produced a non-finite score")
        if self._lightweight_shaping:
            scores += self._lightweight_pipeline_scores(state, projection)
        return scores


__all__ = ["EventSatAnalyticalCEM"]
