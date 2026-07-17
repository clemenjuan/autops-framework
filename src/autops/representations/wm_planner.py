"""Closed-loop EventSat LeWM-CEM representation.

The latent leaf of the shared EventSat CEM adapter: it scores candidates by
rolling the learned world model forward and reading terminal attributes
through the frozen affine probes. Torch is imported only when a real latent
rollout is requested; tests inject a deterministic ``rollout_scorer``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from autops.core.plugin import register
from autops.representations.cem_planner import EventSatCEMBase
from autops.wm.artifact import checkpoint_sha256, resolve_checkpoint
from autops.wm.guidance import CandidateProjection
from autops.wm.scoring import latent_candidate_attributes, validate_planner_checkpoint

RolloutScorer = Callable[[Mapping[str, Any], np.ndarray], np.ndarray]


@register("lewm-cem", mission="eventsat", role="onboard")
class EventSatLeWMCEM(EventSatCEMBase):
    """Learned latent CEM planner deployed from the canonical artifact."""

    token = "lewm-cem"
    scorer_kind = "latent-terminal-affine"
    propagation_model = "lewm-recursive-rollout"
    uses_checkpoint = True

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        scorer = self.config.get("rollout_scorer")
        if scorer is not None and not callable(scorer):
            raise TypeError("rollout_scorer must be callable")
        self._injected_scorer: RolloutScorer | None = scorer
        self._device = str(self.config.get("device", "cpu"))
        self._model: Any | None = None
        self._model_config: Any | None = None

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = super().diagnostics()
        if self._injected_scorer is not None:
            diagnostics.update(
                scorer_kind="injected-rollout",
                propagation_model="injected-rollout",
                uses_checkpoint=False,
            )
        return diagnostics

    def _score_candidates(
        self,
        history: Mapping[str, Any],
        sequences: np.ndarray,
        projection: CandidateProjection | None = None,
    ) -> np.ndarray:
        values = (
            self._injected_scorer(history, sequences)
            if self._injected_scorer is not None
            else self._torch_attributes(history, sequences)
        )
        output = np.asarray(values, dtype=np.float64)
        if output.shape == (sequences.shape[0], len(self.artifact.probe.attribute_names)):
            output = output @ self._weights
        elif output.shape != (sequences.shape[0],):
            raise ValueError("rollout_scorer must return [samples] scores or [samples, attributes]")
        if not np.isfinite(output).all():
            raise ValueError("rollout_scorer returned a non-finite value")
        if self._lightweight_shaping:
            state = history["state"]
            if projection is None:
                projection = self._project_executable(state, sequences)
            output += self._lightweight_pipeline_scores(state, projection)
        return output

    def _torch_attributes(self, history: Mapping[str, Any], sequences: np.ndarray) -> np.ndarray:
        _, model = self._torch_model()
        return latent_candidate_attributes(
            model,
            self.artifact,
            np.asarray(history["obs"], dtype=np.float32),
            np.asarray(history["action"], dtype=np.float32),
            sequences,
            device=self._device,
        )

    def _torch_model(self) -> tuple[Any, Any]:
        from autops.wm.jepa import require_torch
        from autops.wm.training import load_checkpoint

        torch = require_torch()
        if self._model is None:
            checkpoint_value = self.config.get("checkpoint_path")
            if checkpoint_value is not None:
                checkpoint = Path(checkpoint_value)
            elif self._artifact_path is not None:
                checkpoint = resolve_checkpoint(self._artifact_path, self.artifact)
            else:
                raise ValueError("real LeWM rollout requires artifact_path or checkpoint_path")
            digest = checkpoint_sha256(checkpoint)
            self._model, checkpoint_contract = load_checkpoint(checkpoint, device=self._device)
            self._model_config = checkpoint_contract.model_config
            validate_planner_checkpoint(
                self.artifact,
                checkpoint_contract,
                digest,
                checkpoint.stat().st_size,
            )
        return torch, self._model


__all__ = ["EventSatLeWMCEM", "RolloutScorer"]
