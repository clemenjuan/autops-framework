"""Closed-loop EventSat LeWM-CEM representation.

The representation owns only deployment concerns: observation history, mission
action masks, plan-and-hold execution, and the physical-contact downlink
reflex. Artifact validation, CEM, and model reconstruction remain in
``autops.wm``. Torch is imported only when a real latent rollout is requested.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from autops.core.plugin import Representation, register
from autops.core.types import DecisionContext, SpaceSpec
from autops.missions.eventsat.physics import MODES, encode_vectors
from autops.wm.artifact import (
    PlannerArtifact,
    checkpoint_sha256,
    load_artifact,
    resolve_checkpoint,
)
from autops.wm.cem import (
    CEMConfig,
    categorical_cem,
    initial_probabilities,
)
from autops.wm.compute import PlannerComputeEvidence
from autops.wm.guidance import (
    CandidateProjection,
    admissible_action_mask,
    guided_probabilities,
    pipeline_scores,
    project_executable_candidates,
    seed_pipeline_candidate,
)
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS
from autops.wm.scoring import (
    analytical_candidate_attributes,
    latent_candidate_attributes,
    scalarization_weights,
    validate_planner_checkpoint,
)

RolloutScorer = Callable[[Mapping[str, Any], np.ndarray], np.ndarray]


def _configured_artifact(config: Mapping[str, Any]) -> tuple[PlannerArtifact, Path | None]:
    injected = config.get("artifact")
    path_value = config.get("artifact_path", config.get("planner_artifact"))
    artifact_path = Path(path_value) if path_value is not None else None
    if injected is not None:
        if not isinstance(injected, PlannerArtifact):
            raise TypeError("artifact must be a validated PlannerArtifact")
        return injected, artifact_path
    if artifact_path is None:
        raise ValueError("EventSat CEM requires artifact or artifact_path")
    return load_artifact(artifact_path), artifact_path


def _effective_cem(artifact: PlannerArtifact, config: Mapping[str, Any]) -> CEMConfig:
    overrides: dict[str, Any] = {}
    integer_keys = ("horizon", "samples", "elites", "plan_hold", "seed")
    for key in integer_keys:
        if key in config:
            overrides[key] = int(config[key])
    if "iterations" in config or "cem_iterations" in config:
        overrides["iterations"] = int(config.get("iterations", config.get("cem_iterations")))
    if "alpha" in config:
        overrides["alpha"] = float(config["alpha"])
    if "min_probability" in config:
        overrides["min_probability"] = float(config["min_probability"])
    return replace(artifact.cem, **overrides)


def _number(state: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(state.get(key, default))
    except (TypeError, ValueError):
        return default


@register("lewm-cem", mission="eventsat", role="onboard")
class EventSatLeWMCEM(Representation):
    """Thin closed-loop adapter around the canonical artifact and CEM core."""

    scorer_kind = "latent-terminal-affine"
    propagation_model = "lewm-recursive-rollout"
    uses_checkpoint = True

    observation_space = SpaceSpec((25,), "float32", -1.0, 1.0)
    action_space = SpaceSpec((7,), "int64", 0, 1, MODES)

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.artifact, self._artifact_path = _configured_artifact(self.config)
        if self.artifact.model.mission != "eventsat":
            raise ValueError("EventSat lewm-cem requires an EventSat model artifact")
        if self.artifact.model.action_names != EVENTSAT_ACTIONS:
            raise ValueError("EventSat lewm-cem requires the canonical seven-action order")
        if self.artifact.model.observation_names != EVENTSAT_OBSERVATIONS:
            raise ValueError("EventSat lewm-cem requires the canonical observation semantics")
        if self.artifact.model.obs_dim != 25:
            raise ValueError("EventSat lewm-cem requires the canonical 25D observation")
        self.cem = _effective_cem(self.artifact, self.config)
        scorer = self.config.get("rollout_scorer")
        if scorer is not None and not callable(scorer):
            raise TypeError("rollout_scorer must be callable")
        self._injected_scorer: RolloutScorer | None = scorer
        self._device = str(self.config.get("device", "cpu"))
        self._model: Any | None = None
        self._model_config: Any | None = None
        self._normalize_attribute_scale = bool(
            self.config.get("normalize_attribute_scale", self.artifact.normalize_attribute_scale)
        )
        self._weights, self._mission_weights = self._attribute_weights()
        self._reserve_soc = float(self.config.get("reserve_soc", 0.50))
        self._comms_soc_floor = float(self.config.get("comms_soc_floor", 0.25))
        self._downlink_reflex = bool(
            self.config.get("downlink_reflex", self.config.get("contact_reflex_enabled", True))
        )
        if bool(self.config.get("exact_analytic_shaping", False)):
            raise ValueError("exact_analytic_shaping is intentionally unsupported by lewm-cem")
        self._lightweight_shaping = bool(self.config.get("lightweight_shaping", True))
        self._contact_guidance = bool(self.config.get("contact_guidance", True))
        self._contact_guidance_strength = float(self.config.get("contact_guidance_strength", 0.75))
        if not np.isfinite(self._contact_guidance_strength):
            raise ValueError("contact_guidance_strength must be finite")
        self._contact_guidance_strength = min(1.0, max(0.0, self._contact_guidance_strength))
        self._undeliverable_capacity_penalty = float(
            self.config.get("undeliverable_capacity_penalty", 0.02)
        )
        if not np.isfinite(self._undeliverable_capacity_penalty):
            raise ValueError("undeliverable_capacity_penalty must be finite")
        self._undeliverable_capacity_penalty = max(0.0, self._undeliverable_capacity_penalty)
        self._downlink_reward = float(self.config.get("downlink_reward", 1.0))
        self._pass_stage_reward = float(self.config.get("pass_stage_reward", 0.15))
        self._downlink_reference_weight = max(
            1e-9, float(self.config.get("downlink_shaping_reference_weight", 0.25))
        )
        self._action_index = {name: index for index, name in enumerate(EVENTSAT_ACTIONS)}
        self._rng = np.random.default_rng(self.cem.seed)
        self._obs_history: list[np.ndarray] = []
        self._action_history: list[np.ndarray] = []
        self._held_actions: list[int] = []
        self._previous_solution: np.ndarray | None = None
        self._candidate_projection: CandidateProjection | None = None
        self._candidate_projection_step: int | None = None
        self._last_plan: tuple[int, ...] = ()
        self._last_action = self._action_index["charging"]
        self._compute = PlannerComputeEvidence()

    @property
    def last_plan(self) -> tuple[int, ...]:
        return self._last_plan

    def diagnostics(self) -> dict[str, Any]:
        return {
            **self._compute.to_dict(self.cem, self.artifact),
            "mission_mode": self._mission_mode,
            "scorer_kind": self.scorer_kind,
            "propagation_model": self.propagation_model,
            "uses_checkpoint": self.uses_checkpoint,
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._rng = np.random.default_rng(self.cem.seed if seed is None else int(seed))
        self._obs_history.clear()
        self._action_history.clear()
        self._held_actions.clear()
        self._previous_solution = None
        self._candidate_projection = None
        self._candidate_projection_step = None
        self._last_plan = ()
        self._last_action = self._action_index["charging"]
        self._compute.reset()

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        obs, _, raw = encode_vectors(observation)
        return {
            **raw,
            "obs25": obs,
            "timestep": int(observation.get("step", 0)),
            "step_duration_s": float(self.config.get("step_duration_s", 60.0)),
        }

    def mission_action_mask(self, state: Mapping[str, Any]) -> np.ndarray:
        """Return the same admissibility mask used at every projected horizon step."""

        return admissible_action_mask(
            state,
            reserve_soc=self._reserve_soc,
            comms_soc_floor=self._comms_soc_floor,
        )

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        state = context.state
        self._append_history(state)
        mask = self.mission_action_mask(state)
        held = self._held_actions.pop(0) if self._held_actions else None
        if self._should_reflex(state, mask):
            self._compute.reflex_overrides += 1
            if held is None:
                self._previous_solution = None
            return self._choose(
                self._action_index["communication"],
                planned=False,
                rationale="physical-contact OBC downlink reflex",
            )
        if held is not None:
            self._compute.held_action_steps += 1
            repaired = held if mask[held] else self._fallback_action(state, mask)
            rationale = "executed held LeWM-CEM action"
            if repaired != held:
                self._compute.held_action_repairs += 1
                rationale = "repaired inadmissible held action with mission mask"
            return self._choose(repaired, planned=False, rationale=rationale)

        history = self._history(state)
        self._candidate_projection = None
        self._candidate_projection_step = None

        def score(sequences: np.ndarray) -> np.ndarray:
            return self._score_candidates(history, sequences)

        def guide(probabilities: np.ndarray) -> np.ndarray:
            return self._contact_guided_probabilities(state, probabilities)

        def seed_candidates(sequences: np.ndarray) -> np.ndarray:
            return self._seed_pipeline_candidate(state, sequences, mask)

        def project_candidates(sequences: np.ndarray) -> np.ndarray:
            return self._project_candidates(state, sequences)

        started = perf_counter()
        result = categorical_cem(
            score,
            self.cem,
            first_action_mask=mask,
            previous_solution=self._previous_solution,
            initial=self._proposal_probabilities(),
            proposal_guidance=guide if self._contact_guidance else None,
            seed_candidates=seed_candidates if self._contact_guidance else None,
            project_candidates=project_candidates,
            rng=self._rng,
        )
        elapsed = max(0.0, perf_counter() - started)
        self._compute.record_plan(elapsed, samples=self.cem.samples, iterations=self.cem.iterations)
        sequence = result.action_sequence.astype(np.int64, copy=True)
        self._previous_solution = sequence
        self._last_plan = tuple(int(value) for value in sequence)
        self._held_actions = [int(value) for value in sequence[1 : self.cem.plan_hold]]
        return self._choose(
            int(sequence[0]),
            planned=True,
            rationale=f"LeWM-CEM planning event score={result.score:.6g}",
            planner_active_s=elapsed,
        )

    def _proposal_probabilities(self) -> np.ndarray:
        """Return the canonical cold prior or the shifted plan-hold warm start."""

        probabilities = initial_probabilities(self.cem, previous_solution=self._previous_solution)
        previous = self._previous_solution
        cold_start = previous is None
        if previous is not None:
            values = np.asarray(previous, dtype=np.int64).reshape(-1)
            cold_start = values[min(self.cem.plan_hold, values.size) :].size == 0
        if cold_start:
            probabilities[:, self._action_index["charging"]] += 0.08
            probabilities[:, self._action_index["safe"]] *= 0.20
            probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities

    def _attribute_weights(self) -> tuple[np.ndarray, dict[str, float]]:
        custom_weights = self.config.get("mission_weights", self.config.get("mode_weights"))
        if custom_weights is None:
            self._mission_mode = str(self.config.get("mission_mode", "science"))
            if not self._mission_mode:
                raise ValueError("mission_mode must be non-empty")
            supplied = self.artifact.mode_weight_presets.get(self._mission_mode)
            if supplied is None:
                raise ValueError(
                    f"unknown mission_mode {self._mission_mode!r}; artifact has no such preset"
                )
        else:
            self._mission_mode = "custom"
        if not isinstance(supplied, Mapping):
            raise TypeError("mission weights must be a name-to-weight mapping")
        return scalarization_weights(
            self.artifact,
            supplied,
            normalize_attribute_scale=self._normalize_attribute_scale,
        )

    def _append_history(self, state: Mapping[str, Any]) -> None:
        observation = np.asarray(state.get("obs25"), dtype=np.float32).reshape(-1)
        if observation.shape != (self.artifact.model.obs_dim,):
            raise ValueError("encoded observation does not match artifact.model.obs_dim")
        action = np.eye(self.artifact.model.action_dim, dtype=np.float32)[self._last_action]
        self._obs_history.append(observation)
        self._action_history.append(action)
        keep = self.artifact.model.history
        self._obs_history = self._obs_history[-keep:]
        self._action_history = self._action_history[-keep:]

    def _history(self, state: Mapping[str, Any]) -> dict[str, Any]:
        count = self.artifact.model.history
        observations = list(self._obs_history)
        actions = list(self._action_history)
        while len(observations) < count:
            observations.insert(0, observations[0].copy())
            actions.insert(
                0,
                np.eye(self.artifact.model.action_dim, dtype=np.float32)[
                    self._action_index["charging"]
                ],
            )
        return {
            "obs": np.stack(observations),
            "action": np.stack(actions),
            "state": dict(state),
        }

    def _score_candidates(self, history: Mapping[str, Any], sequences: np.ndarray) -> np.ndarray:
        values = (
            self._injected_scorer(history, sequences)
            if self._injected_scorer is not None
            else self._torch_attributes(history, sequences)
        )
        output = np.asarray(values, dtype=np.float64)
        if output.shape == (sequences.shape[0], len(self.artifact.probe.attribute_names)):
            output = output @ self._weights.astype(np.float64)
        elif output.shape != (sequences.shape[0],):
            raise ValueError("rollout_scorer must return [samples] scores or [samples, attributes]")
        if not np.isfinite(output).all():
            raise ValueError("rollout_scorer returned a non-finite value")
        if self._lightweight_shaping:
            projection = self._candidate_projection_for(history["state"], sequences)
            output += self._lightweight_pipeline_scores(
                history["state"], sequences, projection=projection
            )
        return output

    def _contact_guided_probabilities(
        self, state: Mapping[str, Any], probabilities: np.ndarray
    ) -> np.ndarray:
        return guided_probabilities(
            state,
            probabilities,
            enabled=self._contact_guidance,
            strength=self._contact_guidance_strength,
        )

    def _seed_pipeline_candidate(
        self, state: Mapping[str, Any], sequences: np.ndarray, first_mask: np.ndarray
    ) -> np.ndarray:
        return seed_pipeline_candidate(state, sequences, first_action_mask=first_mask)

    def _project_candidates(self, state: Mapping[str, Any], sequences: np.ndarray) -> np.ndarray:
        self._candidate_projection = project_executable_candidates(
            state,
            sequences,
            reserve_soc=self._reserve_soc,
            comms_soc_floor=self._comms_soc_floor,
        )
        self._candidate_projection_step = int(_number(state, "timestep", -1))
        return self._candidate_projection.sequences

    def _candidate_projection_for(
        self, state: Mapping[str, Any], sequences: np.ndarray
    ) -> CandidateProjection:
        projection = self._candidate_projection
        timestep = int(_number(state, "timestep", -1))
        if (
            projection is None
            or self._candidate_projection_step != timestep
            or not np.array_equal(projection.sequences, sequences)
        ):
            projection = project_executable_candidates(
                state,
                sequences,
                reserve_soc=self._reserve_soc,
                comms_soc_floor=self._comms_soc_floor,
            )
            self._candidate_projection = projection
            self._candidate_projection_step = timestep
        return projection

    def _lightweight_pipeline_scores(
        self,
        state: Mapping[str, Any],
        sequences: np.ndarray,
        *,
        projection: CandidateProjection | None = None,
    ) -> np.ndarray:
        return pipeline_scores(
            state,
            sequences,
            downlink_weight=self._mission_weights.get("downlink_progress", 0.0),
            downlink_reward=self._downlink_reward,
            pass_stage_reward=self._pass_stage_reward,
            reference_weight=self._downlink_reference_weight,
            undeliverable_penalty=self._undeliverable_capacity_penalty,
            reserve_soc=self._reserve_soc,
            comms_soc_floor=self._comms_soc_floor,
            projection=projection,
        )

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

    def _should_reflex(self, state: Mapping[str, Any], mask: np.ndarray) -> bool:
        return bool(
            self._downlink_reflex
            and state.get("physical_ground_pass_active", False)
            and _number(state, "obc_data_mb") > 0.01
            and mask[self._action_index["communication"]]
        )

    def _fallback_action(self, state: Mapping[str, Any], mask: np.ndarray) -> int:
        safe = self._action_index["safe"]
        if str(state.get("health_status", "nominal")) != "nominal" and mask[safe]:
            return safe
        charging = self._action_index["charging"]
        if mask[charging]:
            return charging
        return int(np.flatnonzero(mask)[0])

    def _choose(
        self,
        index: int,
        *,
        planned: bool,
        rationale: str,
        planner_active_s: float = 0.0,
    ) -> dict[str, Any]:
        self._last_action = int(index)
        if self._action_history:
            self._action_history[-1] = np.eye(self.artifact.model.action_dim, dtype=np.float32)[
                index
            ]
        mode = self.artifact.model.action_names[index]
        self._last_rationale = rationale
        action: dict[str, Any] = {"mode": mode, "jetson_planned": planned}
        if planned:
            action["planner_active_s"] = max(0.0, float(planner_active_s))
        return {"eventsat_0": action}


@register("analytical-cem", mission="eventsat", role="onboard")
class EventSatAnalyticalCEM(EventSatLeWMCEM):
    """CEM with exact EventSat propagation over the orbit-derived almanac."""

    scorer_kind = "analytical-terminal"
    propagation_model = "orbit-almanac+eventsat-physics"
    uses_checkpoint = False

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        if self._injected_scorer is not None:
            raise ValueError("analytical-cem does not accept an injected rollout scorer")

    def _score_candidates(
        self, history: Mapping[str, Any], sequences: np.ndarray
    ) -> np.ndarray:
        state = history["state"]
        projection = self._candidate_projection_for(state, sequences)
        attributes = analytical_candidate_attributes(
            projection, self.artifact.probe.attribute_names
        )
        scores = attributes.astype(np.float64) @ self._weights.astype(np.float64)
        if self._lightweight_shaping:
            scores += self._lightweight_pipeline_scores(
                state, sequences, projection=projection
            )
        return scores


__all__ = ["EventSatAnalyticalCEM", "EventSatLeWMCEM", "RolloutScorer"]
