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
    guided_probabilities,
    pipeline_scores,
    seed_pipeline_candidate,
)
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS
from autops.wm.scoring import (
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
        raise ValueError("lewm-cem requires artifact or artifact_path")
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
        }

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._rng = np.random.default_rng(self.cem.seed if seed is None else int(seed))
        self._obs_history.clear()
        self._action_history.clear()
        self._held_actions.clear()
        self._previous_solution = None
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
        """Return actions admissible now; future CEM steps remain unconstrained."""

        mask = np.zeros(len(EVENTSAT_ACTIONS), dtype=bool)
        mask[self._action_index["charging"]] = True
        health = str(state.get("health_status", "nominal"))
        soc = _number(state, "battery_soc", 0.5)
        if health != "nominal" or soc <= 0.22:
            mask[self._action_index["safe"]] = True
            return mask

        obc = _number(state, "obc_data_mb")
        raw = _number(state, "jetson_raw_mb")
        compressed = _number(state, "jetson_compressed_mb")
        capacity = max(1.0, _number(state, "storage_capacity_mb", 4096.0))
        stored = _number(state, "data_stored_mb", obc + raw + compressed)
        physical = bool(state.get("physical_ground_pass_active", False))
        estimated = bool(state.get("contact_window_active", state.get("ground_pass_active", False)))
        settling = max(0, int(_number(state, "settling_time_steps")))
        time_to_pass = _number(state, "time_to_next_pass", float("inf"))
        precontact = not (physical or estimated) and 0.0 < time_to_pass <= settling
        mask[self._action_index["communication"]] = (
            (physical or estimated or precontact) and obc > 0.01 and soc >= self._comms_soc_floor
        )
        if soc < self._reserve_soc:
            return mask
        mask[self._action_index["payload_observe"]] = stored < 0.80 * capacity
        mask[self._action_index["payload_compress"]] = (
            _number(state, "uncompressed_observations") > 0.0
        )
        mask[self._action_index["payload_detect"]] = _number(state, "undetected_observations") > 0.0
        mask[self._action_index["payload_send"]] = compressed > 0.01 and obc < 0.98 * capacity
        return mask

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
                rationale = "repaired inadmissible held action with mission mask"
            return self._choose(repaired, planned=False, rationale=rationale)

        history = self._history(state)

        def score(sequences: np.ndarray) -> np.ndarray:
            return self._score_candidates(history, sequences)

        def guide(probabilities: np.ndarray) -> np.ndarray:
            return self._contact_guided_probabilities(state, probabilities)

        def seed_candidates(sequences: np.ndarray) -> np.ndarray:
            return self._seed_pipeline_candidate(state, sequences, mask)

        started = perf_counter()
        result = categorical_cem(
            score,
            self.cem,
            first_action_mask=mask,
            previous_solution=self._previous_solution,
            initial=self._proposal_probabilities(),
            proposal_guidance=guide if self._contact_guidance else None,
            seed_candidates=seed_candidates if self._contact_guidance else None,
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
            output += self._lightweight_pipeline_scores(history["state"], sequences)
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

    def _lightweight_pipeline_scores(
        self, state: Mapping[str, Any], sequences: np.ndarray
    ) -> np.ndarray:
        return pipeline_scores(
            state,
            sequences,
            downlink_weight=self._mission_weights.get("downlink_progress", 0.0),
            downlink_reward=self._downlink_reward,
            pass_stage_reward=self._pass_stage_reward,
            reference_weight=self._downlink_reference_weight,
            undeliverable_penalty=self._undeliverable_capacity_penalty,
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

    def _choose(self, index: int, *, planned: bool, rationale: str) -> dict[str, Any]:
        self._last_action = int(index)
        if self._action_history:
            self._action_history[-1] = np.eye(self.artifact.model.action_dim, dtype=np.float32)[
                index
            ]
        mode = self.artifact.model.action_names[index]
        self._last_rationale = rationale
        return {"eventsat_0": {"mode": mode, "jetson_planned": planned}}


__all__ = ["EventSatLeWMCEM", "RolloutScorer"]
