"""Strict loader for the canonical EventSat world-model recipe.

The checked-in YAML is the runtime authority.  This module translates it into
the same typed configurations consumed by training, artifact fitting, and the
closed-loop planner so that a documentation-only recipe cannot drift from the
executed experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autops.config import asset_root, load_yaml
from autops.wm.cem import CEMConfig
from autops.wm.jepa import LeWMConfig
from autops.wm.probes import DEFAULT_ATTRIBUTES
from autops.wm.schema import EVENTSAT_ACTIONS, EVENTSAT_OBSERVATIONS, TRACE_SCHEMA_VERSION
from autops.wm.training import TrainingConfig

_RECIPE_FIELDS = {"dataset", "model", "loss", "training", "probes", "planner"}
_DATASET_FIELDS = {"schema", "history", "prediction_steps"}
_MODEL_FIELDS = {
    "observation_dim",
    "action_dim",
    "embed_dim",
    "encoder_hidden_dim",
    "projector_hidden_dim",
    "predictor_blocks",
    "predictor_heads",
    "predictor_head_dim",
    "predictor_mlp_dim",
    "dropout",
    "embedding_dropout",
}
_LOSS_FIELDS = {"sigreg_weight", "sigreg_knots", "sigreg_projections"}
_TRAINING_FIELDS = {
    "seed",
    "batch_size",
    "optimizer_steps",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "warmup_steps",
    "validation_interval",
    "train_fraction",
    "validation_sample_size",
    "train_loss_window",
    "device",
}
_PLANNER_FIELDS = {
    "artifact_root",
    "horizon",
    "samples",
    "elites",
    "iterations",
    "alpha",
    "min_probability",
    "plan_hold",
    "seed",
    "reserve_soc",
    "comms_soc_floor",
    "normalize_attribute_scale",
    "downlink_reflex",
    "exact_analytic_shaping",
    "lightweight_shaping",
    "contact_guidance",
    "contact_guidance_strength",
    "undeliverable_capacity_penalty",
    "downlink_reward",
    "pass_stage_reward",
    "downlink_shaping_reference_weight",
    "mode_weight_presets",
}


def _exact(mapping: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{context} must be a mapping")
    unknown = set(mapping) - expected
    missing = expected - set(mapping)
    if unknown or missing:
        raise ValueError(
            f"invalid {context} fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return mapping


@dataclass(frozen=True)
class ProbeRecipe:
    ridge: float
    attributes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.ridge < 0.0:
            raise ValueError("probe ridge must be non-negative")
        if self.attributes != DEFAULT_ATTRIBUTES:
            raise ValueError("probe attributes must match the canonical EventSat target order")


@dataclass(frozen=True)
class PlannerRecipe:
    artifact_root: str
    cem: CEMConfig
    reserve_soc: float
    comms_soc_floor: float
    normalize_attribute_scale: bool
    downlink_reflex: bool
    exact_analytic_shaping: bool
    lightweight_shaping: bool
    contact_guidance: bool
    contact_guidance_strength: float
    undeliverable_capacity_penalty: float
    downlink_reward: float
    pass_stage_reward: float
    downlink_shaping_reference_weight: float
    mode_weight_presets: dict[str, dict[str, float]]

    def __post_init__(self) -> None:
        if not self.artifact_root or Path(self.artifact_root).is_absolute():
            raise ValueError("planner artifact_root must be a non-empty relative path")
        if self.exact_analytic_shaping:
            raise ValueError("canonical LeWM planning must not use exact analytical shaping")
        if not 0.0 <= self.reserve_soc <= 1.0 or not 0.0 <= self.comms_soc_floor <= 1.0:
            raise ValueError("planner SOC thresholds must lie in [0, 1]")
        if not 0.0 <= self.contact_guidance_strength <= 1.0:
            raise ValueError("contact_guidance_strength must lie in [0, 1]")
        if self.undeliverable_capacity_penalty < 0.0:
            raise ValueError("undeliverable_capacity_penalty must be non-negative")
        if self.downlink_shaping_reference_weight <= 0.0:
            raise ValueError("downlink_shaping_reference_weight must be positive")
        if set(self.mode_weight_presets) != {"science", "safe", "downlink"}:
            raise ValueError("planner must define science, safe, and downlink presets")
        expected = set(DEFAULT_ATTRIBUTES)
        for name, weights in self.mode_weight_presets.items():
            if set(weights) != expected:
                raise ValueError(f"planner preset {name!r} must define every probe attribute")

    def representation_config(self) -> dict[str, Any]:
        """Return planner controls in the public representation configuration shape."""

        return {
            "horizon": self.cem.horizon,
            "samples": self.cem.samples,
            "elites": self.cem.elites,
            "iterations": self.cem.iterations,
            "alpha": self.cem.alpha,
            "min_probability": self.cem.min_probability,
            "plan_hold": self.cem.plan_hold,
            "seed": self.cem.seed,
            "reserve_soc": self.reserve_soc,
            "comms_soc_floor": self.comms_soc_floor,
            "normalize_attribute_scale": self.normalize_attribute_scale,
            "downlink_reflex": self.downlink_reflex,
            "exact_analytic_shaping": self.exact_analytic_shaping,
            "lightweight_shaping": self.lightweight_shaping,
            "contact_guidance": self.contact_guidance,
            "contact_guidance_strength": self.contact_guidance_strength,
            "undeliverable_capacity_penalty": self.undeliverable_capacity_penalty,
            "downlink_reward": self.downlink_reward,
            "pass_stage_reward": self.pass_stage_reward,
            "downlink_shaping_reference_weight": self.downlink_shaping_reference_weight,
        }


@dataclass(frozen=True)
class EventSatWMRecipe:
    model: LeWMConfig
    training: TrainingConfig
    probes: ProbeRecipe
    planner: PlannerRecipe


def _recipe_sections(
    recipe_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    payload = _exact(load_yaml(recipe_path), _RECIPE_FIELDS, "world-model recipe")
    return (
        _exact(payload["dataset"], _DATASET_FIELDS, "dataset"),
        _exact(payload["model"], _MODEL_FIELDS, "model"),
        _exact(payload["loss"], _LOSS_FIELDS, "loss"),
        _exact(payload["training"], _TRAINING_FIELDS, "training"),
        _exact(payload["probes"], {"ridge", "attributes"}, "probes"),
        _exact(payload["planner"], _PLANNER_FIELDS, "planner"),
    )


def _validate_dataset_model(dataset: dict[str, Any], model: dict[str, Any]) -> None:
    if str(dataset["schema"]) != TRACE_SCHEMA_VERSION:
        raise ValueError("recipe dataset schema does not match the trace contract")
    if int(model["observation_dim"]) != len(EVENTSAT_OBSERVATIONS):
        raise ValueError("recipe observation_dim is not canonical EventSat")
    if int(model["action_dim"]) != len(EVENTSAT_ACTIONS):
        raise ValueError("recipe action_dim is not canonical EventSat")


def _model_config(
    dataset: dict[str, Any], model: dict[str, Any], loss: dict[str, Any]
) -> LeWMConfig:
    return LeWMConfig(
        obs_dim=int(model["observation_dim"]),
        action_dim=int(model["action_dim"]),
        embed_dim=int(model["embed_dim"]),
        history=int(dataset["history"]),
        predictions=int(dataset["prediction_steps"]),
        encoder_hidden_dim=int(model["encoder_hidden_dim"]),
        predictor_depth=int(model["predictor_blocks"]),
        predictor_heads=int(model["predictor_heads"]),
        predictor_head_dim=int(model["predictor_head_dim"]),
        predictor_mlp_dim=int(model["predictor_mlp_dim"]),
        projector_hidden_dim=int(model["projector_hidden_dim"]),
        dropout=float(model["dropout"]),
        embedding_dropout=float(model["embedding_dropout"]),
        sigreg_weight=float(loss["sigreg_weight"]),
        sigreg_knots=int(loss["sigreg_knots"]),
        sigreg_projections=int(loss["sigreg_projections"]),
    )


def _training_config(training: dict[str, Any]) -> TrainingConfig:
    return TrainingConfig(
        max_steps=int(training["optimizer_steps"]),
        warmup_steps=int(training["warmup_steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip=float(training["gradient_clip"]),
        train_fraction=float(training["train_fraction"]),
        seed=int(training["seed"]),
        validation_interval=int(training["validation_interval"]),
        validation_sample_size=int(training["validation_sample_size"]),
        train_loss_window=int(training["train_loss_window"]),
        device=str(training["device"]),
    )


def _planner_recipe(planner: dict[str, Any], model: LeWMConfig) -> PlannerRecipe:
    cem = CEMConfig(
        horizon=int(planner["horizon"]),
        action_dim=model.action_dim,
        samples=int(planner["samples"]),
        elites=int(planner["elites"]),
        iterations=int(planner["iterations"]),
        alpha=float(planner["alpha"]),
        min_probability=float(planner["min_probability"]),
        plan_hold=int(planner["plan_hold"]),
        seed=int(planner["seed"]),
    )
    presets = {
        str(name): {str(attribute): float(weight) for attribute, weight in weights.items()}
        for name, weights in dict(planner["mode_weight_presets"]).items()
    }
    return PlannerRecipe(
        artifact_root=str(planner["artifact_root"]),
        cem=cem,
        reserve_soc=float(planner["reserve_soc"]),
        comms_soc_floor=float(planner["comms_soc_floor"]),
        normalize_attribute_scale=bool(planner["normalize_attribute_scale"]),
        downlink_reflex=bool(planner["downlink_reflex"]),
        exact_analytic_shaping=bool(planner["exact_analytic_shaping"]),
        lightweight_shaping=bool(planner["lightweight_shaping"]),
        contact_guidance=bool(planner["contact_guidance"]),
        contact_guidance_strength=float(planner["contact_guidance_strength"]),
        undeliverable_capacity_penalty=float(planner["undeliverable_capacity_penalty"]),
        downlink_reward=float(planner["downlink_reward"]),
        pass_stage_reward=float(planner["pass_stage_reward"]),
        downlink_shaping_reference_weight=float(planner["downlink_shaping_reference_weight"]),
        mode_weight_presets=presets,
    )


def load_eventsat_recipe(path: str | Path | None = None) -> EventSatWMRecipe:
    """Load and fully validate the one canonical EventSat LeWM recipe."""

    recipe_path = asset_root() / "configs" / "wm" / "eventsat.yaml" if path is None else Path(path)
    dataset, model, loss, training, probes, planner = _recipe_sections(recipe_path)
    _validate_dataset_model(dataset, model)
    model_config = _model_config(dataset, model, loss)
    return EventSatWMRecipe(
        model=model_config,
        training=_training_config(training),
        probes=ProbeRecipe(
            ridge=float(probes["ridge"]),
            attributes=tuple(str(value) for value in probes["attributes"]),
        ),
        planner=_planner_recipe(planner, model_config),
    )


__all__ = [
    "EventSatWMRecipe",
    "PlannerRecipe",
    "ProbeRecipe",
    "load_eventsat_recipe",
]
