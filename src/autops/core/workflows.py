"""Reusable command workflows for sweeps, LeWM training, probes, and artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root, expand_coordinate, load_yaml
from autops.core.provenance import collect_provenance
from autops.core.runner import ExperimentRunner
from autops.wm.artifact import (
    ModelContract,
    NormalizationContract,
    PlannerArtifact,
    ProbeContract,
    ProbeEvidenceContract,
    checkpoint_sha256,
    save_artifact,
)
from autops.wm.evaluation import evaluate_lewm_cem
from autops.wm.probes import DEFAULT_ATTRIBUTES, ProbeFit, build_eventsat_targets, fit_ridge_probe
from autops.wm.recipe import load_eventsat_recipe
from autops.wm.schema import EVENTSAT_OBSERVATIONS, load_trace, trace_sha256
from autops.wm.tracking import WandbTrainingRun
from autops.wm.training import (
    load_checkpoint,
    save_checkpoint,
    train_lewm,
)


def matrix_coordinates(
    mission: str,
    *,
    organisation: str | None = None,
    paradigm: str | None = None,
    representation: str | None = None,
) -> list[str]:
    """Expand only implemented runtime cells; reserved tokens never appear."""

    matrix = load_yaml(asset_root() / "configs" / "matrix.yaml")
    mission_rule = matrix.get("missions", {}).get(mission)
    if not isinstance(mission_rule, dict):
        raise ValueError(f"unknown mission {mission!r}")
    organisations = [
        token
        for token in mission_rule.get("organisations", [])
        if organisation is None or token == organisation
    ]
    coordinates: list[str] = []
    for organisation_token in organisations:
        for paradigm_token, rule in mission_rule.get("paradigms", {}).items():
            if paradigm is not None and paradigm_token != paradigm:
                continue
            if paradigm_token == "ah":
                pairs = product(rule.get("onboard", []), rule.get("ground", []))
                for onboard, ground in pairs:
                    if representation is None or representation in {onboard, ground}:
                        coordinates.append(f"{mission}/{organisation_token}/ah/{onboard}/{ground}")
                continue
            slot = "onboard" if paradigm_token == "ao" else "ground"
            for token in rule.get(slot, []):
                if representation is None or token == representation:
                    coordinates.append(f"{mission}/{organisation_token}/{paradigm_token}/{token}")
    if not coordinates:
        raise ValueError("filters select no runnable matrix coordinates")
    return coordinates


def run_sweep(
    mission: str,
    *,
    episodes: int,
    steps: int | None,
    seeds: list[int] | None,
    overrides: dict[str, Any],
    organisation: str | None = None,
    paradigm: str | None = None,
    representation: str | None = None,
    prefer_orekit: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    coordinates = matrix_coordinates(
        mission,
        organisation=organisation,
        paradigm=paradigm,
        representation=representation,
    )
    specs = [
        expand_coordinate(
            coordinate,
            episodes=episodes,
            steps=steps,
            seeds=seeds,
            overrides=overrides,
        )
        for coordinate in coordinates
    ]
    if dry_run:
        return {"coordinates": coordinates, "completed": 0}
    summaries: list[dict[str, Any]] = []
    for coordinate, spec in zip(coordinates, specs, strict=True):
        result = ExperimentRunner(spec, prefer_orekit=prefer_orekit).run()
        summaries.append({"coordinate": coordinate, "metrics": result["metrics"]})
    return {"coordinates": coordinates, "completed": len(summaries), "results": summaries}


def train_world_model(
    trace_path: str | Path,
    output: str | Path,
    *,
    max_steps: int = 150_000,
    batch_size: int = 64,
    device: str = "cpu",
    wandb_project: str = "space-world-models",
    wandb_entity: str | None = None,
    wandb_name: str | None = None,
) -> dict[str, Any]:
    trace = load_trace(trace_path)
    recipe = load_eventsat_recipe()
    config = replace(recipe.training, max_steps=max_steps, batch_size=batch_size, device=device)
    digest = trace_sha256(trace)
    tracking_config = _training_tracking_config(trace, recipe.model, config)
    tracker = WandbTrainingRun.start(
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_name or f"autops-{trace.metadata.mission}-lewm-{digest[:8]}-{device}",
        config=tracking_config,
        trace_path=trace_path,
        trace_metadata=tracking_config["trace"],
    )
    try:
        result = train_lewm(
            trace,
            model_config=recipe.model,
            training_config=config,
            on_validation=tracker.log_validation,
        )
        checkpoint = save_checkpoint(output, result)
        checkpoint_metadata = {
            "trace_sha256": digest,
            "best_validation_step": result.best_validation_step,
            "best_validation_loss": result.best_validation_loss,
            "training_steps": config.max_steps,
        }
        tracker.log_checkpoint(checkpoint, checkpoint_metadata)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    run_id, run_url = tracker.run_id, tracker.url
    tracker.finish(exit_code=0)
    summary = {
        "checkpoint": str(checkpoint),
        "train_loss": result.train_loss,
        "validation_loss": result.validation_loss,
        "best_validation_step": result.best_validation_step,
        "best_validation_loss": result.best_validation_loss,
        "validation_history": [
            {"step": step, "loss": loss} for step, loss in result.validation_history
        ],
        "trace_sha256": result.checkpoint_contract.trace_sha256,
        "train_episodes": list(result.checkpoint_contract.episodes.train),
        "validation_episodes": list(result.checkpoint_contract.episodes.validation),
        "model": asdict(result.model_config),
        "training": asdict(result.training_config),
        "wandb": {
            "project": wandb_project,
            "run_id": run_id,
            "url": run_url,
        },
    }
    return summary


def _training_tracking_config(
    trace: Any, model_config: Any, training_config: Any
) -> dict[str, Any]:
    payload = {
        "schema_version": "autops.wandb-training/v1",
        "trace": {
            "schema_version": trace.metadata.schema_version,
            "trace_sha256": trace_sha256(trace),
            "mission": trace.metadata.mission,
            "episodes": trace.n_episodes,
            "steps_per_episode": trace.n_steps,
            "sources": [source.to_dict() for source in trace.metadata.sources],
        },
        "model": asdict(model_config),
        "training": asdict(training_config),
    }
    payload["provenance"] = collect_provenance(payload, asset_root())
    return payload


def _latent_features(model: Any, observations: np.ndarray, device: str) -> np.ndarray:
    from autops.wm.jepa import require_torch

    torch = require_torch()
    shape = observations.shape
    values = torch.from_numpy(observations.reshape(-1, shape[-1])).to(torch.device(device))
    model.eval()
    with torch.no_grad():
        features = model.encode(values).detach().cpu().numpy()
    return features.reshape(*shape[:-1], features.shape[-1]).astype(np.float32)


@dataclass(frozen=True)
class _PlannerFit:
    checkpoint_contract: Any
    probe: ProbeFit
    ridge: float
    recipe: Any


def _fit_planner_probe(
    trace_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str,
    ridge: float | None,
    seed: int,
) -> _PlannerFit:
    trace = load_trace(trace_path)
    if trace.metadata.mission != "eventsat":
        raise ValueError("the current planner artifact contract targets EventSat")
    if trace.metadata.observation_names != EVENTSAT_OBSERVATIONS:
        raise ValueError("EventSat planner fitting requires canonical observation semantics")
    model, checkpoint_contract = load_checkpoint(checkpoint_path, device=device)
    checkpoint_contract.validate_trace(trace)
    normalizer = checkpoint_contract.normalizer
    features = _latent_features(model, normalizer.normalize_obs(trace.obs), device)
    recipe = load_eventsat_recipe()
    ridge_value = recipe.probes.ridge if ridge is None else ridge
    probe = fit_ridge_probe(
        features,
        build_eventsat_targets(trace),
        attribute_names=DEFAULT_ATTRIBUTES,
        ridge=ridge_value,
        episodes=checkpoint_contract.episodes,
        seed=seed,
    )
    return _PlannerFit(checkpoint_contract, probe, ridge_value, recipe)


def _probe_contract(probe: ProbeFit) -> ProbeContract:
    return ProbeContract(
        W=tuple(tuple(float(value) for value in row) for row in probe.W),
        b=tuple(float(value) for value in probe.b),
        attribute_names=probe.attribute_names,
        target_mean=tuple(float(value) for value in probe.target_mean),
        target_std=tuple(float(value) for value in probe.target_std),
        degenerate=probe.degenerate,
    )


def _probe_evidence(fit: _PlannerFit, checkpoint_path: str | Path) -> ProbeEvidenceContract:
    probe = fit.probe
    return ProbeEvidenceContract(
        attribute_names=probe.attribute_names,
        rmse={name: float(value) for name, value in probe.rmse.items()},
        rmse_over_std={
            name: float(value) if math.isfinite(value) else None
            for name, value in probe.rmse_over_std.items()
        },
        r2={
            name: float(value) if math.isfinite(value) else None for name, value in probe.r2.items()
        },
        train_episodes=probe.train_episodes,
        validation_episodes=probe.validation_episodes,
        ridge=fit.ridge,
        checkpoint_size_bytes=Path(checkpoint_path).stat().st_size,
    )


def _planner_artifact(
    fit: _PlannerFit, checkpoint_path: str | Path, *, seed: int
) -> PlannerArtifact:
    contract = fit.checkpoint_contract
    model_config = contract.model_config
    normalizer = contract.normalizer
    return PlannerArtifact(
        model=ModelContract(
            checkpoint="model.pt",
            mission=contract.mission,
            obs_dim=model_config.obs_dim,
            action_dim=model_config.action_dim,
            embed_dim=model_config.embed_dim,
            history=model_config.history,
            observation_names=contract.observation_names,
            action_names=contract.action_names,
            trace_sha256=contract.trace_sha256,
            checkpoint_sha256=checkpoint_sha256(checkpoint_path),
        ),
        normalization=NormalizationContract(
            obs_mean=tuple(float(value) for value in normalizer.obs_mean),
            obs_std=tuple(float(value) for value in normalizer.obs_std),
            action_mean=tuple(float(value) for value in normalizer.action_mean),
            action_std=tuple(float(value) for value in normalizer.action_std),
        ),
        probe=_probe_contract(fit.probe),
        probe_evidence=_probe_evidence(fit, checkpoint_path),
        cem=replace(fit.recipe.planner.cem, seed=seed),
        mode_weight_presets=fit.recipe.planner.mode_weight_presets,
        normalize_attribute_scale=fit.recipe.planner.normalize_attribute_scale,
    )


def _probe_summary(probe: ProbeFit) -> dict[str, Any]:
    return {
        "rmse": probe.rmse,
        "rmse_over_std": probe.rmse_over_std,
        "r2": probe.r2,
        "degenerate": list(probe.degenerate),
        "train_episodes": list(probe.train_episodes),
        "validation_episodes": list(probe.validation_episodes),
    }


def fit_planner_artifact(
    trace_path: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    ridge: float | None = None,
    seed: int = 3072,
) -> dict[str, Any]:
    fit = _fit_planner_probe(
        trace_path,
        checkpoint_path,
        device=device,
        ridge=ridge,
        seed=seed,
    )
    artifact = _planner_artifact(fit, checkpoint_path, seed=seed)
    destination = save_artifact(output, artifact, checkpoint_source=checkpoint_path)
    return {"artifact": str(destination), "probe": _probe_summary(fit.probe)}


__all__ = [
    "evaluate_lewm_cem",
    "fit_planner_artifact",
    "matrix_coordinates",
    "run_sweep",
    "train_world_model",
]
