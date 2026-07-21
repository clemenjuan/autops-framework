"""Strict evidence validation for result documents consumed by the board."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autops.board.model import LeWMEvidence, LeWMTreatment
from autops.core.provenance import scientific_config_sha256
from autops.missions.eventsat.metrics import experiment_statistics

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40,64}")


@dataclass(frozen=True)
class _LeWMEpisode:
    treatment: LeWMTreatment
    semantic_identity: tuple[Any, ...]
    planner_compute_energy_wh: float
    planning_events: int
    cem_latency_total_s: float
    evaluated_rollouts: int
    checkpoint_size_bytes: int
    probe_error_mean: float


@dataclass(frozen=True)
class ValidatedResult:
    experiment: dict[str, Any]
    metrics: dict[str, float]
    metric_names: dict[str, str]
    episodes: int
    steps: int
    config_sha256: str
    lewm: LeWMEvidence | None


def _finite_mapping(value: Any, source: Path, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{source}: {label} must be a non-empty mapping")
    result: dict[str, float] = {}
    for name, raw in value.items():
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{source}: {label} {name!r} is not numeric")
        numeric = float(raw)
        if not math.isfinite(numeric):
            raise ValueError(f"{source}: {label} {name!r} is not finite")
        result[str(name)] = numeric
    return result


def _is_lewm(experiment: dict[str, Any]) -> bool:
    paradigm = experiment.get("paradigm")
    if paradigm == "ao":
        return experiment.get("representation") == "lewm-cem"
    if paradigm == "ah":
        return experiment.get("onboard_representation") == "lewm-cem"
    return False


def _finite_number(
    value: Any,
    source: Path,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}: {label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{source}: {label} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{source}: {label} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{source}: {label} must be at most {maximum}")
    return result


def _integer(value: Any, source: Path, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{source}: {label} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{source}: {label} must be {qualifier}")
    return value


def _sha256(value: Any, source: Path, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{source}: {label} must be a lowercase SHA-256")
    return value


def _reject_nonfinite(value: Any, source: Path, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{source}: {label} contains a non-finite value")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, source, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, source, f"{label}[{index}]")


def _expected_mission_mode(experiment: dict[str, Any], source: Path) -> str:
    config = experiment.get("representation_config", {})
    if not isinstance(config, dict):
        raise ValueError(f"{source}: LeWM representation_config must be a mapping")
    custom_weights = config.get("mission_weights", config.get("mode_weights"))
    if custom_weights is not None:
        return "custom"
    mode = config.get("mission_mode", "science")
    if not isinstance(mode, str) or not mode:
        raise ValueError(f"{source}: LeWM mission_mode must be non-empty")
    return mode


def _onboard_compute_w(experiment: dict[str, Any], source: Path) -> float:
    mission = experiment.get("mission_config")
    if not isinstance(mission, dict):
        raise ValueError(f"{source}: LeWM result lacks mission configuration")
    power = mission.get("power")
    if not isinstance(power, dict) or "onboard_compute_w" not in power:
        raise ValueError(f"{source}: LeWM result lacks onboard_compute_w")
    return _finite_number(
        power["onboard_compute_w"],
        source,
        "mission_config.power.onboard_compute_w",
        minimum=0.0,
    )


def _probe_errors(value: Any, source: Path) -> tuple[tuple[str, float | None], ...]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{source}: LeWM probe_rmse_over_std must be a non-empty mapping")
    errors: list[tuple[str, float | None]] = []
    for raw_name, raw_value in value.items():
        name = str(raw_name)
        if raw_value is None:
            errors.append((name, None))
            continue
        error = _finite_number(
            raw_value,
            source,
            f"decision_diagnostics.onboard.probe_rmse_over_std.{name}",
            minimum=0.0,
        )
        errors.append((name, error))
    if not any(value is not None for _, value in errors):
        raise ValueError(f"{source}: LeWM normalized probe evidence has no finite attributes")
    return tuple(sorted(errors))


def _identity_from_experiment(experiment: dict[str, Any], source: Path) -> tuple[str, ...]:
    identity = experiment.get("planner_artifact_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"{source}: LeWM result lacks planner_artifact_identity")
    schema = identity.get("schema_version")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{source}: LeWM planner artifact schema identity is missing")
    return (
        schema,
        _sha256(identity.get("artifact_sha256"), source, "planner artifact SHA-256"),
        _sha256(identity.get("trace_sha256"), source, "planner trace SHA-256"),
        _sha256(identity.get("checkpoint_sha256"), source, "planner checkpoint SHA-256"),
    )


def _lewm_treatment(
    diagnostics: dict[str, Any],
    experiment: dict[str, Any],
    source: Path,
    expected_identity: tuple[str, ...],
) -> tuple[LeWMTreatment, str, str]:
    mission_mode = diagnostics.get("mission_mode")
    if not isinstance(mission_mode, str) or not mission_mode:
        raise ValueError(f"{source}: LeWM onboard diagnostics lacks mission_mode")
    if mission_mode != _expected_mission_mode(experiment, source):
        raise ValueError(f"{source}: LeWM mission_mode disagrees with effective configuration")
    values = {
        name: _integer(diagnostics.get(name), source, f"LeWM {name}", positive=True)
        for name in ("plan_hold", "horizon", "samples", "elites", "iterations")
    }
    if values["plan_hold"] > values["horizon"]:
        raise ValueError(f"{source}: LeWM plan_hold exceeds horizon")
    if values["elites"] > values["samples"]:
        raise ValueError(f"{source}: LeWM elites exceeds samples")

    artifact = diagnostics.get("artifact_identity")
    checkpoint = diagnostics.get("checkpoint_identity")
    if not isinstance(artifact, dict) or not isinstance(checkpoint, dict):
        raise ValueError(f"{source}: LeWM onboard diagnostics lacks planner identities")
    schema = artifact.get("schema_version")
    relative_path = checkpoint.get("relative_path")
    if not isinstance(schema, str) or not schema:
        raise ValueError(f"{source}: LeWM artifact schema identity is missing")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"{source}: LeWM checkpoint relative path is missing")
    identity = (
        schema,
        _sha256(artifact.get("sha256"), source, "diagnostic artifact SHA-256"),
        _sha256(artifact.get("trace_sha256"), source, "diagnostic trace SHA-256"),
        _sha256(checkpoint.get("sha256"), source, "diagnostic checkpoint SHA-256"),
    )
    if identity != expected_identity:
        raise ValueError(f"{source}: LeWM diagnostic identity disagrees with experiment")
    return (
        LeWMTreatment(
            mission_mode=mission_mode,
            onboard_compute_w=_onboard_compute_w(experiment, source),
            artifact_sha256=identity[1],
            trace_sha256=identity[2],
            checkpoint_sha256=identity[3],
            **values,
        ),
        schema,
        relative_path,
    )


def _lewm_compute(
    diagnostics: dict[str, Any],
    episode: dict[str, Any],
    treatment: LeWMTreatment,
    source: Path,
) -> tuple[int, float, int]:
    planning_events = _integer(diagnostics.get("planning_events"), source, "LeWM planning_events")
    held_steps = _integer(diagnostics.get("held_action_steps"), source, "LeWM held_action_steps")
    reflexes = _integer(diagnostics.get("reflex_overrides"), source, "LeWM reflex_overrides")
    evaluated = _integer(diagnostics.get("evaluated_rollouts"), source, "LeWM evaluated_rollouts")
    steps = _integer(episode.get("steps"), source, "LeWM episode steps", positive=True)
    if planning_events + held_steps + reflexes != steps:
        raise ValueError(f"{source}: LeWM decision counters do not cover the episode")
    if evaluated != planning_events * treatment.samples * treatment.iterations:
        raise ValueError(f"{source}: LeWM evaluated_rollouts disagrees with CEM treatment")

    latency_total = _finite_number(
        diagnostics.get("cem_latency_total_s"), source, "LeWM cem_latency_total_s", minimum=0.0
    )
    latency_mean = _finite_number(
        diagnostics.get("cem_latency_mean_s"), source, "LeWM cem_latency_mean_s", minimum=0.0
    )
    expected_mean = latency_total / planning_events if planning_events else 0.0
    if not math.isclose(latency_mean, expected_mean, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{source}: LeWM mean CEM latency is internally inconsistent")
    throughput = _finite_number(
        diagnostics.get("rollouts_per_second"),
        source,
        "LeWM rollouts_per_second",
        minimum=0.0,
    )
    expected_throughput = evaluated / latency_total if latency_total > 0.0 else 0.0
    if not math.isclose(throughput, expected_throughput, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{source}: LeWM rollout throughput is internally inconsistent")
    return planning_events, latency_total, evaluated


def _lewm_probe(
    diagnostics: dict[str, Any], source: Path
) -> tuple[int, float, tuple[tuple[str, float | None], ...]]:
    checkpoint_size = _integer(
        diagnostics.get("checkpoint_size_bytes"),
        source,
        "LeWM checkpoint_size_bytes",
        positive=True,
    )
    errors = _probe_errors(diagnostics.get("probe_rmse_over_std"), source)
    probe_mean = _finite_number(
        diagnostics.get("probe_rmse_over_std_mean"),
        source,
        "LeWM probe_rmse_over_std_mean",
        minimum=0.0,
    )
    finite_errors = [value for _, value in errors if value is not None]
    if not math.isclose(
        probe_mean, sum(finite_errors) / len(finite_errors), rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError(f"{source}: LeWM normalized probe mean is internally inconsistent")
    return checkpoint_size, probe_mean, errors


def _validate_lewm_episode(
    episode: dict[str, Any],
    experiment: dict[str, Any],
    source: Path,
    *,
    expected_identity: tuple[str, ...],
) -> _LeWMEpisode:
    decision = episode.get("decision_diagnostics")
    diagnostics = decision.get("onboard") if isinstance(decision, dict) else None
    if not isinstance(diagnostics, dict):
        episode_id = episode.get("episode_id")
        raise ValueError(f"{source}: LeWM episode {episode_id} lacks onboard diagnostics")
    _reject_nonfinite(diagnostics, source, "decision_diagnostics.onboard")
    treatment, schema, relative_path = _lewm_treatment(
        diagnostics, experiment, source, expected_identity
    )
    planning_events, latency_total, evaluated = _lewm_compute(
        diagnostics, episode, treatment, source
    )
    checkpoint_size, probe_mean, probe_errors = _lewm_probe(diagnostics, source)
    planner_energy = _finite_number(
        episode.get("planner_compute_energy_wh"),
        source,
        "LeWM planner_compute_energy_wh",
        minimum=0.0,
    )
    return _LeWMEpisode(
        treatment=treatment,
        semantic_identity=(
            treatment,
            schema,
            relative_path,
            checkpoint_size,
            probe_mean,
            probe_errors,
        ),
        planner_compute_energy_wh=planner_energy,
        planning_events=planning_events,
        cem_latency_total_s=latency_total,
        evaluated_rollouts=evaluated,
        checkpoint_size_bytes=checkpoint_size,
        probe_error_mean=probe_mean,
    )


def _validate_lewm_evidence(
    experiment: dict[str, Any],
    episodes: list[dict[str, Any]],
    episode_metrics: list[dict[str, float]],
    means: dict[str, float],
    deviations: dict[str, float],
    expected_statistics: dict[str, dict[str, float]],
    source: Path,
) -> LeWMEvidence | None:
    if not _is_lewm(experiment):
        return None
    for name in ("final_battery_soc", "total_energy_consumed_wh"):
        if name not in means or name not in deviations:
            raise ValueError(f"{source}: LeWM statistics omit {name}")
        if any(name not in metrics for metrics in episode_metrics):
            raise ValueError(f"{source}: LeWM episode metrics omit {name}")
        if not math.isclose(
            means[name],
            expected_statistics["mean"][name],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{source}: LeWM statistics.mean {name} disagrees with episodes")
        if not math.isclose(
            deviations[name],
            expected_statistics["std"][name],
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{source}: LeWM statistics.std {name} disagrees with episodes")
    if any(not 0.0 <= metrics["final_battery_soc"] <= 1.0 for metrics in episode_metrics):
        raise ValueError(f"{source}: LeWM final battery SOC is outside [0, 1]")
    if any(metrics["total_energy_consumed_wh"] < 0.0 for metrics in episode_metrics):
        raise ValueError(f"{source}: LeWM total energy must be non-negative")

    expected_identity = _identity_from_experiment(experiment, source)
    evidence = [
        _validate_lewm_episode(
            episode,
            experiment,
            source,
            expected_identity=expected_identity,
        )
        for episode in episodes
    ]
    first = evidence[0]
    if any(item.semantic_identity != first.semantic_identity for item in evidence[1:]):
        raise ValueError(f"{source}: LeWM episodes have inconsistent semantic identities")

    total_events = sum(item.planning_events for item in evidence)
    total_latency = sum(item.cem_latency_total_s for item in evidence)
    total_rollouts = sum(item.evaluated_rollouts for item in evidence)
    total_steps = sum(int(episode["steps"]) for episode in episodes)
    count = len(evidence)
    return LeWMEvidence(
        treatment=first.treatment,
        mean_final_battery_soc=means["final_battery_soc"],
        mean_total_energy_consumed_wh=means["total_energy_consumed_wh"],
        mean_planner_compute_energy_wh=(
            sum(item.planner_compute_energy_wh for item in evidence) / count
        ),
        planning_duty_cycle=total_events / total_steps,
        mean_planning_events=total_events / count,
        mean_cem_latency_s=total_latency / total_events if total_events else 0.0,
        rollouts_per_second=total_rollouts / total_latency if total_latency > 0.0 else 0.0,
        model_checkpoint_mb=first.checkpoint_size_bytes / 1_000_000.0,
        mean_normalized_probe_error=first.probe_error_mean,
    )


def _validate_provenance(provenance: Any, experiment: dict[str, Any], source: Path) -> None:
    if not isinstance(provenance, dict):
        raise ValueError(f"{source}: result lacks reproducibility provenance")
    digest = provenance.get("config_sha256")
    expected = scientific_config_sha256(experiment)
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest) or digest != expected:
        raise ValueError(f"{source}: configuration provenance does not match experiment")
    revision = provenance.get("source_revision") or provenance.get("git_commit")
    if not isinstance(revision, str) or not _GIT_COMMIT.fullmatch(revision):
        raise ValueError(f"{source}: result lacks a source revision")
    if provenance.get("git_dirty") is not False:
        raise ValueError(f"{source}: board input must come from a clean source revision")
    if not provenance.get("python") or not isinstance(provenance.get("dependencies"), dict):
        raise ValueError(f"{source}: result lacks runtime provenance")


def validate_result_document(payload: Any, source: Path) -> ValidatedResult:
    """Reject results whose displayed values cannot be traced to run evidence."""

    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"{source}: unsupported or missing result schema")
    experiment = payload.get("experiment")
    episodes = payload.get("episodes")
    statistics = payload.get("statistics")
    registry = payload.get("metric_registry")
    if not isinstance(experiment, dict) or not isinstance(episodes, list):
        raise ValueError(f"{source}: result lacks experiment or episode evidence")
    if not isinstance(statistics, dict) or not isinstance(registry, dict) or not registry:
        raise ValueError(f"{source}: result lacks statistics or metric registry")
    expected_n = int(experiment.get("episodes", 0))
    expected_steps = int(experiment.get("steps", 0))
    if expected_n < 1 or len(episodes) != expected_n:
        raise ValueError(f"{source}: result is incomplete ({len(episodes)}/{expected_n} episodes)")
    if expected_steps < 1 or any(
        not isinstance(item, dict) or int(item.get("steps", -1)) != expected_steps
        for item in episodes
    ):
        raise ValueError(f"{source}: result contains an incomplete episode")
    if sorted(int(item.get("episode_id", -1)) for item in episodes) != list(range(expected_n)):
        raise ValueError(f"{source}: episode identities are incomplete or duplicated")

    metrics = _finite_mapping(payload.get("metrics"), source, "metric")
    means = _finite_mapping(statistics.get("mean"), source, "statistics.mean")
    deviations = _finite_mapping(statistics.get("std"), source, "statistics.std")
    if set(metrics) != set(registry):
        raise ValueError(f"{source}: displayed metrics do not match the metric registry")
    names = {str(key): str(value) for key, value in registry.items()}
    for metric_id, name in names.items():
        if name not in means or not math.isclose(metrics[metric_id], means[name], rel_tol=1e-9):
            raise ValueError(f"{source}: displayed metric {metric_id!r} disagrees with statistics")
    episode_evidence: list[dict[str, float]] = []
    for episode in episodes:
        episode_metrics = _finite_mapping(episode.get("metrics"), source, "episode metric")
        if not set(names.values()) <= set(episode_metrics):
            raise ValueError(f"{source}: episode evidence omits a registered metric")
        episode_evidence.append(episode_metrics)
    expected_statistics = experiment_statistics(episode_evidence)
    for name in names.values():
        expected_mean = expected_statistics["mean"][name]
        expected_std = expected_statistics["std"][name]
        if not math.isclose(means[name], expected_mean, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError(f"{source}: statistics.mean disagrees with episode evidence")
        if name not in deviations or not math.isclose(
            deviations[name], expected_std, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(f"{source}: statistics.std disagrees with episode evidence")

    lewm = _validate_lewm_evidence(
        experiment,
        episodes,
        episode_evidence,
        means,
        deviations,
        expected_statistics,
        source,
    )
    _validate_provenance(payload.get("provenance"), experiment, source)
    return ValidatedResult(
        experiment=experiment,
        metrics=metrics,
        metric_names=names,
        episodes=expected_n,
        steps=expected_steps,
        config_sha256=str(payload["provenance"]["config_sha256"]),
        lewm=lewm,
    )


__all__ = ["LeWMEvidence", "LeWMTreatment", "ValidatedResult", "validate_result_document"]
