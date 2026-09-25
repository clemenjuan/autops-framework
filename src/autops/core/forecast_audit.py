"""P2 recursive forecast audit under logged command sequences.

At sampled contexts of held-out episodes, the learned model rolls out for
``steps`` under the logged requested commands, and the frozen readouts are
applied to every predicted latent. Stocks are compared with the privileged
target at each step and flows as sums since the context, the quantity a
planner scores. Three references use at most the onboard information: the
readout of the encoded true future record, which isolates the readout's own
error; persistence (stocks hold, flows repeat the last interval); and the
planner's analytical projection of the same commands with present visibility
and persistent sunlight, without mission-policy repair. Per method, attribute
and step, the audit reports RMSE and the skill 1 - MSE/Var across contexts,
overall and split by proximity to contact. Each context row keeps the truth and
every forecast as ``[step][attribute]``, so intervals can resample physical seeds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autops.config import asset_root
from autops.core.offline import (
    context_record,
    evaluation_record,
    finite_values,
    held_out,
    load_planner_bundle,
    near_contact,
    planner_history,
    sample_contexts,
    write_evidence,
)
from autops.core.provenance import collect_provenance
from autops.core.replay import replay_environments
from autops.missions.eventsat.observation import encode_vectors, onboard_view
from autops.wm.artifact import (
    PlannerArtifact,
    artifact_sha256,
)
from autops.wm.guidance import project_command_prefixes
from autops.wm.probes import DEFAULT_ATTRIBUTES, FLOW_ATTRIBUTES, build_eventsat_targets
from autops.wm.schema import TraceDataset
from autops.wm.scoring import (
    analytical_candidate_attributes,
    latent_rollout_readouts,
)

FORECAST_SCHEMA_VERSION = "autops.forecast-audit/v2"


def _accumulate(values: np.ndarray, flows: list[int]) -> np.ndarray:
    """Turn per-step flow values ``[context, step, attribute]`` into sums since the context."""

    result = np.array(values, dtype=np.float64)
    result[..., flows] = np.cumsum(result[..., flows], axis=1)
    return result


def _encoded_readouts(
    model: Any, artifact: PlannerArtifact, observations: np.ndarray, device: str
) -> np.ndarray:
    from autops.wm.jepa import require_torch

    torch = require_torch()
    normalizer = artifact.normalization
    normalized = (observations - np.asarray(normalizer.obs_mean)) / np.asarray(normalizer.obs_std)
    with torch.no_grad():
        latent = model.encode(torch.as_tensor(normalized.astype(np.float32), device=device))
    matrix = np.asarray(artifact.probe.W, dtype=np.float32)
    return latent.cpu().numpy() @ matrix.T + np.asarray(artifact.probe.b, dtype=np.float32)


def _analytical(
    trace: TraceDataset,
    contexts: list[tuple[int, int, bool]],
    steps: int,
    names: tuple[str, ...],
) -> np.ndarray:
    forecasts = {}
    for episode in sorted({episode for episode, _, _ in contexts}):
        wanted = [step for item, step, _ in contexts if item == episode]
        for step, env in replay_environments(trace, episode, wanted).items():
            state = onboard_view(encode_vectors(env.observe())[2])
            commands = trace.mode[episode, step : step + steps]
            projection = project_command_prefixes(state, commands)
            forecasts[episode, step] = analytical_candidate_attributes(state, projection, names)
    return np.stack([forecasts[episode, step] for episode, step, _ in contexts])


def _scores(predicted: np.ndarray, truth: np.ndarray) -> dict[str, list[float | None]]:
    error = predicted - truth
    mse = np.mean(error**2, axis=0)
    variance = np.var(truth, axis=0)
    skill = np.where(variance > 1e-12, 1.0 - mse / np.maximum(variance, 1e-12), np.nan)
    return {
        "rmse": np.sqrt(mse).tolist(),
        "skill": [None if np.isnan(value) else float(value) for value in skill],
    }


def _metrics(
    forecasts: dict[str, np.ndarray], truth: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    return {
        method: {
            name: _scores(values[..., index], truth[..., index]) for index, name in enumerate(names)
        }
        for method, values in forecasts.items()
    }


def _forecasts(
    model: Any,
    artifact: PlannerArtifact,
    trace: TraceDataset,
    contexts: list[tuple[int, int, bool]],
    steps: int,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return every method's forecast and the truth, as ``[context, step, attribute]``."""

    names = artifact.probe.attribute_names
    flows = [index for index, name in enumerate(names) if name in FLOW_ATTRIBUTES]
    episodes = np.asarray([episode for episode, _, _ in contexts])[:, None]
    offsets = np.asarray([step for _, step, _ in contexts])[:, None] + np.arange(steps + 1)
    columns = [DEFAULT_ATTRIBUTES.index(name) for name in names]
    targets = build_eventsat_targets(trace)[..., columns][episodes, offsets]
    histories = [
        planner_history(trace, episode, step, artifact.model.history)
        for episode, step, _ in contexts
    ]
    learned = latent_rollout_readouts(
        model,
        artifact,
        np.stack([history["obs"] for history in histories]),
        np.stack([history["action"] for history in histories]),
        trace.mode[episodes, offsets[:, :-1]],
        device=device,
    )
    encoded = _encoded_readouts(model, artifact, trace.obs[episodes, offsets[:, 1:]], device)
    forecasts = {
        "lewm-rollout": _accumulate(learned, flows),
        "encoded-truth-readout": _accumulate(encoded, flows),
        "persistence": _accumulate(np.repeat(targets[:, :1], steps, axis=1), flows),
        "analytical-projection": _analytical(trace, contexts, steps, names),
    }
    return forecasts, _accumulate(targets[:, 1:], flows)


def audit_recursive_forecasts(
    trace_path: str | Path,
    artifact_path: str | Path,
    *,
    test_trace_path: str | Path | None = None,
    output: str | Path | None = None,
    contexts: int = 256,
    steps: int = 48,
    near_contact_fraction: float = 0.5,
    device: str = "cpu",
    seed: int = 3072,
) -> dict[str, Any]:
    trace, artifact, model, contract = load_planner_bundle(trace_path, artifact_path, device)
    evaluation, episodes = held_out(trace, test_trace_path, contract)
    if steps < 1 or steps >= evaluation.n_steps:
        raise ValueError("forecast steps must be positive and shorter than an episode")
    near = near_contact(evaluation, artifact.cem.horizon)
    sampled = sample_contexts(
        near,
        episodes,
        contexts,
        near_contact_fraction,
        np.random.default_rng(seed),
        last_step=evaluation.n_steps - steps - 1,
    )
    forecasts, truth = _forecasts(model, artifact, evaluation, sampled, steps, device)
    names = artifact.probe.attribute_names
    flags = np.asarray([flag for _, _, flag in sampled])
    metrics = {"all": _metrics(forecasts, truth, names)}
    for label, mask in (("near_contact", flags), ("away_from_contact", ~flags)):
        if mask.sum() > 1:
            subset = {method: values[mask] for method, values in forecasts.items()}
            metrics[label] = _metrics(subset, truth[mask], names)
    settings = {
        "contexts": contexts,
        "steps": steps,
        "near_contact_fraction": near_contact_fraction,
        "seed": seed,
    }
    return write_evidence(
        output,
        {
            "schema_version": FORECAST_SCHEMA_VERSION,
            "trace_sha256": artifact.model.trace_sha256,
            "artifact_sha256": artifact_sha256(artifact),
            "checkpoint_sha256": artifact.model.checkpoint_sha256,
            "evaluation": evaluation_record(None if test_trace_path is None else evaluation),
            "attributes": {name: "flow" if name in FLOW_ATTRIBUTES else "stock" for name in names},
            "config": settings,
            "provenance": collect_provenance(settings, asset_root()),
            "context_count": {"all": len(sampled), "near_contact": int(flags.sum())},
            "metrics": metrics,
            "contexts": [
                {
                    **context_record(evaluation, context),
                    "truth": finite_values(truth[index]),
                    "forecasts": {
                        method: finite_values(values[index]) for method, values in forecasts.items()
                    },
                }
                for index, context in enumerate(sampled)
            ],
        },
    )


__all__ = ["FORECAST_SCHEMA_VERSION", "audit_recursive_forecasts"]
