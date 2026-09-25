"""P4 action-conditioning audit: counterfactual commands in the model and the simulator.

At sampled contexts of held-out episodes, one set of requested command
sequences is rolled out by the learned model from the logged history and by the
simulator from the replayed state: each of the seven modes held for the whole
horizon, plus random sequences. Three properties are measured. Contact
opportunity is exogenous, so its readout must not change with the commands; the
spread of the prediction across sequences is reported beside the simulator's,
which is zero. The response to commands (each sequence minus holding charging)
is compared with the simulator's response per attribute and step. Steps where
the simulator dropped or overrode the request, while settling or for safety,
are scored apart, so predicting the requested rather than the executed outcome
shows up. Each context row keeps the model and simulator values as
``[sequence][step][attribute]`` and the altered-step flags, so intervals can
resample physical seeds.
"""

from __future__ import annotations

from copy import deepcopy
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
from autops.missions.eventsat.env import EventSatEnvironment
from autops.missions.eventsat.observation import encode_vectors
from autops.wm.probes import DEFAULT_ATTRIBUTES, eventsat_targets
from autops.wm.schema import EVENTSAT_ACTIONS, TraceDataset
from autops.wm.scoring import latent_rollout_readouts

COUNTERFACTUAL_SCHEMA_VERSION = "autops.counterfactual-audit/v2"
_CHARGING = EVENTSAT_ACTIONS.index("charging")


def _simulate(
    env: EventSatEnvironment, sequences: np.ndarray, columns: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return simulator targets ``[sequence, step, attribute]`` and altered-command flags."""

    targets, altered = [], []
    for sequence in sequences:
        simulation = deepcopy(env)
        states = [encode_vectors(simulation.observe())[1]]
        forced, changed = [], []
        for action in sequence:
            info = simulation.step({"eventsat_0": {"mode": EVENTSAT_ACTIONS[action]}}).info
            forced.append(bool(info["forced"]))
            changed.append(bool(info["forced"] or info["command_ignored"]))
            states.append(encode_vectors(simulation.observe())[1])
        rows = eventsat_targets(np.stack(states), np.asarray([*forced, False]))
        targets.append(rows[1:, columns])
        altered.append(changed)
    return np.stack(targets), np.asarray(altered)


def _response_scores(model: np.ndarray, truth: np.ndarray) -> dict[str, list[float | None]]:
    """Compare responses ``[context, sequence, step]`` against holding charging."""

    rmse, correlation = [], []
    for step in range(model.shape[-1]):
        predicted, actual = model[..., step].ravel(), truth[..., step].ravel()
        rmse.append(float(np.sqrt(np.mean((predicted - actual) ** 2))))
        spread = predicted.std() * actual.std()
        correlation.append(float(np.corrcoef(predicted, actual)[0, 1]) if spread > 1e-12 else None)
    return {
        "rmse": rmse,
        "correlation": correlation,
        "truth_std": truth.reshape(-1, truth.shape[-1]).std(axis=0).tolist(),
    }


def _metrics(
    model: np.ndarray, truth: np.ndarray, altered: np.ndarray, names: tuple[str, ...]
) -> dict[str, Any]:
    """Score model and simulator values ``[context, sequence, step, attribute]``."""

    others = [index for index in range(model.shape[1]) if index != _CHARGING]
    error = np.abs(model - truth)

    def mean(values: np.ndarray) -> float | None:
        return float(values.mean()) if values.size else None

    exogenous = None
    if "communication_opportunity" in names:
        index = names.index("communication_opportunity")
        exogenous = {
            "model": float(model[..., index].std(axis=1).mean()),
            "simulator": float(truth[..., index].std(axis=1).mean()),
        }
    return {
        "exogenous_spread": exogenous,
        "response": {
            name: _response_scores(
                model[:, others, :, index] - model[:, [_CHARGING], :, index],
                truth[:, others, :, index] - truth[:, [_CHARGING], :, index],
            )
            for index, name in enumerate(names)
        },
        "absolute_error": {
            name: {
                "executed_as_requested": mean(error[..., index][~altered]),
                "dropped_or_overridden": mean(error[..., index][altered]),
            }
            for index, name in enumerate(names)
        },
        "altered_step_fraction": float(altered.mean()),
    }


def _counterfactuals(
    model: Any,
    artifact: Any,
    trace: TraceDataset,
    contexts: list[tuple[int, int, bool]],
    sequences: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = [DEFAULT_ATTRIBUTES.index(name) for name in artifact.probe.attribute_names]
    predicted, simulated, altered = [], [], []
    for episode in sorted({episode for episode, _, _ in contexts}):
        steps = [step for item, step, _ in contexts if item == episode]
        for step, env in replay_environments(trace, episode, steps).items():
            history = planner_history(trace, episode, step, artifact.model.history)
            count = len(sequences)
            predicted.append(
                latent_rollout_readouts(
                    model,
                    artifact,
                    np.repeat(history["obs"][None], count, axis=0),
                    np.repeat(history["action"][None], count, axis=0),
                    sequences,
                    device=device,
                )
            )
            truth, changed = _simulate(env, sequences, columns)
            simulated.append(truth)
            altered.append(changed)
    return np.stack(predicted), np.stack(simulated), np.stack(altered)


def audit_action_conditioning(
    trace_path: str | Path,
    artifact_path: str | Path,
    *,
    test_trace_path: str | Path | None = None,
    output: str | Path | None = None,
    contexts: int = 32,
    steps: int = 12,
    random_sequences: int = 3,
    near_contact_fraction: float = 0.5,
    device: str = "cpu",
    seed: int = 3072,
) -> dict[str, Any]:
    trace, artifact, model, contract = load_planner_bundle(trace_path, artifact_path, device)
    evaluation, episodes = held_out(trace, test_trace_path, contract)
    if steps < 1 or steps >= evaluation.n_steps or random_sequences < 0:
        raise ValueError("steps must fit inside an episode and random_sequences be non-negative")
    rng = np.random.default_rng(seed)
    sampled = sample_contexts(
        near_contact(evaluation, artifact.cem.horizon),
        episodes,
        contexts,
        near_contact_fraction,
        rng,
        last_step=evaluation.n_steps - steps - 1,
    )
    held = np.repeat(np.arange(len(EVENTSAT_ACTIONS))[:, None], steps, axis=1)
    random = rng.integers(0, len(EVENTSAT_ACTIONS), size=(random_sequences, steps))
    sequences = np.concatenate([held, random])
    model_values, truth, altered = _counterfactuals(
        model, artifact, evaluation, sampled, sequences, device
    )
    settings = {
        "contexts": contexts,
        "steps": steps,
        "random_sequences": random_sequences,
        "near_contact_fraction": near_contact_fraction,
        "seed": seed,
    }
    return write_evidence(
        output,
        {
            "schema_version": COUNTERFACTUAL_SCHEMA_VERSION,
            "trace_sha256": artifact.model.trace_sha256,
            "checkpoint_sha256": artifact.model.checkpoint_sha256,
            "evaluation": evaluation_record(None if test_trace_path is None else evaluation),
            "sequences": [[EVENTSAT_ACTIONS[action] for action in row] for row in sequences],
            "config": settings,
            "provenance": collect_provenance(settings, asset_root()),
            "context_count": len(sampled),
            "metrics": _metrics(model_values, truth, altered, artifact.probe.attribute_names),
            "attributes": list(artifact.probe.attribute_names),
            "contexts": [
                {
                    **context_record(evaluation, context),
                    "model": finite_values(model_values[index]),
                    "simulator": finite_values(truth[index]),
                    "altered": altered[index].tolist(),
                }
                for index, context in enumerate(sampled)
            ],
        },
    )


__all__ = ["COUNTERFACTUAL_SCHEMA_VERSION", "audit_action_conditioning"]
