"""Trained-policy loading, checkpoint manifests, and the CI mock policy.

Evaluation restores only the trained RLlib policy (``Policy.from_checkpoint``)
rather than the whole algorithm, so no Ray workers, training GPUs, or rollout
runners start, and an Orekit JVM can already be running. A manifest written next
to every checkpoint declares the observation contract; a checkpoint trained on any
other contract is rejected before loading.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from autops.rl.spaces import RLSpec

MANIFEST_SCHEMA_VERSION = "autops.rl.checkpoint/v1"
MANIFEST_NAME = "manifest.json"
_LOADED: dict[tuple[str, str], Any] = {}


def read_manifest(checkpoint: str | Path) -> tuple[Path, dict[str, Any]]:
    """Return the checkpoint directory and its validated-schema manifest."""

    directory = Path(checkpoint).expanduser().resolve()
    path = directory / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"RL checkpoint manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"unsupported RL checkpoint manifest {manifest.get('schema_version')!r}")
    return directory, manifest


def validate_manifest(
    manifest: Mapping[str, Any], spec: RLSpec, policy_id: str, obs_dim: int, action_dims: list[int]
) -> None:
    """Reject a checkpoint trained on another observation or action contract."""

    if (
        manifest.get("observation_schema_id") != spec.schema_id
        or tuple(manifest.get("observation_names", ())) != spec.observation_names
    ):
        raise ValueError(
            f"RL checkpoint observation contract {manifest.get('observation_schema_id')!r} "
            f"differs from {spec.schema_id!r}; retrain with the current encoder"
        )
    shape = manifest.get("policy_observation_shapes", {}).get(policy_id)
    nvec = manifest.get("policy_action_nvec", {}).get(policy_id)
    if list(shape or []) != [obs_dim] or list(nvec or []) != list(action_dims):
        raise ValueError(
            f"RL checkpoint policy {policy_id!r} has shape {shape} and actions {nvec}; "
            f"expected [{obs_dim}] and {action_dims}"
        )


def checkpoint_identity(directory: Path, manifest: Mapping[str, Any], policy_id: str) -> dict:
    """Public-safe identity of the deployed policy for result provenance."""

    return {
        "source": "checkpoint",
        "manifest_sha256": hashlib.sha256((directory / MANIFEST_NAME).read_bytes()).hexdigest(),
        "observation_schema_id": manifest["observation_schema_id"],
        "sampled_steps": manifest.get("sampled_steps"),
        "policy_id": policy_id,
        "model_architecture": manifest.get("model_architecture"),
    }


def merge_identities(identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One run-level identity: agents may use several policies of a single checkpoint."""

    shared = [
        {key: value for key, value in item.items() if key != "policy_id"} for item in identities
    ]
    if not shared or any(item != shared[0] for item in shared[1:]):
        raise ValueError("organisation agents deployed different RL checkpoints")
    policy_ids = sorted({str(item["policy_id"]) for item in identities if "policy_id" in item})
    return {**shared[0], **({"policy_ids": policy_ids} if policy_ids else {})}


class RLlibPolicy:
    """One trained RLlib policy with deterministic or privately seeded sampling."""

    def __init__(self, directory: Path, policy_id: str, action_dims: Sequence[int]) -> None:
        key = (str(directory), policy_id)
        if key not in _LOADED:
            from ray.rllib.policy.policy import Policy

            from autops.rl.models import register_models

            register_models()
            _LOADED[key] = Policy.from_checkpoint(str(directory / "policies" / policy_id))
        self._policy = _LOADED[key]
        self._action_dims = [int(dim) for dim in action_dims]
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def act(self, obs: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return the action and the first head's probabilities."""

        action, _, extra = self._policy.compute_single_action(
            np.asarray(obs, np.float32), explore=False
        )
        logits = np.asarray(extra.get("action_dist_inputs", []), np.float64)
        heads = _split_softmax(logits, self._action_dims)
        if deterministic or heads is None:
            return np.asarray(action, int).reshape(-1), _first(heads, self._action_dims)
        sampled = [int(self._rng.choice(len(probs), p=probs)) for probs in heads]
        return np.asarray(sampled, int), heads[0]


class RandomPolicy:
    """Uniform, seeded mode sampling for CI; never a benchmark policy."""

    def __init__(self, action_dims: Sequence[int]) -> None:
        self._action_dims = [int(dim) for dim in action_dims]
        self._rng = np.random.default_rng()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def act(self, obs: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, np.ndarray]:
        del obs, deterministic
        action = np.asarray([self._rng.integers(0, dim) for dim in self._action_dims], int)
        return action, np.full(self._action_dims[0], 1.0 / self._action_dims[0])


def _split_softmax(logits: np.ndarray, dims: Sequence[int]) -> list[np.ndarray] | None:
    if logits.size < sum(dims) or not np.all(np.isfinite(logits[: sum(dims)])):
        return None
    heads, offset = [], 0
    for dim in dims:
        head = np.exp(logits[offset : offset + dim] - logits[offset : offset + dim].max())
        heads.append(head / head.sum())
        offset += dim
    return heads


def _first(heads: list[np.ndarray] | None, dims: Sequence[int]) -> np.ndarray:
    return heads[0] if heads is not None else np.full(dims[0], 1.0 / dims[0])


__all__ = [
    "MANIFEST_NAME",
    "MANIFEST_SCHEMA_VERSION",
    "RLlibPolicy",
    "RandomPolicy",
    "checkpoint_identity",
    "merge_identities",
    "read_manifest",
    "validate_manifest",
]
