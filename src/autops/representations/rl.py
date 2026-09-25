"""The ``rl`` representation: a trained PPO policy decides every step on the OBC.

Ported from the agentic framework's subsymbolic representations. The policy reads
the observation through the mission's RL adapter and its commands pass the same
controller-visible shield as in training, so evaluation replays the training MDP.
The small network runs on the OBC and never charges the Jetson. ``rl_mock`` uses a
seeded uniform policy for CI; its results are diagnostic and never boardable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autops.config import runtime_root
from autops.core.plugin import Representation, register
from autops.core.types import DecisionContext
from autops.rl.policy import (
    RandomPolicy,
    RLlibPolicy,
    checkpoint_identity,
    read_manifest,
    validate_manifest,
)
from autops.rl.spaces import make_space_adapter, rl_spec


class RLPolicy(Representation):
    """A checkpointed RLlib policy behind the common representation seam."""

    mission = ""
    default_satellites: tuple[str, ...] = ()

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.act_ids = list(self.config.get("act_ids", self.default_satellites))
        self.observe_ids = list(self.config.get("observe_ids", self.act_ids))
        self.adapter = make_space_adapter(self.mission, self.act_ids, self.observe_ids)
        self.deterministic = bool(self.config.get("deterministic", True))
        self._decisions = 0
        self._overrides = 0
        if self.config.get("rl_mock", False):
            self._policy: RLlibPolicy | RandomPolicy = RandomPolicy(self.adapter.action_dims)
            self.identity: dict[str, Any] = {"source": "mock"}
            return
        if not self.config.get("checkpoint"):
            raise ValueError("the rl representation requires representation.checkpoint")
        checkpoint = Path(str(self.config["checkpoint"])).expanduser()
        directory, manifest = read_manifest(
            checkpoint if checkpoint.is_absolute() else runtime_root() / checkpoint
        )
        agent_policies = manifest.get("agent_policies", {})
        policy_id = str(
            self.config.get("policy_id")
            or agent_policies.get(str(self.config.get("agent_id", "central_agent")))
            or "shared_policy"
        )
        spec = rl_spec(self.mission)
        validate_manifest(
            manifest,
            spec,
            policy_id,
            spec.obs_dim * len(self.observe_ids),
            self.adapter.action_dims,
        )
        # The identity check verifies the weights against the manifest before loading them.
        self.identity = checkpoint_identity(directory, manifest, policy_id)
        self._policy = RLlibPolicy(directory, policy_id, self.adapter.action_dims)

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self._policy.seed(0 if seed is None else seed)

    def encode_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        return {"vector": self.adapter.encode_observation(observation)}

    def select_action(self, context: DecisionContext) -> dict[str, Any]:
        action, probabilities = self._policy.act(
            context.state["vector"], deterministic=self.deterministic
        )
        decoded = self.adapter.decode_action(action)
        commands = self.adapter.ground_decoded_action(decoded, context.observation)
        self._decisions += 1
        self._overrides += sum(
            decoded[satellite]["mode"] != commands[satellite]["mode"] for satellite in commands
        )
        first = int(np.asarray(action).reshape(-1)[0])
        summary = ", ".join(f"{sat}={command['mode']}" for sat, command in commands.items())
        self._last_rationale = (
            f"{'RLlib PPO' if self.identity['source'] == 'checkpoint' else 'mock policy'}: "
            f"{summary} (first head p={float(probabilities[first]):.2f})"
        )
        return commands

    def diagnostics(self) -> dict[str, Any]:
        return {
            "policy_identity": dict(self.identity),
            "decisions": self._decisions,
            "grounding_overrides": self._overrides,
        }


@register("rl", mission="eventsat", role="onboard")
class EventSatRL(RLPolicy):
    """PPO mode selection over the EventSat onboard information boundary."""

    mission = "eventsat"
    default_satellites = ("eventsat_0",)


@register("rl", mission="ssa", role="onboard")
class SSARL(RLPolicy):
    """PPO modes for one organisation agent's SSA satellites, from its scoped view."""

    mission = "ssa"


__all__ = ["SSARL", "EventSatRL", "RLPolicy"]
