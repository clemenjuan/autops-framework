"""RLlib actor-critic of the agentic framework's EventSat PPO.

A shared 256x256 tanh trunk, one categorical head per action dimension, and a
scalar critic, with orthogonal initialisation (sqrt 2 trunk, 0.01 actor heads, 1.0
critic). RLlib consumes the concatenated head logits for MultiDiscrete actions.
Importing this module requires the rl extra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn

MODEL_ARCHITECTURE = "autops_actor_critic_v1"


class AUTOPSActorCriticModel(TorchModelV2, nn.Module):
    """Shared trunk, per-dimension actor heads, and one value head."""

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: dict[str, Any],
        name: str,
        **kwargs: Any,
    ) -> None:
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        hidden = int(model_config.get("custom_model_config", {}).get("hidden_size", 256))
        action_dims = [int(dim) for dim in action_space.nvec]
        if num_outputs != sum(action_dims):
            raise ValueError(f"expected {sum(action_dims)} logits, got num_outputs={num_outputs}")
        self.trunk = nn.Sequential(
            nn.Linear(int(np.prod(obs_space.shape)), hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor_heads = nn.ModuleList([nn.Linear(hidden, dim) for dim in action_dims])
        self.critic_head = nn.Linear(hidden, 1)
        self._value: torch.Tensor | None = None
        for module in self.trunk.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        for head in self.actor_heads:
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
        nn.init.zeros_(self.critic_head.bias)

    def forward(
        self, input_dict: dict[str, Any], state: list[Any], seq_lens: Any
    ) -> tuple[torch.Tensor, list[Any]]:
        obs = input_dict.get("obs_flat")
        features = self.trunk((input_dict["obs"] if obs is None else obs).float())
        self._value = self.critic_head(features).squeeze(-1)
        return torch.cat([head(features) for head in self.actor_heads], dim=-1), state

    def value_function(self) -> torch.Tensor:
        if self._value is None:
            raise RuntimeError("value_function() called before forward()")
        return self._value


def register_models() -> None:
    """Register the AUTOPS model with RLlib; idempotent within a process."""

    ModelCatalog.register_custom_model(MODEL_ARCHITECTURE, AUTOPSActorCriticModel)


__all__ = ["MODEL_ARCHITECTURE", "AUTOPSActorCriticModel", "register_models"]
