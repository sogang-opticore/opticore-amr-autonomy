"""Actor-critic networks for PPO v2.

Actor, critic and (optionally) cost critic have independent towers, as in the
M1 Stable PPO. The scan encoder is either a flat MLP over the 296-d
observation or a 1D CNN over rays with the four frames as channels plus a
small state branch. Exported actors are re-implemented in NumPy by
``amr_lite.deploy.runtime``.
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from amr_lite.config import deep_merge

SCAN_SIZE, STATE_SIZE, FRAMES, RAYS = 288, 8, 4, 72
ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}
DEFAULT_ARCHITECTURE: dict[str, Any] = {
    "encoder": "mlp",
    "hidden_sizes": [256, 256],
    "activation": "tanh",
    "log_std_init": -1.0,
    "cost_critic": False,
    "cnn": {"channels": [16, 32], "kernel_sizes": [5, 5], "strides": [2, 2], "state_hidden": 64},
}


def resolve_architecture(architecture: dict[str, Any] | None) -> dict[str, Any]:
    resolved = deep_merge(DEFAULT_ARCHITECTURE, architecture)
    if resolved["encoder"] not in {"mlp", "cnn"}:
        raise ValueError(f"Unknown encoder {resolved['encoder']!r}")
    if resolved["activation"] not in ACTIVATIONS:
        raise ValueError(f"Unknown activation {resolved['activation']!r}")
    return resolved


def _mlp(sizes: list[int], activation: type[nn.Module]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for size_in, size_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(size_in, size_out), activation()]
    return nn.Sequential(*layers)


class ScanCNNEncoder(nn.Module):
    """``Conv1d`` stack over the ray axis (frames as channels) + state branch."""

    def __init__(self, cnn: dict[str, Any], activation: type[nn.Module]):
        super().__init__()
        layers: list[nn.Module] = []
        channels, length = FRAMES, RAYS
        for out_channels, kernel, stride in zip(cnn["channels"], cnn["kernel_sizes"], cnn["strides"]):
            layers += [nn.Conv1d(channels, int(out_channels), int(kernel), int(stride)), activation()]
            channels, length = int(out_channels), (length - int(kernel)) // int(stride) + 1
            if length < 1:
                raise ValueError("CNN kernels/strides shrink the scan below one ray")
        self.conv = nn.Sequential(*layers)
        self.state = nn.Sequential(nn.Linear(STATE_SIZE, int(cnn["state_hidden"])), activation())
        self.output_size = channels * length + int(cnn["state_hidden"])

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        scans = observation[:, :SCAN_SIZE].reshape(-1, FRAMES, RAYS)
        return torch.cat((self.conv(scans).flatten(1), self.state(observation[:, SCAN_SIZE:])), dim=1)


class PolicyNetwork(nn.Module):
    checkpoint_kind = "ppo_v2"

    def __init__(self, architecture: dict[str, Any] | None = None):
        super().__init__()
        self.architecture = resolve_architecture(architecture)
        arch = self.architecture
        activation = ACTIVATIONS[arch["activation"]]
        hidden = [int(size) for size in arch["hidden_sizes"]]

        def tower() -> nn.Module:
            if arch["encoder"] == "cnn":
                encoder = ScanCNNEncoder(arch["cnn"], activation)
                return nn.Sequential(encoder, _mlp([encoder.output_size, *hidden], activation))
            return _mlp([SCAN_SIZE + STATE_SIZE, *hidden], activation)

        self.actor_body, self.actor_mean = tower(), nn.Linear(hidden[-1], 2)
        self.critic_body, self.critic = tower(), nn.Linear(hidden[-1], 1)
        self.cost_body = tower() if arch["cost_critic"] else None
        self.cost_critic = nn.Linear(hidden[-1], 1) if arch["cost_critic"] else None
        self.log_std = nn.Parameter(torch.full((2,), float(arch["log_std_init"])))
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor_mean.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)
        if self.cost_critic is not None:
            nn.init.orthogonal_(self.cost_critic.weight, 1.0)

    def zero_mean_head(self) -> None:
        """Residual policies start as the base policy: ``tanh(0) == 0``."""
        nn.init.zeros_(self.actor_mean.weight)
        nn.init.zeros_(self.actor_mean.bias)

    def actor_parameters(self) -> list[nn.Parameter]:
        return [*self.actor_body.parameters(), *self.actor_mean.parameters(), self.log_std]

    def mean(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.actor_body(observation))

    def forward(self, observation: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        mean = self.mean(observation)
        value = self.critic(self.critic_body(observation)).squeeze(-1)
        cost = (self.cost_critic(self.cost_body(observation)).squeeze(-1)
                if self.cost_critic is not None else None)
        return mean, self.log_std.expand_as(mean), value, cost
