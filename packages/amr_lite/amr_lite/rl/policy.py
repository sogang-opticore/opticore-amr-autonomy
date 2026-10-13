"""Inference wrapper for PPO v2 policies (direct or residual on the rule baseline)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from amr_lite.planning.rule_baseline import RuleBasedPlanner

from .networks import PolicyNetwork


def combine_residual(base: np.ndarray, policy: np.ndarray, scale: float) -> np.ndarray:
    """``a = clip(a_base + scale * a_policy, -1, 1)`` (ARCHITECTURE §5.2)."""
    return np.clip(base + scale * policy, -1.0, 1.0).astype(np.float32)


class PPOPolicy:
    def __init__(self, network: PolicyNetwork, mode: str = "direct", residual_scale: float = 0.5,
                 base: RuleBasedPlanner | None = None, name: str = "ppo_v2"):
        if mode not in {"direct", "residual"}:
            raise ValueError(f"Unknown policy mode {mode!r}")
        if mode == "residual" and base is None:
            raise ValueError("A residual policy needs a base policy")
        self.network, self.mode, self.residual_scale, self.base = network.eval(), mode, float(residual_scale), base
        self.name = name

    def base_actions(self, observations: np.ndarray) -> np.ndarray:
        return np.stack([self.base.predict(item) for item in observations]).astype(np.float32)

    def to_action(self, observations: np.ndarray, policy_actions: np.ndarray) -> np.ndarray:
        if self.mode == "direct":
            return policy_actions.astype(np.float32)
        return combine_residual(self.base_actions(observations), policy_actions, self.residual_scale)

    @torch.no_grad()
    def predict_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32).reshape(-1, 296)
        mean = self.network.mean(torch.as_tensor(observations))
        if not deterministic:
            mean = mean + torch.randn_like(mean) * self.network.log_std.exp()
        return self.to_action(observations, torch.tanh(mean).numpy())

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.predict_batch(np.asarray(observation)[None, :], deterministic)[0]


def build_policy(checkpoint: dict[str, Any], name: str = "ppo_v2") -> PPOPolicy:
    network = PolicyNetwork(checkpoint["architecture"])
    network.load_state_dict(checkpoint["model_state"])
    policy = checkpoint.get("policy") or {}
    mode = str(policy.get("mode", "direct"))
    base = None
    if mode == "residual":
        base = RuleBasedPlanner(checkpoint["env_config"], checkpoint.get("baseline_config"))
    return PPOPolicy(network, mode, float(policy.get("residual_scale", 0.5)), base, name)


def load_ppo_v2(path: str | Path) -> PPOPolicy:
    return build_policy(torch.load(path, map_location="cpu", weights_only=False))
