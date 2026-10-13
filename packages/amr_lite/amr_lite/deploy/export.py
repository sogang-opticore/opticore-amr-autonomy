"""Export trained checkpoints as torch-free policy bundles (ARCHITECTURE §7).

A bundle holds the actor as a small op graph with float32 weights, the
observation/action contracts, the robot profile it was trained for, the
residual base-policy settings, provenance and golden input/output vectors
recorded from the torch policy at export time.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from amr_lite.config import load_config, resolve_env_config
from amr_lite.experiments.manifest import git_snapshot, sha256_file
from amr_lite.learning.action import action_contract
from amr_lite.learning.networks import load_policy
from amr_lite.learning.observation import observation_contract

BUNDLE_FORMAT = "opticore-learned-planner-bundle"
BUNDLE_FORMAT_VERSION = 1
SCAN_SHIELD_KEYS = ("scan_shield", "scan_tracker", "shield_strategy")
BASE_POLICY_ENV_KEYS = ("lidar_max_range", "goal_distance_scale", "max_cross_track_error",
                        "v_max", "w_max", "decel_max")
PROFILE_KEYS = ("robot_profile", "robot_radius", "v_max", "w_max", "accel_max", "decel_max",
                "angular_accel_max", "policy_dt", "lidar_rays", "lidar_fov_deg", "lidar_max_range",
                "lidar_model", "sensor_model", "lookahead_distance", "goal_tolerance")


class _Graph:
    """Collects ops and named float32 weights while walking torch modules."""

    def __init__(self) -> None:
        self.weights: dict[str, np.ndarray] = {}

    def _store(self, tensor: torch.Tensor) -> str:
        name = f"w{len(self.weights)}"
        self.weights[name] = tensor.detach().cpu().numpy().astype(np.float32)
        return name

    def ops(self, module: nn.Module) -> list[dict[str, Any]]:
        if isinstance(module, nn.Sequential):
            return [op for child in module for op in self.ops(child)]
        if isinstance(module, nn.Linear):
            return [{"op": "linear", "weight": self._store(module.weight), "bias": self._store(module.bias)}]
        if isinstance(module, nn.Conv1d):
            if module.padding != (0,) or module.dilation != (1,) or module.groups != 1:
                raise ValueError("Only unpadded, undilated, ungrouped Conv1d layers can be exported")
            return [{"op": "conv1d", "weight": self._store(module.weight), "bias": self._store(module.bias),
                     "stride": int(module.stride[0])}]
        if isinstance(module, nn.Tanh):
            return [{"op": "tanh"}]
        if isinstance(module, nn.ReLU):
            return [{"op": "relu"}]
        raise ValueError(f"Cannot export module {type(module).__name__}")


def _actor_graph(checkpoint: dict[str, Any], model: nn.Module) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    graph = _Graph()
    kind = checkpoint["kind"]
    if kind == "bc":
        # ActorMLP already ends with Tanh: Linear-ReLU-Linear-ReLU-Linear-Tanh.
        return {"type": "mlp", "layers": graph.ops(model.model), "output": "identity"}, graph.weights
    if kind == "ppo":
        layers = graph.ops(model.body) + graph.ops(model.actor_mean)
        return {"type": "mlp", "layers": layers, "output": "tanh"}, graph.weights
    if kind in {"ppo_stable", "ppo_v2"}:
        body = model.actor_body
        from amr_lite.rl.networks import ScanCNNEncoder

        if isinstance(body, nn.Sequential) and len(body) and isinstance(body[0], ScanCNNEncoder):
            encoder = body[0]
            spec = {"type": "scan_cnn", "frames": 4, "rays": 72, "conv": graph.ops(encoder.conv),
                    "state": graph.ops(encoder.state), "layers": graph.ops(body[1]) + graph.ops(model.actor_mean),
                    "output": "tanh"}
            return spec, graph.weights
        return {"type": "mlp", "layers": graph.ops(body) + graph.ops(model.actor_mean), "output": "tanh"}, graph.weights
    raise ValueError(f"Unsupported checkpoint kind {kind!r}")


def _torch_model(policy: Any) -> nn.Module:
    return getattr(policy, "network", None) or policy.model


def golden_observations(env_config: dict[str, Any], count: int = 64, seed: int = 5) -> np.ndarray:
    """Contract observations from rule-baseline rollouts with action noise."""
    from amr_lite.envs import WarehouseEnv
    from amr_lite.planning.rule_baseline import RuleBasedPlanner

    planner, rng = RuleBasedPlanner(env_config), np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    scenarios = ["S3", "D0", "S1", "P-PINCH"]
    for index, scenario in enumerate(scenarios):
        env = WarehouseEnv(env_config, scenario)
        observation, _ = env.reset(seed=seed + index)
        for _ in range(int(np.ceil(count / len(scenarios)))):
            rows.append(observation)
            action = np.clip(planner.predict(observation) + rng.normal(0.0, 0.4, 2), -1.0, 1.0)
            observation, _, terminated, truncated, _ = env.step(action.astype(np.float32))
            if terminated or truncated:
                observation, _ = env.reset(seed=seed + 100 + len(rows))
    return np.stack(rows[:count]).astype(np.float32)


def export_bundle(checkpoint: str | Path, output: str | Path, *, robot_profile: str | None = None,
                  selection: dict[str, Any] | None = None, golden_count: int = 64) -> dict[str, Any]:
    """Write ``bundle.json``, ``weights.npz`` and ``golden.npz`` into ``output``.

    PPO v2 checkpoints carry their environment config; older kinds (BC, M1
    PPO) were trained on ``lite_default`` unless ``robot_profile`` says
    otherwise, and the bundle records that the profile was assumed.
    """
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = load_policy(str(checkpoint))
    if "env_config" in payload:
        env_config, profile_source = payload["env_config"], "checkpoint"
    else:
        env_config, profile_source = resolve_env_config(robot_profile or "lite_default"), "assumed"
    network, weights = _actor_graph(payload, _torch_model(policy))
    mode = (payload.get("policy") or {}).get("mode", "direct")
    observations = golden_observations(env_config, golden_count)
    expected = policy.predict_batch(observations, deterministic=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "weights.npz", **weights)
    np.savez(output / "golden.npz", observations=observations, actions=expected.astype(np.float32))
    training = payload.get("config") or {}
    manifest: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_contract": observation_contract(env_config),
        "action_contract": action_contract(env_config),
        "robot_profile": {"name": env_config.get("robot_profile", "lite_default"), "source": profile_source,
                          "values": {key: env_config[key] for key in PROFILE_KEYS if key in env_config}},
        "policy": {
            "kind": payload["kind"], "mode": mode, "network": network,
            "residual_scale": float((payload.get("policy") or {}).get("residual_scale", 0.5)),
            "base_policy": ({"name": "lite_rule_baseline",
                             "env_config": {key: env_config[key] for key in BASE_POLICY_ENV_KEYS},
                             "config": payload.get("baseline_config") or load_config("baseline")}
                            if mode == "residual" else None),
        },
        "safety": {key: env_config[key] for key in SCAN_SHIELD_KEYS if key in env_config},
        "provenance": {
            "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
            "training_config_sha256": hashlib.sha256(
                json.dumps(training, sort_keys=True, default=str).encode("utf-8")).hexdigest(),
            "git": {key: value for key, value in git_snapshot().items() if key in {"commit", "branch", "dirty"}},
            "training_seed": payload.get("training_seed", training.get("seed")),
            "total_timesteps": payload.get("total_timesteps"),
            "selection": selection or payload.get("evaluation"),
        },
        "files": {name: sha256_file(output / name) for name in ("weights.npz", "golden.npz")},
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest
