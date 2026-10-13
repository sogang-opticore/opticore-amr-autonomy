"""Torch-free policy runtime for exported bundles (used by the ROS learned planner).

Only NumPy, PyYAML (through ``amr_lite.config``) and the AMR-Lite contract
modules are imported, so the robot does not need PyTorch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from amr_lite.learning.action import ACTION_CONTRACT_VERSION
from amr_lite.learning.observation import OBSERVATION_CONTRACT_VERSION, OBSERVATION_SIZE

SUPPORTED_FORMAT = ("opticore-learned-planner-bundle", 1)
GOLDEN_TOLERANCE = 1e-5


class BundleError(RuntimeError):
    """Raised with a health-fault code when a bundle must not be used."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _activation(name: str, value: np.ndarray) -> np.ndarray:
    if name == "tanh":
        return np.tanh(value)
    if name == "relu":
        return np.maximum(value, 0.0)
    if name == "identity":
        return value
    raise BundleError("BUNDLE_LOAD_FAILED", f"unknown activation {name!r}")


def _conv1d(value: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int) -> np.ndarray:
    """``value`` (B, C_in, L), ``weight`` (C_out, C_in, K) -> (B, C_out, L_out); no padding."""
    windows = np.lib.stride_tricks.sliding_window_view(value, weight.shape[2], axis=2)[:, :, ::stride, :]
    return np.einsum("bclk,ock->bol", windows, weight, optimize=True) + bias[None, :, None]


class PolicyRuntime:
    def __init__(self, manifest: dict[str, Any], weights: dict[str, np.ndarray], directory: Path | None = None):
        self.manifest, self.weights, self.directory = manifest, weights, directory
        policy = manifest["policy"]
        self.network = policy["network"]
        self.mode = policy["mode"]
        self.residual_scale = float(policy.get("residual_scale", 0.5))
        self.base = None
        if self.mode == "residual":
            from amr_lite.planning.rule_baseline import RuleBasedPlanner

            base = policy["base_policy"]
            self.base = RuleBasedPlanner(dict(base["env_config"]), dict(base["config"]))
        self.name = f"bundle:{policy['kind']}:{self.mode}"
        self.observation_overrides = {"history_mode": manifest["observation_contract"].get("history_mode", "stack")}

    @classmethod
    def load(cls, directory: str | Path, verify: bool = True) -> "PolicyRuntime":
        directory = Path(directory)
        try:
            manifest = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BundleError("BUNDLE_LOAD_FAILED", f"cannot read bundle.json: {error}") from error
        if (manifest.get("format"), manifest.get("format_version")) != SUPPORTED_FORMAT:
            raise BundleError("BUNDLE_LOAD_FAILED", f"unsupported format {manifest.get('format')!r} "
                                                    f"v{manifest.get('format_version')}")
        for name, digest in manifest.get("files", {}).items():
            path = directory / name
            if not path.exists() or _sha256(path) != digest:
                raise BundleError("BUNDLE_LOAD_FAILED", f"{name} is missing or its SHA-256 does not match")
        if manifest["observation_contract"].get("version") != OBSERVATION_CONTRACT_VERSION:
            raise BundleError("CONTRACT_MISMATCH", "observation contract "
                                                   f"{manifest['observation_contract'].get('version')!r}")
        if manifest["action_contract"].get("version") != ACTION_CONTRACT_VERSION:
            raise BundleError("CONTRACT_MISMATCH", f"action contract {manifest['action_contract'].get('version')!r}")
        with np.load(directory / "weights.npz") as archive:
            weights = {key: archive[key].astype(np.float32) for key in archive.files}
        runtime = cls(manifest, weights, directory)
        if verify:
            runtime.verify_golden()
        return runtime

    def verify_golden(self, tolerance: float = GOLDEN_TOLERANCE) -> float:
        if self.directory is None:
            raise BundleError("GOLDEN_PARITY_FAILED", "bundle has no directory")
        with np.load(self.directory / "golden.npz") as golden:
            observations, expected = golden["observations"], golden["actions"]
        error = float(np.max(np.abs(self.predict_batch(observations) - expected)))
        if not error <= tolerance:
            raise BundleError("GOLDEN_PARITY_FAILED", f"max |action error| {error:.3e} > {tolerance:.0e}")
        return error

    def _run(self, layers: list[dict[str, Any]], value: np.ndarray) -> np.ndarray:
        for layer in layers:
            op = layer["op"]
            if op == "linear":
                value = value @ self.weights[layer["weight"]].T + self.weights[layer["bias"]]
            elif op == "conv1d":
                value = _conv1d(value, self.weights[layer["weight"]], self.weights[layer["bias"]],
                                int(layer.get("stride", 1)))
            else:
                value = _activation(op, value)
        return value

    def policy_actions(self, observations: np.ndarray) -> np.ndarray:
        """Raw network output in ``[-1, 1]`` (before any residual combination)."""
        value = np.asarray(observations, dtype=np.float32).reshape(-1, OBSERVATION_SIZE)
        network = self.network
        if network["type"] == "scan_cnn":
            frames, rays = int(network["frames"]), int(network["rays"])
            scans = value[:, :frames * rays].reshape(-1, frames, rays)
            features = np.concatenate((self._run(network["conv"], scans).reshape(len(value), -1),
                                       self._run(network["state"], value[:, frames * rays:])), axis=1)
            value = self._run(network["layers"], features)
        else:
            value = self._run(network["layers"], value)
        return _activation(network.get("output", "tanh"), value).astype(np.float32)

    def predict_batch(self, observations: np.ndarray, deterministic: bool = True) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32).reshape(-1, OBSERVATION_SIZE)
        actions = self.policy_actions(observations)
        if self.mode == "residual":
            base = np.stack([self.base.predict(item) for item in observations]).astype(np.float32)
            actions = np.clip(base + self.residual_scale * actions, -1.0, 1.0).astype(np.float32)
        return actions

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        return self.predict_batch(np.asarray(observation)[None, :])[0]

    @property
    def robot_profile(self) -> dict[str, Any]:
        return self.manifest["robot_profile"]

    @property
    def action_limits(self) -> tuple[float, float]:
        contract = self.manifest["action_contract"]
        return float(contract["v_max"]), float(contract["w_max"])
