"""Export a deterministic M3 SB3 actor and cross-check its NumPy runtime."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.torch_layers import FlattenExtractor

from amr_lite.deploy.export import BUNDLE_FORMAT, BUNDLE_FORMAT_VERSION, PROFILE_KEYS, SCAN_SHIELD_KEYS, _Graph
from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.experiments.manifest import sha256_file
from amr_lite.learning.action import action_contract
from amr_lite.learning.observation import observation_contract
from amr_lite.learning.stamped_io import M3_IO_ID


def export_sb3(model, checkpoint: Path, output: Path, config, observations, provenance) -> float:
    policy = model.policy
    if not isinstance(policy.features_extractor, FlattenExtractor):
        raise ValueError("M3 export expects identity features from a flat MlpPolicy observation")
    if policy.squash_output or not np.all(model.action_space.low == -1) or not np.all(model.action_space.high == 1):
        raise ValueError("M3 export expects the SB3 Gaussian actor with normalized Box clipping")
    graph = _Graph()
    layers = graph.ops(policy.mlp_extractor.policy_net) + graph.ops(policy.action_net)
    expected, _ = model.predict(observations, deterministic=True)
    output.mkdir()
    np.savez(output / "weights.npz", **graph.weights)
    np.savez(output / "golden.npz", observations=observations, actions=expected.astype(np.float32))
    manifest = {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_FORMAT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_contract": {**observation_contract(config), "acquisition_contract": M3_IO_ID},
        "action_contract": action_contract(config),
        "robot_profile": {"name": config["robot_profile"], "source": "training",
                          "values": {key: config[key] for key in PROFILE_KEYS if key in config}},
        "normalization": {"observations": "fixed_contract_scales", "rewards": "none"},
        "policy": {"kind": "sb3_ppo", "mode": "direct",
                   "network": {"type": "mlp", "layers": layers, "output": "clip"}},
        "safety": {key: config[key] for key in SCAN_SHIELD_KEYS if key in config},
        "provenance": {**provenance, "checkpoint": checkpoint.name,
                       "checkpoint_sha256": sha256_file(checkpoint)},
        "files": {name: sha256_file(output / name) for name in ("weights.npz", "golden.npz")},
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return PolicyRuntime.load(output, verify=False).verify_golden()
