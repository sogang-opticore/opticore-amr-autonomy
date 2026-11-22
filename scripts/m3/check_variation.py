#!/usr/bin/env python3
"""Run the same development actor with nominal and measured Lite dynamics."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.envs.m3_variation import measured_variation
from amr_lite.envs.m3_warehouse import DynamicsProfile, M3WarehouseEnv, development_missions
from amr_lite.experiments.manifest import sha256_file
from amr_lite.learning.stamped_io import M3_IO_ID


def run(policy, mission, dynamics, seed):
    env = M3WarehouseEnv(mission, dynamics)
    try:
        observation, _ = env.reset(seed=seed)
        score = 0.0
        while True:
            observation, reward, terminated, truncated, info = env.step(policy.predict(observation))
            score += reward
            if terminated or truncated:
                return {
                    "mission": mission,
                    "seed": seed,
                    "profile": asdict(dynamics),
                    "result": info["disposition"],
                    "elapsed_s": info["elapsed"],
                    "goal_error_m": info["goal_error"],
                    "min_clearance_m": info["min_clearance"],
                    "shield_interventions": info["shield_interventions"],
                    "reward": score,
                }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    policy = PolicyRuntime.load(args.bundle)
    if policy.manifest["observation_contract"].get("acquisition_contract") != M3_IO_ID:
        raise ValueError("expected a stamped M3 policy")
    profile_seed = 20261123
    rng = np.random.default_rng(profile_seed)
    rows = []
    for index, mission in enumerate(development_missions()):
        for label, profile in (("nominal", DynamicsProfile()), ("measured", measured_variation(rng))):
            rows.append({"profile_kind": label, **run(policy, mission, profile, 100 + index)})
    print(json.dumps({
        "profile_seed": profile_seed, "bundle_sha256": sha256_file(args.bundle / "bundle.json"),
        "runs": rows,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
