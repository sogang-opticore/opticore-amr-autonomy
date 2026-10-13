"""Regenerate observation_v1.npz (only when the observation contract version changes).

Inputs are what a ROS node receives (raw 720-sample scans with the odometry
pose at reception, robot state, path); outputs are the Lite observations.
Run from packages/amr_lite:  PYTHONPATH=. python tests/golden/make_observation_golden.py
"""
from pathlib import Path

import numpy as np

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.learning.observation import OBSERVATION_CONTRACT_VERSION
from amr_lite.planning import RuleBasedPlanner

SCENARIO, SEED, STEPS = "S3", 33, 24


def generate() -> dict[str, np.ndarray]:
    env = WarehouseEnv(resolve_env_config("opticore_gazebo", {"shield_enabled": False}), SCENARIO)
    observation, _ = env.reset(seed=SEED)
    planner = RuleBasedPlanner(env.config)
    raw, scan_pose, refresh, robot, observations = [], [], [], [], []
    last = None
    for _ in range(STEPS):
        refresh.append(env.last_raw_scan is not last)
        last = env.last_raw_scan
        raw.append(np.asarray(last, dtype=np.float32))
        scan_pose.append(env.last_scan_pose)
        robot.append((env.robot.x, env.robot.y, env.robot.theta, env.robot.v, env.robot.w))
        observations.append(observation)
        observation, _, terminated, truncated, _ = env.step(planner.predict(observation))
        assert not (terminated or truncated)
    return {"contract_version": np.array(OBSERVATION_CONTRACT_VERSION), "scenario": np.array(SCENARIO),
            "seed": np.array(SEED), "path": np.asarray(env.global_path, dtype=np.float64),
            "raw": np.stack(raw), "scan_pose": np.asarray(scan_pose, dtype=np.float64),
            "refresh": np.asarray(refresh), "robot": np.asarray(robot, dtype=np.float64),
            "observation": np.stack(observations).astype(np.float32)}


if __name__ == "__main__":
    target = Path(__file__).with_name("observation_v1.npz")
    np.savez_compressed(target, **generate())
    print(f"wrote {target}")
