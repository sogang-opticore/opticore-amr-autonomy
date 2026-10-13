from __future__ import annotations

import math
from typing import Any

import numpy as np

from amr_lite.envs.dynamics import RobotState
from amr_lite.planning.path_utils import PathState
from amr_lite.sensors.lidar import RESAMPLE_RULE_VERSION


OBSERVATION_SIZE = 296
OBSERVATION_CONTRACT_VERSION = "amr-lite-obs-v1"
HISTORY_FRAMES = 4
OBSERVATION_TAIL = (
    "goal_distance", "goal_sin", "goal_cos", "linear_velocity", "angular_velocity",
    "cross_track_error", "path_heading_sin", "path_heading_cos",
)


def observation_contract(config: dict[str, Any]) -> dict[str, Any]:
    """Machine-checkable description of the observation contract (ARCHITECTURE §4)."""
    return {
        "version": OBSERVATION_CONTRACT_VERSION,
        "size": OBSERVATION_SIZE,
        "history_frames": HISTORY_FRAMES,
        "frame_order": "oldest_to_newest",
        "lidar_rays": int(config["lidar_rays"]),
        "lidar_fov_deg": float(config["lidar_fov_deg"]),
        "lidar_max_range": float(config["lidar_max_range"]),
        "resample_rule": RESAMPLE_RULE_VERSION,
        "tail": list(OBSERVATION_TAIL),
        "goal_distance_scale": float(config["goal_distance_scale"]),
        "max_cross_track_error": float(config["max_cross_track_error"]),
        "lookahead_distance": float(config["lookahead_distance"]),
    }


def build_observation(scan_history: list[np.ndarray], path: PathState, robot: RobotState,
                      lidar_max_range: float, goal_distance_scale: float,
                      v_max: float, w_max: float, max_cross_track_error: float) -> np.ndarray:
    if len(scan_history) != 4 or any(scan.shape != (72,) for scan in scan_history):
        raise ValueError("Observation contract requires four 72-ray scans")
    scans = np.concatenate([np.clip(scan, 0.0, lidar_max_range) / lidar_max_range for scan in scan_history])
    tail = np.asarray([
        np.clip(path.rho_goal / goal_distance_scale, 0.0, 1.0),
        math.sin(path.alpha_goal), math.cos(path.alpha_goal),
        np.clip(robot.v / v_max, 0.0, 1.0), np.clip(robot.w / w_max, -1.0, 1.0),
        np.clip(path.cross_track_error / max_cross_track_error, -1.0, 1.0),
        math.sin(path.heading_error), math.cos(path.heading_error),
    ], dtype=np.float32)
    observation = np.concatenate((scans.astype(np.float32), tail)).astype(np.float32)
    if observation.shape != (OBSERVATION_SIZE,) or not np.isfinite(observation).all():
        raise ValueError("Invalid observation")
    return observation
