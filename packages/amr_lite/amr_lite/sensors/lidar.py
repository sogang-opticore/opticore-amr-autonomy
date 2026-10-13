"""LiDAR models shared by the Lite simulator and the ROS2 adapter.

The observation contract (``amr-lite-obs-v1``) uses 72 range bins over 270°
measured from the robot centre. ``resample_scan`` converts any raw LaserScan
(arbitrary sample count, field of view and mount pose) into that contract.
The ROS node imports this exact function, so Lite and ROS cannot diverge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from amr_lite.envs.dynamics import RobotState
from amr_lite.envs.raycast import cast_rays

RESAMPLE_RULE_VERSION = "resample-v1"


@dataclass(frozen=True)
class ContractScanSpec:
    rays: int = 72
    fov: float = math.radians(270.0)
    max_range: float = 8.0

    @classmethod
    def from_env_config(cls, config: dict[str, Any]) -> "ContractScanSpec":
        return cls(int(config["lidar_rays"]), math.radians(float(config["lidar_fov_deg"])),
                   float(config["lidar_max_range"]))

    def angles(self) -> np.ndarray:
        return np.linspace(-self.fov / 2.0, self.fov / 2.0, self.rays, dtype=np.float64)


@dataclass(frozen=True)
class RawLidarSpec:
    """Raw sensor geometry, e.g. the Gazebo ``gpu_lidar`` of opticore-amr."""

    samples: int = 720
    angle_min: float = -3.14159
    angle_max: float = 3.14159
    range_min: float = 0.12
    range_max: float = 25.0
    mount_x: float = 0.25
    mount_y: float = 0.0
    mount_yaw: float = 0.0

    @classmethod
    def from_env_config(cls, config: dict[str, Any]) -> "RawLidarSpec":
        model = config.get("lidar_model", {})
        defaults = cls()
        return cls(
            samples=int(model.get("raw_samples", defaults.samples)),
            angle_min=float(model.get("raw_angle_min", defaults.angle_min)),
            angle_max=float(model.get("raw_angle_max", defaults.angle_max)),
            range_min=float(model.get("raw_range_min", defaults.range_min)),
            range_max=float(model.get("raw_range_max", defaults.range_max)),
            mount_x=float(model.get("mount_x", defaults.mount_x)),
            mount_y=float(model.get("mount_y", defaults.mount_y)),
            mount_yaw=float(model.get("mount_yaw", defaults.mount_yaw)),
        )

    @property
    def angle_increment(self) -> float:
        if self.samples <= 1:
            return 0.0
        return (self.angle_max - self.angle_min) / (self.samples - 1)

    def beam_angles(self) -> np.ndarray:
        return self.angle_min + np.arange(self.samples, dtype=np.float64) * self.angle_increment

    def mount_pose(self, state: RobotState) -> tuple[float, float, float]:
        c, s = math.cos(state.theta), math.sin(state.theta)
        return (state.x + c * self.mount_x - s * self.mount_y,
                state.y + s * self.mount_x + c * self.mount_y,
                state.theta + self.mount_yaw)


def simulate_raw_scan(state: RobotState, warehouse_map, dynamic_obstacles: Sequence,
                      spec: RawLidarSpec, *, cast_range: float | None = None,
                      angular_bias: float = 0.0) -> np.ndarray:
    """Noise-free raw scan in the sensor frame, ``+inf`` where nothing is hit.

    ``cast_range`` limits the simulated distance when far returns cannot reach
    the contract (they are clipped to the contract range anyway). A non-zero
    ``angular_bias`` casts beams at ``reported angle + bias``, i.e. a mount yaw
    error the robot does not know about.
    """
    ox, oy, yaw = spec.mount_pose(state)
    limit = spec.range_max if cast_range is None else min(spec.range_max, float(cast_range))
    angles = [yaw + angular_bias + float(angle) for angle in spec.beam_angles()]
    distance = cast_rays(ox, oy, angles, warehouse_map, dynamic_obstacles, limit)
    ranges = distance.astype(np.float32)
    ranges[distance >= limit] = np.inf
    return ranges


def resample_scan(ranges: Sequence[float] | np.ndarray, angle_min: float, angle_increment: float,
                  range_min: float, range_max: float, *, mount_x: float = 0.0,
                  mount_y: float = 0.0, mount_yaw: float = 0.0,
                  contract: ContractScanSpec | None = None) -> np.ndarray:
    """Convert a raw LaserScan into the contract scan (rule ``resample-v1``).

    1. NaN, -inf and readings below ``range_min`` are invalid and dropped;
       +inf and readings above ``range_max`` mean "no return".
    2. Valid returns are moved from the sensor frame into the robot frame.
    3. Each point goes to the contract bin whose centre is closest to its
       bearing from the robot centre; points outside the field of view are
       dropped.
    4. A bin keeps the smallest distance assigned to it, or ``max_range``.
    """
    spec = contract or ContractScanSpec()
    values = np.asarray(ranges, dtype=np.float64).reshape(-1)
    beam = float(angle_min) + np.arange(values.size, dtype=np.float64) * float(angle_increment)
    valid = np.isfinite(values) & (values >= float(range_min)) & (values <= float(range_max))
    output = np.full(spec.rays, spec.max_range, dtype=np.float64)
    if np.any(valid):
        r, angle = values[valid], beam[valid] + float(mount_yaw)
        px = float(mount_x) + r * np.cos(angle)
        py = float(mount_y) + r * np.sin(angle)
        bearing = np.arctan2(py, px)
        step = spec.fov / (spec.rays - 1)
        index = np.rint((bearing + spec.fov / 2.0) / step).astype(np.int64)
        inside = (index >= 0) & (index < spec.rays)
        np.minimum.at(output, index[inside], np.hypot(px, py)[inside])
    return np.clip(output, 0.0, spec.max_range).astype(np.float32)


def scan_points(ranges: Sequence[float] | np.ndarray, angle_min: float, angle_increment: float,
                range_min: float, range_max: float, *, mount_x: float = 0.0,
                mount_y: float = 0.0, mount_yaw: float = 0.0,
                max_distance: float | None = None) -> np.ndarray:
    """Valid returns as ``(N, 2)`` robot-frame points (input of the scan shield)."""
    values = np.asarray(ranges, dtype=np.float64).reshape(-1)
    beam = float(angle_min) + np.arange(values.size, dtype=np.float64) * float(angle_increment)
    valid = np.isfinite(values) & (values >= float(range_min)) & (values <= float(range_max))
    r, angle = values[valid], beam[valid] + float(mount_yaw)
    points = np.stack((float(mount_x) + r * np.cos(angle), float(mount_y) + r * np.sin(angle)), axis=1)
    if max_distance is not None and len(points):
        points = points[np.hypot(points[:, 0], points[:, 1]) <= float(max_distance)]
    return points


def contract_scan_points(scan: np.ndarray, contract: ContractScanSpec | None = None) -> np.ndarray:
    """Robot-frame points of a contract scan, ignoring max-range bins."""
    spec = contract or ContractScanSpec()
    values = np.asarray(scan, dtype=np.float64)
    angles = spec.angles()
    hit = values < spec.max_range
    return np.stack((values[hit] * np.cos(angles[hit]), values[hit] * np.sin(angles[hit])), axis=1)
