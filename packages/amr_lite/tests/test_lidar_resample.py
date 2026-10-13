import math

import numpy as np
import pytest

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.envs.dynamics import RobotState
from amr_lite.envs.geometry import Rectangle
from amr_lite.envs.map import WarehouseMap
from amr_lite.envs.raycast import lidar_scan
from amr_lite.sensors.lidar import (
    ContractScanSpec,
    RawLidarSpec,
    contract_scan_points,
    resample_scan,
    scan_points,
    simulate_raw_scan,
)

CONTRACT = ContractScanSpec()


def _empty_scan(spec: RawLidarSpec) -> np.ndarray:
    return np.full(spec.samples, np.inf, dtype=np.float32)


def _beam(spec: RawLidarSpec, angle: float) -> int:
    return int(np.argmin(np.abs(spec.beam_angles() - angle)))


def _resample(spec: RawLidarSpec, ranges: np.ndarray) -> np.ndarray:
    return resample_scan(ranges, spec.angle_min, spec.angle_increment, spec.range_min,
                         spec.range_max, mount_x=spec.mount_x, mount_y=spec.mount_y,
                         mount_yaw=spec.mount_yaw)


def test_return_lands_in_nearest_contract_bin():
    spec = RawLidarSpec(mount_x=0.0)
    ranges = _empty_scan(spec)
    ranges[_beam(spec, CONTRACT.angles()[50])] = 3.0
    scan = _resample(spec, ranges)
    assert scan.shape == (72,) and scan.dtype == np.float32
    assert scan[50] == pytest.approx(3.0, abs=1e-6)
    assert np.all(np.delete(scan, 50) == CONTRACT.max_range)


def test_mount_offset_is_applied_before_binning():
    spec = RawLidarSpec()  # Gazebo sensor 0.25 m ahead of base_footprint
    ranges = _empty_scan(spec)
    ranges[_beam(spec, 0.0)] = 1.0
    scan = _resample(spec, ranges)
    hit = np.flatnonzero(scan < CONTRACT.max_range)
    assert hit.tolist() in ([35], [36])
    assert scan[hit[0]] == pytest.approx(1.25, abs=1e-3)


def test_mount_yaw_rotates_returns():
    spec = RawLidarSpec(mount_x=0.0, mount_yaw=math.radians(90.0))
    ranges = _empty_scan(spec)
    ranges[_beam(spec, 0.0)] = 2.0  # straight ahead of a sensor facing left
    scan = _resample(spec, ranges)
    left_bin = int(np.argmin(np.abs(CONTRACT.angles() - math.radians(90.0))))
    assert scan[left_bin] == pytest.approx(2.0, abs=1e-3)


def test_bin_keeps_nearest_return_and_ignores_invalid_or_rear_points():
    spec = RawLidarSpec(mount_x=0.0)
    ranges = _empty_scan(spec)
    base = _beam(spec, CONTRACT.angles()[20])
    ranges[base], ranges[base + 1] = 4.0, 2.5
    ranges[_beam(spec, math.pi)] = 1.0           # behind the robot, outside 270 deg
    ranges[_beam(spec, 0.5)] = np.nan
    ranges[_beam(spec, 0.6)] = -np.inf
    ranges[_beam(spec, 0.7)] = 0.05              # below range_min
    ranges[_beam(spec, 0.8)] = 30.0              # beyond range_max
    scan = _resample(spec, ranges)
    assert scan[20] == pytest.approx(2.5, abs=1e-6)
    assert np.all(np.delete(scan, 20) == CONTRACT.max_range)


def test_dense_centered_raw_scan_agrees_with_contract_ray_cast():
    world = WarehouseMap(obstacles=[Rectangle(5.0, 2.5, 5.6, 5.0), Rectangle(2.0, 6.0, 4.0, 6.3)])
    state = RobotState(3.0, 4.0, 0.4)
    spec = RawLidarSpec(samples=7200, mount_x=0.0)
    raw = simulate_raw_scan(state, world, [], spec)
    resampled = _resample(spec, raw)
    contract = lidar_scan(state, world, [], 72, CONTRACT.fov, CONTRACT.max_range)
    # A bin holds the nearest return in its wedge, so it can only be shorter
    # than the centre ray; large gaps appear only where a wedge clips a corner.
    assert np.all(resampled <= contract + 1e-3)
    assert np.median(np.abs(resampled - contract)) < 0.06


def test_raw_lidar_environment_uses_the_shared_resampler():
    config = resolve_env_config("opticore_gazebo", {"sensor_model": {"range_noise_std": 0.0}})
    env = WarehouseEnv(config, "S3")
    observation, _ = env.reset(seed=4)
    spec = RawLidarSpec.from_env_config(config)
    raw = simulate_raw_scan(env.robot, env.scenario.warehouse_map, env.dynamic_obstacles, spec,
                            cast_range=env._raw_cast_range)
    expected = _resample(spec, raw) / np.float32(CONTRACT.max_range)
    assert np.allclose(observation[216:288], expected, atol=1e-7)
    assert np.array_equal(env.last_raw_scan, raw)


def test_scan_points_respect_mount_and_validity():
    spec = RawLidarSpec()
    ranges = _empty_scan(spec)
    ranges[_beam(spec, 0.0)] = 1.0
    ranges[_beam(spec, 1.0)] = np.nan
    points = scan_points(ranges, spec.angle_min, spec.angle_increment, spec.range_min,
                         spec.range_max, mount_x=spec.mount_x)
    assert points.shape == (1, 2)
    assert points[0] == pytest.approx([1.25, 0.0], abs=1e-2)
    contract = np.full(72, 8.0, dtype=np.float32)
    contract[36] = 2.0
    assert contract_scan_points(contract).shape == (1, 2)
