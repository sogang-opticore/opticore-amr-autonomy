import math

import numpy as np
import pytest

from amr_lite.envs.dynamics import RobotState
from amr_lite.envs.map import WarehouseMap
from amr_lite.envs.scenarios import DynamicObstacle
from amr_lite.learning.stamped_io import StampedPolicyIO
from amr_lite.sensors.lidar import RawLidarSpec, resample_scan, simulate_raw_scan


SCAN = RawLidarSpec()
ROBOT = RobotState(2, 4, 0)
PATH = [(2, 4), (6, 4), (9, 4)]


def scan_inputs(stamp, ranges):
    return ("/amr1", "amr1/lidar", stamp, ranges, SCAN.angle_min, SCAN.angle_increment,
            SCAN.range_min, SCAN.range_max)


def cast(stamped, stamp, distance=math.inf):
    return stamped.ingest_scan(*scan_inputs(stamp, [distance] * SCAN.samples))


def test_distinct_acquisitions_and_identical_lite_ros_observations():
    lite, ros = StampedPolicyIO(), StampedPolicyIO()
    obstacle = DynamicObstacle(4, 3.7, 0, 2.0)
    warehouse = WarehouseMap()
    for index in range(5):
        stamp = index / 10
        raw = simulate_raw_scan(ROBOT, warehouse, [obstacle], SCAN)
        bins = resample_scan(raw, SCAN.angle_min, SCAN.angle_increment, SCAN.range_min, SCAN.range_max,
                             mount_x=SCAN.mount_x)
        assert lite.ingest_bins("/amr1", "amr1/lidar", stamp, bins)
        assert ros.ingest_scan(*scan_inputs(stamp, raw))
        obstacle.update(stamp, 0.1)
        for now in (stamp, stamp + 0.05):
            lite_tick = lite.observe(now, now, ROBOT, PATH)
            ros_tick = ros.observe(now, now, ROBOT, PATH)
            if index < 3:
                assert lite_tick.fault == ros_tick.fault == "SCAN_WARMUP"
                assert lite_tick.tensor is ros_tick.tensor is None
                continue
            assert ros_tick.stamps == lite_tick.stamps == tuple(j / 10 for j in range(index - 3, index + 1))
            assert ros_tick.tensor.shape == (296,) and ros_tick.tensor.dtype == np.float32
            np.testing.assert_array_equal(lite_tick.tensor, ros_tick.tensor)
            scans = ros_tick.tensor[:288].reshape(4, 72)
            assert len(np.unique(scans, axis=0)) == 4
            np.testing.assert_allclose(ros_tick.tensor[288:], [0.65/8, 0, 1, 0, 0, 0, 0, 1])
            assert lite.actuate([0, 1], ROBOT) == ros.actuate([0, 1], ROBOT)


def test_duplicate_reorder_stale_future_and_epoch_reset():
    inputs = StampedPolicyIO()
    assert all(cast(inputs, stamp) for stamp in (0.0, 0.1, 0.2, 0.3))
    assert not cast(inputs, 0.3, 0.5)
    assert not cast(inputs, 0.1, 0.5)
    assert inputs.observe(0.35, 0.33, ROBOT, PATH).stamps == (0.0, 0.1, 0.2, 0.3)
    assert inputs.observe(0.81, 0.81, ROBOT, PATH).fault == "SCAN_STALE"
    # Arrival time of an old scan cannot restore coverage.
    assert not cast(inputs, 0.3)
    assert all(cast(inputs, stamp) for stamp in (0.9, 1.0, 1.1, 1.2))
    assert inputs.observe(1.21, 1.0, ROBOT, PATH).tensor is not None
    assert inputs.observe(1.25, 1.26, ROBOT, PATH).fault == "ODOM_FUTURE"
    assert inputs.observe(1.27, 1.26, ROBOT, PATH).fault == "SCAN_WARMUP"
    assert cast(inputs, 2.0)
    assert inputs.observe(1.28, 1.28, ROBOT, PATH).fault == "SCAN_FUTURE"
    assert cast(inputs, 1.3)
    assert inputs.observe(0.0, 0.0, ROBOT, PATH).fault == "CLOCK_RESET"
    assert cast(inputs, 0.0)
    assert inputs.observe(0.6, 0.0, ROBOT, PATH).fault == "ODOM_STALE"
    assert not cast(inputs, 0.0)


def test_snapshot_rebase_and_physical_stop():
    inputs = StampedPolicyIO()
    assert all(cast(inputs, stamp) for stamp in range(4))
    assert inputs.observe(3, 3, ROBOT, PATH).fault == "SCAN_WARMUP"
    assert all(cast(inputs, stamp) for stamp in (3.1, 3.2, 3.3))
    normal = inputs.observe(3.3, 3.3, ROBOT, PATH)
    turned = inputs.observe(3.35, 3.35, ROBOT, [(2, 4), (2, 7), (9, 7)])
    assert normal.stamps == turned.stamps
    np.testing.assert_allclose(normal.tensor[290], 1)
    np.testing.assert_allclose(turned.tensor[289], 1)
    assert inputs.observe(3.4, 3.4, ROBOT, []).fault == "PATH_UNAVAILABLE"
    assert inputs.ingest_bins("/amr2", "amr2/lidar", 3.4, np.full(72, 8.0))
    assert inputs.observe(3.4, 3.4, ROBOT, PATH).stamps == (3.4,)
    assert inputs.observe(3.4, 3.4, ROBOT, PATH).fault == "SCAN_WARMUP"
    assert not cast(inputs, 3.4, math.nan)
    assert inputs.observe(3.4, 3.4, ROBOT, PATH).fault == "SCAN_INVALID"
    assert cast(inputs, 3.5)
    assert not inputs.ingest_scan("/amr2", "amr2/lidar", 3.6, [1.0], 0, 0, 0.1, 10.0)
    assert inputs.observe(3.6, 3.6, ROBOT, PATH).fault == "SCAN_INVALID"
    moving = RobotState(2, 4, 0, v=0.7, w=0.2)
    command = inputs.actuate([0.0, 0.0], moving)
    assert command.target == (0.75, 0.0)
    assert command.command == pytest.approx((0.75, 0.125))
    for action in ([math.nan, 0.0], [0.0], [0.0, "invalid"]):
        result = inputs.actuate(action, moving)
        assert result.command == (0, 0) and result.fault == "INVALID_ACTION"
    result = inputs.actuate([1, 1], moving, inference_ms=26)
    assert result.command == (0, 0) and result.fault == "INFERENCE_DEADLINE"
