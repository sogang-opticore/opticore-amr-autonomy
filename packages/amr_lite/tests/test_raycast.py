import math

import numpy as np
import pytest

from amr_lite.envs.dynamics import RobotState
from amr_lite.envs.geometry import Rectangle
from amr_lite.envs.map import WarehouseMap
from amr_lite.envs.raycast import lidar_scan, lidar_scan_reference
from amr_lite.envs.scenarios import DynamicObstacle, make_scenario


def test_empty_room_and_front_wall_ordering():
    world = WarehouseMap(width=20, height=20)
    scan = lidar_scan(RobotState(10, 10, 0), world, [], 72, math.radians(270), 8.0)
    assert scan.shape == (72,)
    assert np.all(scan <= 8.0)
    wall = WarehouseMap(width=20, height=20, obstacles=[Rectangle(11, 5, 11.1, 15)])
    scan = lidar_scan(RobotState(10, 10, 0), wall, [], 73, math.radians(270), 8.0)
    assert scan[36] == pytest.approx(1.0, abs=0.02)


def test_circle_is_first_hit_and_rotation_changes_scan():
    world = WarehouseMap(width=20, height=20, obstacles=[Rectangle(14, 5, 14.1, 15)])
    obstacle = DynamicObstacle(12, 10, 0, 0, radius=0.5)
    forward = lidar_scan(RobotState(10, 10, 0), world, [obstacle], 73, math.radians(270), 8.0)
    rotated = lidar_scan(RobotState(10, 10, math.pi / 2), world, [obstacle], 73, math.radians(270), 8.0)
    assert forward[36] == pytest.approx(1.5, abs=0.02)
    assert not np.allclose(forward, rotated)



def test_vectorized_cast_is_bit_identical_to_reference():
    rng = np.random.default_rng(7)
    for scenario_id in ("S1", "S2", "S6", "D0", "D2"):
        scenario = make_scenario(scenario_id, 0)
        obstacles = list(scenario.dynamic_obstacles) + [DynamicObstacle(6.0, 4.0, 0.0, 0.0, radius=0.4)]
        rectangles = scenario.warehouse_map.all_rectangles()
        for index in range(60):
            if index % 3 == 0:  # exactly on a rectangle edge coordinate
                rect = rectangles[int(rng.integers(len(rectangles)))]
                x, y = rect.x_min, float(rng.uniform(0.1, 7.9))
            else:
                x, y = float(rng.uniform(-0.2, 12.2)), float(rng.uniform(-0.2, 8.2))
            theta = float(rng.choice([0.0, math.pi / 2, math.pi, rng.uniform(-math.pi, math.pi)]))
            state = RobotState(x, y, theta)
            for rays, fov in ((72, math.radians(270)), (720, 2 * math.pi)):
                fast = lidar_scan(state, scenario.warehouse_map, obstacles, rays, fov, 8.0)
                slow = lidar_scan_reference(state, scenario.warehouse_map, obstacles, rays, fov, 8.0)
                assert fast.tobytes() == slow.tobytes()
