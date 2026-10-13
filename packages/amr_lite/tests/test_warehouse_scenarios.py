import math

import pytest

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.envs.geometry import Rectangle
from amr_lite.envs.map import WarehouseMap
from amr_lite.envs.scenarios import WaypointObstacle, make_scenario
from amr_lite.envs.warehouse_scenarios import paired_spec
from amr_lite.planning.astar import astar, clearance_preference_cost


def test_paired_legs_chain_in_map_coordinates():
    config = resolve_env_config("opticore_gazebo")
    spec = paired_spec("warehouse_v1")
    first = make_scenario("WH-warehouse_v1-00", 1, None, config)
    second = make_scenario("WH-warehouse_v1-01", 1, None, config)
    grid = first.warehouse_map
    start = spec["start"]
    assert grid.to_map(first.start.x, first.start.y) == pytest.approx((start["x"], start["y"]))
    goal = spec["legs"][0]["goal"]
    assert grid.to_map(*first.goal) == pytest.approx((goal["x"], goal["y"]))
    assert (second.start.x, second.start.y) == pytest.approx(first.goal)
    path = first.reference_path
    assert second.start.theta == pytest.approx(math.atan2(path[-1][1] - path[-2][1], path[-1][0] - path[-2][0]))
    assert len(first.dynamic_obstacles) == len(spec["agents"])
    assert all(isinstance(agent, WaypointObstacle) for agent in first.dynamic_obstacles)
    with pytest.raises(KeyError):
        make_scenario("WH-warehouse_v1-99", 1, None, config)


def test_random_warehouse_pairs_are_reproducible():
    config = resolve_env_config("opticore_gazebo")
    first, again = (make_scenario("WH-RANDOM", 4, None, config) for _ in range(2))
    assert first.instance_id == again.instance_id
    assert make_scenario("WH-RANDOM", 5, None, config).instance_id != first.instance_id
    assert make_scenario("WHD-RANDOM", 4, None, config).dynamic_obstacles


def test_warehouse_episode_runs_in_the_gazebo_profile():
    env = WarehouseEnv(resolve_env_config("opticore_gazebo"), "WH-warehouse_v1-05")
    observation, info = env.reset(seed=2)
    assert info["sensor_model"]["lidar_type"] == "raw"
    for _ in range(10):
        observation, _, terminated, truncated, info = env.step([0.0, 0.0])
    assert observation.shape == (296,) and not terminated


def test_clearance_cost_matches_ros_formula_and_centres_paths():
    params = dict(inflation=0.4, preferred_clearance=0.5, clearance_cost_weight=9.0,
                  wall_avoid_clearance=0.55, wall_avoid_cost_weight=3.0, wall_avoid_min_margin=0.05)
    assert clearance_preference_cost(1.0, **params) == 0.0
    # 0.50 m: no preferred-clearance term; barrier 0.15 / 0.10 - 1 = 0.5 -> 3 * 0.25
    assert clearance_preference_cost(0.5, **params) == pytest.approx(3.0 * 0.25)
    # 0.45 m: ratio 0.1 -> 9 * 0.01; barrier 0.15 / 0.05 - 1 = 2 -> 3 * 4
    assert clearance_preference_cost(0.45, **params) == pytest.approx(9.0 * 0.01 + 3.0 * 4.0)
    # 1.6 m corridor, endpoints 0.42 m from one wall: shortest path hugs that wall,
    # the cost moves the interior of the path out to the 0.55 m wall-avoid band.
    world = WarehouseMap(obstacles=[Rectangle(0.0, 0.0, 12.0, 3.2), Rectangle(0.0, 4.8, 12.0, 8.0)])
    shortest = astar(world, (1.0, 3.62), (11.0, 3.62), 0.1, 0.4)
    costed = astar(world, (1.0, 3.62), (11.0, 3.62), 0.1, 0.4,
                   {"enabled": True, "simplify_min_clearance": 0.45, **{k: v for k, v in params.items()
                                                                     if k != "inflation"}})
    assert shortest == [(1.0, 3.62), (11.0, 3.62)]
    assert len(costed) > 2
    assert min(world.clearance(*point) for point in costed[1:-1]) >= 0.55 - 1e-9
    assert abs(world.planning_clearance(5.0, 3.62) - 0.42) < 1e-9  # bilinear, not cell-quantized
