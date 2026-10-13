import math

import numpy as np
import pytest

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.envs.procedural import path_length, procedural_spec
from amr_lite.envs.scenarios import DynamicObstacle, WaypointObstacle, make_scenario
from amr_lite.planning.astar import astar


def _config(split: str = "train") -> dict:
    return resolve_env_config(overrides={"procedural": {"split": split}})


@pytest.mark.parametrize("scenario_id", ["P-AISLE", "P-HALL", "P-PINCH", "P-JUNCTION", "PD-MIXED"])
@pytest.mark.parametrize("split", ["train", "test"])
def test_procedural_instances_are_reproducible_and_solvable(scenario_id, split):
    config = _config(split)
    first = make_scenario(scenario_id, 3, None, config)
    again = make_scenario(scenario_id, 3, None, config)
    other = make_scenario(scenario_id, 4, None, config)
    assert first.instance_id == again.instance_id != other.instance_id
    assert split in first.instance_id
    path = first.reference_path
    assert path[0] == (first.start.x, first.start.y) and path[-1] == first.goal
    spec = procedural_spec()
    assert spec["min_path_length"] <= path_length(path) <= spec["max_path_length"]
    inflation = config["robot_radius"] + config["obstacle_inflation"]
    assert astar(first.warehouse_map, path[0], path[-1], config["grid_resolution"], inflation)


def test_splits_use_different_generation_rules():
    splits = procedural_spec()["splits"]
    assert splits["test"]["aisle"]["aisle_width"][1] <= splits["train"]["aisle"]["aisle_width"][0]
    assert "patrol" in splits["test"]["dynamic"]["patterns"]
    assert "patrol" not in splits["train"]["dynamic"]["patterns"]
    train = make_scenario("P-AISLE", 5, None, _config("train"))
    test = make_scenario("P-AISLE", 5, None, _config("test"))
    assert train.instance_id != test.instance_id


def test_dynamic_agents_follow_the_split_patterns():
    allowed = {"cross": "cross", "stop": "stop", "patrol": "patrol", "head_on": "approach",
               "follow": "same_direction", "static": "stopped"}
    for split in ("train", "test"):
        patterns = {allowed[name] for name in procedural_spec()["splits"][split]["dynamic"]["patterns"]}
        for seed in range(6):
            scenario = make_scenario("PD-AISLE", seed, None, _config(split))
            assert scenario.dynamic_obstacles
            assert {agent.motion_type for agent in scenario.dynamic_obstacles} <= patterns


def test_unknown_family_or_split_is_rejected():
    with pytest.raises(KeyError):
        make_scenario("P-WAREHOUSE", 1, None, _config())
    with pytest.raises(KeyError):
        make_scenario("P-AISLE", 1, None, _config("holdout"))


def test_environment_uses_the_generator_path_and_scaled_timeout():
    config = resolve_env_config(overrides={"episode_timeout_per_meter": 2.0})
    env = WarehouseEnv(config, "P-PINCH")
    observation, info = env.reset(seed=8)
    assert observation.shape == (296,)
    assert tuple(env.global_path) == env.scenario.reference_path
    assert env.episode_timeout == pytest.approx(max(18.0, 6.0 + 2.0 * info["shortest_path_length"]))
    for _ in range(5):
        observation, *_ = env.step([0.0, 0.0])
    assert np.isfinite(observation).all()


def test_patrol_agent_reverses_direction():
    agent = DynamicObstacle(0.0, 0.0, 1.0, 0.0, patrol_distance=0.5)
    for step in range(60):
        agent.update(step * 0.01, 0.01)
    assert agent.vx < 0.0
    assert 0.0 < agent.x <= 0.5 + 1e-9


def test_waypoint_agent_eases_between_waypoints_and_pauses():
    agent = WaypointObstacle(0.0, 0.0, 0.0, 0.0, waypoints=((0.0, 0.0), (1.0, 0.0)), speed=0.5,
                             pause_range=(1.0, 1.0))
    for step in range(200):  # 2 s: exactly one traverse
        agent.update(step * 0.01, 0.01)
    assert agent.x == pytest.approx(1.0) and agent.vx == 0.0
    for step in range(50):  # paused
        agent.update(2.0 + step * 0.01, 0.01)
    assert agent.x == pytest.approx(1.0)
    for step in range(100):
        agent.update(2.5 + step * 0.01, 0.01)
    assert agent.x < 1.0 and agent.vx < 0.0
    assert math.isfinite(agent.vy)
