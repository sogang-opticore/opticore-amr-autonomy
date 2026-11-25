import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from amr_lite.envs.m3_warehouse import DynamicsProfile, M3WarehouseEnv, development_missions
from amr_lite.envs.m4_curriculum import M4WarehouseEnv, curriculum_stages
from amr_lite.planning.path_utils import path_state


def test_curriculum_is_development_only_and_every_start_can_observe():
    stages = curriculum_stages()
    assert len(stages) == 3
    assert set(development_missions()).issubset(stages[-1])
    for mission_id in {name for stage in stages for name in stage}:
        env = M4WarehouseEnv(mission_id)
        obs, info = env.reset(seed=3)
        assert env.observation_space.contains(obs) and info["min_clearance"] > 0
    for reserved in ("test-crossing", "selection-static", "m4-train-unknown"):
        with pytest.raises(ValueError, match="train mission"):
            M4WarehouseEnv(reserved)


@pytest.mark.parametrize("mission", development_missions())
def test_original_train_conditions_match_m3_physics(mission):
    base, current = M3WarehouseEnv(mission), M4WarehouseEnv(mission)
    for env in (base, current):
        env.reset(seed=101)
    assert (base.path, base.goal, base.timeout, base.config["goal_tolerance"]) == (
        current.path, current.goal, current.timeout, current.config["goal_tolerance"])
    for action in ([0.5, 0.0], [-0.3, 1.0], [0.1, 0.2]) * 5:
        obs, _, terminated, truncated, info = base.step(action)
        obs4, _, terminated4, truncated4, info4 = current.step(action)
        np.testing.assert_array_equal(obs4, obs)
        assert (terminated, truncated) == (terminated4, truncated4)
        for field in ("pose_lite", "goal_error", "min_clearance", "command", "shield_mode", "disposition"):
            assert info[field] == info4[field]


def test_two_actors_occlude_and_collide_independently():
    env = M4WarehouseEnv("m4-train-multi")
    env.reset(seed=9)
    geometry = env.geometry
    grid = geometry.grid
    # At 4 s the crossing actor is on the aisle, in front of the head-on actor.
    geometry.set_time(4.0)
    origin = grid.to_lite(8, 22)
    dx, dy = np.array([1.0]), np.array([0.0])
    near, far = geometry.actors
    assert geometry.ray_distances(*origin, dx, dy, 20)[0] == pytest.approx(3.75)
    assert far.ray_distances(*origin, dx, dy, 20)[0] > 12
    for actor in (near, far):
        left, bottom, right, top = actor.box
        x, y = (left + right) / 2, (bottom + top) / 2
        assert grid.clearance(x, y, 0.2) > 0
        assert geometry.clearance(x, y, 0.2) < 0
    # Once the crossing clears the ray, the second box is visible.
    geometry.set_time(7.0)
    distance = geometry.ray_distances(*origin, dx, dy, 20)[0]
    assert distance == pytest.approx(far.box[0] - origin[0])


def test_block_release_affects_lidar_contact_and_restart():
    env = M4WarehouseEnv("m4-train-release")
    env.reset(seed=31)
    geometry = env.geometry
    center = geometry.grid.to_lite(46.5, 0)
    rx, ry = np.array([0.0]), np.array([1.0])
    geometry.set_time(6.999)
    assert geometry.clearance(*center, env.radius) < 0
    assert geometry.ray_distances(*env.start, rx, ry, 8)[0] == pytest.approx(2.2)
    # Wait through the release. The robot begins this scenario facing the block.
    for _ in range(141):
        obs, _, _, _, info = env.step([-1.0, 0.0])
    assert info["block_removed_at"] == 7
    assert geometry.clearance(*center, env.radius) > 0
    assert geometry.ray_distances(*env.start, rx, ry, 8)[0] > 4
    for _ in range(35):
        obs, _, _, _, info = env.step([0.45, 0.0])
    assert info["pose_lite"][1] > env.start[1] + 0.25


def test_replacing_route_is_immediately_observable_without_progress_reward():
    env = M4WarehouseEnv("m4-train-rejoin")
    env.reset(seed=19)
    for _ in range(39):
        before, _, _, _, _ = env.step([-1.0, 0.0])
    changed, _, _, _, info = env.step([-1.0, 0.0])
    assert env.replacement_active
    assert not info["route_rejoined"]
    assert not np.array_equal(before[288:], changed[288:])
    assert info["reward_components"]["progress"] == pytest.approx(0)
    _, _, _, _, info = env.step([-1.0, 0.0])
    assert info["reward_components"]["progress"] == pytest.approx(0)
    # A physical approach from outside (x, y) enters the replacement corridor.
    for _ in range(140):
        x, y, heading = env._pose()
        target = env.path[1][0], env.path[1][1]
        desired = math.atan2(target[1] - y, target[0] - x)
        error = math.atan2(math.sin(desired - heading), math.cos(desired - heading))
        _, _, done, truncated, info = env.step([0.3, np.clip(2 * error, -1, 1)])
        if info["route_rejoined"] or done or truncated:
            break
    assert info["route_rejoined"] and info["route_rejoin_s"] > 0
    env.reset(seed=19)
    assert not env.replacement_active and env.route_rejoin_s is None


def test_overrunning_goal_increases_remaining_cost():
    env = M4WarehouseEnv("m4-train-short")
    env.reset(seed=7)
    for dx, dy in ((0.0, 0.9), (2.0, 0.0), (2.0, 1.0)):
        env.robot.x, env.robot.y = env.goal[0] + dx, env.goal[1] + dy
        state = path_state(env.path, env.robot, 0.65)
        assert env._remaining_distance(state) == pytest.approx(math.hypot(dx, dy))


def test_stale_scan_clears_delayed_commands_before_motion():
    env = M4WarehouseEnv("m4-train-multi", DynamicsProfile(drive_delay_ticks=2))
    env.reset(seed=7)
    for _ in range(6):
        env.step([1.0, 0.0])
    env.io.frames.clear()
    before = env._pose()
    _, _, terminated, truncated, info = env.step([1.0, 0.0])
    assert terminated and not truncated and info["disposition"] == "invalid"
    assert info["applied_command"] == info["command"] == (0, 0)
    assert env._pose() == before and not env._delay
