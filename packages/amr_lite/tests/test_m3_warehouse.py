import numpy as np
import pytest

pytest.importorskip("gymnasium")

from amr_lite.envs.m3_warehouse import DynamicsProfile, M3WarehouseEnv, development_missions


def test_warmup_uses_four_physical_acquisitions_and_holds_scan_at_20_hz():
    env = M3WarehouseEnv()
    initial, info = env.reset(seed=7)
    np.testing.assert_allclose(info["scan_stamps"], [0.0, 0.1, 0.2, 0.3])
    assert info["elapsed"] == 0.0
    assert info["velocity"] == (0.0, 0.0)
    assert not np.array_equal(initial[:72], initial[72:144])
    first, _, _, _, info = env.step([0.0, 0.0])
    np.testing.assert_array_equal(initial[:288], first[:288])
    np.testing.assert_allclose(info["scan_stamps"], [0.0, 0.1, 0.2, 0.3])
    second, _, _, _, info = env.step([0.0, 0.0])
    np.testing.assert_array_equal(initial[72:288], second[:216])
    np.testing.assert_allclose(info["scan_stamps"], [0.1, 0.2, 0.3, 0.4])
    repeated, _ = env.reset(seed=7)
    np.testing.assert_array_equal(initial, repeated)


def test_invalid_action_physically_stops_and_finishes_without_advancing():
    env = M3WarehouseEnv(dynamics=DynamicsProfile(drive_delay_ticks=1))
    env.reset(seed=9)
    for _ in range(4):
        env.step([1.0, 0.0])
    pose = env.last_observation.copy()
    elapsed = env.elapsed
    obs, reward, terminated, truncated, info = env.step([float("nan"), 0.0])
    assert terminated and not truncated
    assert reward == 0.0 and env.elapsed == elapsed
    assert info["disposition"] == "invalid"
    assert info["command"] == info["applied_command"] == (0.0, 0.0)
    np.testing.assert_array_equal(obs, pose)


def test_train_mission_geometry_and_removal():
    assert development_missions() == ("train-static", "train-crossing", "train-headon", "train-blockage")
    with pytest.raises(ValueError, match="train mission"):
        M3WarehouseEnv("test-static")
    env = M3WarehouseEnv("train-blockage")
    env.reset(seed=7)
    assert env.timeout == 120
    assert env.geometry.box is not None
    x_min, y_min, x_max, y_max = env.geometry.box
    assert x_max - x_min == pytest.approx(1.8)
    assert y_max - y_min == pytest.approx(0.6)
    env.geometry.set_time(20)
    assert env.geometry.box is None
    env = M3WarehouseEnv("train-crossing")
    env.reset(seed=7)
    start = env.geometry.box
    env.geometry.set_time(12)
    assert env.geometry.box[1] - start[1] == pytest.approx(5.25)


def test_collision_wins_when_goal_inside_contact():
    env = M3WarehouseEnv("train-headon")
    env.reset(seed=7)
    env.goal = env.start
    env.geometry.clearance = lambda *_: -0.01
    _, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and not truncated
    assert info["disposition"] == "collision"
    assert info["reward_components"]["collision"] == -20.0
    assert info["reward_components"]["goal"] == 0.0
    assert reward < -20.0


def test_timeout_keeps_terminal_observation_for_bootstrap():
    env = M3WarehouseEnv()
    env.reset(seed=7)
    env.timeout = 0.07
    env.step([-1.0, 0.0])
    final, _, terminated, truncated, info = env.step([-1.0, 0.0])
    assert truncated and not terminated
    assert info["disposition"] == "timeout"
    assert info["elapsed"] == pytest.approx(0.07)
    assert env.observation_space.contains(final)
