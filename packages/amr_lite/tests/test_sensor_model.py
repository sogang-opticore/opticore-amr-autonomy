import math

import numpy as np
import pytest

from amr_lite.config import load_config, load_robot_profile, resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.sensors.randomization import episode_sensor_model, perturb_path


def test_robot_profile_overrides_only_listed_keys():
    config = resolve_env_config("opticore_gazebo")
    assert config["robot_profile"] == "opticore_gazebo"
    assert config["w_max"] == pytest.approx(1.5)
    assert config["lidar_model"]["type"] == "raw"
    assert config["lidar_model"]["mount_x"] == pytest.approx(0.25)
    assert config["sensor_model"]["scan_period"] == pytest.approx(0.1)
    assert config["sensor_model"]["v_scale"] == pytest.approx(1.0)  # untouched nested key
    assert config["seed"] == load_config("env")["seed"]
    assert resolve_env_config()["robot_profile"] == "lite_default"
    with pytest.raises(FileNotFoundError):
        load_robot_profile("missing_robot")


def test_default_environment_has_identity_sensor_model():
    env = WarehouseEnv(scenario_id="S0")
    _, info = env.reset(seed=1)
    model = env.sensor_model
    assert model.lidar_type == "contract" and not model.randomized
    assert model.actuation_identity and model.localization_identity
    assert not model.scan_noise and not model.perturbs_path
    assert info["sensor_model"]["scan_period"] == pytest.approx(0.05)


def test_domain_randomization_is_seeded_and_bounded():
    config = resolve_env_config("opticore_gazebo", {"domain_randomization": {"enabled": True}})
    first, _ = episode_sensor_model(config, 101)
    again, _ = episode_sensor_model(config, 101)
    other, _ = episode_sensor_model(config, 102)
    assert first == again and first != other
    assert first.randomized
    for name, (low, high) in config["domain_randomization"]["ranges"].items():
        assert low <= getattr(first, name) <= high, name
    assert 0.0 <= first.scan_phase < first.scan_period
    assert math.hypot(first.bias_x, first.bias_y) == pytest.approx(first.localization_bias_xy)


def test_zero_probability_keeps_profile_values():
    config = resolve_env_config("opticore_gazebo",
                                {"domain_randomization": {"enabled": True, "probability": 0.0}})
    model, _ = episode_sensor_model(config, 5)
    assert not model.randomized
    assert model.range_noise_std == pytest.approx(0.03)
    assert model.scan_period == pytest.approx(0.1)


def test_unknown_randomization_key_is_rejected():
    config = resolve_env_config(overrides={"domain_randomization": {
        "enabled": True, "ranges": {"not_a_parameter": [0.0, 1.0]}}})
    with pytest.raises(KeyError):
        episode_sensor_model(config, 1)


def test_scan_hold_repeats_frames_between_ten_hertz_updates():
    config = resolve_env_config(overrides={"sensor_model": {"scan_period": 0.1}})
    env = WarehouseEnv(config, "D0")
    env.reset(seed=3)
    observations = [env.step([0.5, 0.0])[0] for _ in range(3)]
    newest, previous = slice(216, 288), slice(144, 216)
    assert not np.array_equal(observations[1][newest], observations[1][previous])  # t=0.10 update
    assert np.array_equal(observations[2][newest], observations[2][previous])      # t=0.15 held


def test_action_delay_executes_the_stop_action_first():
    config = resolve_env_config(overrides={"shield_enabled": False,
                                           "sensor_model": {"action_delay_steps": 1}})
    env = WarehouseEnv(config, "S0")
    env.reset(seed=1)
    env.step([1.0, 0.0])
    assert env.robot.v == 0.0
    env.step([1.0, 0.0])
    assert env.robot.v == pytest.approx(config["accel_max"] * config["policy_dt"])


def test_observation_delay_returns_the_previous_observation():
    base = resolve_env_config(overrides={"shield_enabled": False})
    delayed = resolve_env_config(overrides={"shield_enabled": False,
                                            "sensor_model": {"observation_delay_steps": 1}})
    now, late = WarehouseEnv(base, "D0"), WarehouseEnv(delayed, "D0")
    current, _ = now.reset(seed=2)
    stale, _ = late.reset(seed=2)
    assert np.array_equal(current, stale)
    for _ in range(5):
        previous = current
        current = now.step([0.3, 0.1])[0]
        stale = late.step([0.3, 0.1])[0]
        assert np.array_equal(stale, previous)


def test_actuation_scale_and_lag_slow_the_robot():
    def distance(sensor_model: dict) -> float:
        env = WarehouseEnv(resolve_env_config(overrides={"shield_enabled": False,
                                                         "sensor_model": sensor_model}), "S0")
        env.reset(seed=1)
        for _ in range(20):
            env.step([1.0, 0.0])
        return env.robot.x - env.scenario.start.x

    nominal = distance({})
    assert distance({"v_scale": 0.9}) < nominal
    assert distance({"actuator_time_constant": 0.2}) < nominal


def test_localization_bias_moves_only_the_perceived_pose():
    config = resolve_env_config(overrides={"sensor_model": {"localization_bias_xy": 0.1}})
    env = WarehouseEnv(config, "S0")
    observation, _ = env.reset(seed=4)
    perceived = env._perceived_robot()
    assert math.hypot(perceived.x - env.robot.x, perceived.y - env.robot.y) == pytest.approx(0.1)
    reference, _ = WarehouseEnv(scenario_id="S0").reset(seed=4)
    assert np.array_equal(observation[:288], reference[:288])
    assert not np.array_equal(observation[288:], reference[288:])


def test_perturb_path_keeps_endpoints_and_stays_close():
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0)]
    output = perturb_path(path, 0.1, 0.02, np.random.default_rng(0))
    assert output[0] == path[0] and output[-1] == path[-1]
    assert len(output) >= 70
    assert max(abs(y) for x, y in output if x < 3.5) < 0.1
