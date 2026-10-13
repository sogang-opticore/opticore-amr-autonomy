import math

import numpy as np
import pytest

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.envs.dynamics import RobotState
from amr_lite.planning.scan_shield import ScanObstacleTracker, ScanSafetyShield, transform_points


def _config(**scan_shield):
    return resolve_env_config("opticore_gazebo", {"scan_shield": scan_shield})


def _wall(x: float, half_width: float = 1.5, spacing: float = 0.02) -> np.ndarray:
    ys = np.arange(-half_width, half_width + 1e-9, spacing)
    return np.stack((np.full_like(ys, x), ys), axis=1)


def test_transform_points_round_trip():
    points = np.array([[1.0, 0.0], [0.5, -2.0]])
    source, target = (1.0, 2.0, 0.3), (-0.5, 0.7, -1.1)
    moved = transform_points(points, source, target)
    assert np.allclose(transform_points(moved, target, source), points)
    assert np.allclose(transform_points(points, source, source), points)


def test_clear_path_passes_with_limited_command():
    config = _config()
    result = ScanSafetyShield(config).apply(RobotState(0, 0, 0, 0.5, 0.0), 1.5, 0.0, _wall(6.0))
    assert result.mode == "PASS" and not result.unsafe_without_shield
    assert result.v == pytest.approx(0.5 + config["accel_max"] * config["policy_dt"])


def test_wall_ahead_is_braked_with_reachable_commands():
    config = _config()
    state = RobotState(0, 0, 0, 1.2, 0.0)
    result = ScanSafetyShield(config).apply(state, 1.5, 0.0, _wall(1.0))
    assert result.unsafe_without_shield and result.mode in {"CLAMP", "STOP"}
    assert result.risk_source == "SCAN_STATIC"
    assert state.v - result.v <= config["decel_max"] * config["policy_dt"] + 1e-9


def test_invalid_target_stops():
    result = ScanSafetyShield(_config()).apply(RobotState(0, 0, 0), math.nan, 0.0, np.zeros((0, 2)))
    assert (result.v, result.w, result.mode) == (0.0, 0.0, "STOP")


def test_approaching_track_is_predicted_without_static_points():
    shield = ScanSafetyShield(_config(dynamic_horizon=2.5))
    state = RobotState(0, 0, 0, 0.8, 0.0)
    head_on = [(3.0, 0.0, -1.0, 0.0, 0.3)]
    assert shield.apply(state, 1.0, 0.0, np.zeros((0, 2))).mode == "PASS"
    result = shield.apply(state, 1.0, 0.0, np.zeros((0, 2)), head_on)
    assert result.unsafe_without_shield and result.risk_source == "SCAN_TRACK"
    assert result.predicted_ttc < 2.5


def test_reverse_candidates_only_when_enabled():
    blocked = np.concatenate((_wall(0.55), _wall(-3.0)))
    state = RobotState(0, 0, 0, 0.0, 0.0)
    head_on = [(1.6, 0.0, -0.8, 0.0, 0.3)]
    forward_only = ScanSafetyShield(_config(dynamic_horizon=2.5)).apply(state, 0.5, 0.0, blocked, head_on)
    with_reverse = ScanSafetyShield(_config(dynamic_horizon=2.5, reverse_enabled=True)).apply(
        state, 0.5, 0.0, blocked, head_on)
    assert forward_only.action_type != "REVERSE" and forward_only.v >= 0.0
    assert with_reverse.action_type == "REVERSE" and with_reverse.v < 0.0


def test_tracker_ignores_walls_and_estimates_velocity():
    tracker = ScanObstacleTracker()
    pose = (0.0, 0.0, 0.0)
    for step in range(6):
        time = 0.1 * step
        angles = np.linspace(-0.08, 0.08, 9)
        person = np.stack((3.0 - 0.5 * time + 0.25 * np.cos(angles), 0.25 * np.sin(angles)), axis=1)
        tracker.update(np.concatenate((_wall(-2.0, 3.0), person)), pose, time)
    assert len(tracker.tracks) == 1
    track = tracker.tracks[0]
    assert track.age == 6
    assert track.vx == pytest.approx(-0.5, abs=0.05) and abs(track.vy) < 0.05
    relative = tracker.relative_tracks((0.0, 0.0, math.pi), 0.5, min_speed=0.15)
    assert relative[0][2] == pytest.approx(0.5, abs=0.05)  # seen from a robot facing -x
    tracker.update(np.zeros((0, 2)), pose, 2.0)
    assert tracker.tracks == []


def test_tracker_static_filter_drops_map_obstacles():
    tracker = ScanObstacleTracker()
    pillar = np.stack((2.0 + 0.1 * np.cos(np.linspace(0, 1, 6)), 0.1 * np.sin(np.linspace(0, 1, 6))), axis=1)
    tracker.update(pillar, (0.0, 0.0, 0.0), 0.0, static_filter=lambda x, y: abs(x - 2.0) < 0.5)
    assert tracker.tracks == []


def test_scan_shield_environment_prevents_driving_into_an_obstacle():
    outcomes = {}
    for enabled in (False, True):
        config = resolve_env_config("opticore_gazebo", {"shield_strategy": "scan",
                                                        "shield_enabled": enabled})
        env = WarehouseEnv(config, "S3")
        env.reset(seed=5)
        for _ in range(200):
            _, _, terminated, truncated, info = env.step([1.0, 0.0])
            if terminated or truncated:
                break
        outcomes[enabled] = info
    assert outcomes[False]["collision"]
    assert not outcomes[True]["collision"]
    assert outcomes[True]["shield_counts"]["STOP"] + outcomes[True]["shield_counts"]["CLAMP"] > 0
    assert "SCAN_STATIC" in outcomes[True]["shield_risk_counts"]
