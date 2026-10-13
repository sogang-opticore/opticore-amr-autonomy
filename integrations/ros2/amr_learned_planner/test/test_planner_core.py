import json
import math

import numpy as np
import pytest

from amr_lite.config import resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.action import normalize_command
from amr_lite.planning import RuleBasedPlanner
from amr_lite.sensors.lidar import RawLidarSpec

from amr_learned_planner.core.planner_core import (
    LaserScanData,
    LearnedPlannerCore,
    Odometry,
    PlannerConfig,
)

SPEC = RawLidarSpec()


def _core(bundle_dir, **overrides) -> LearnedPlannerCore:
    config = PlannerConfig(bundle_path=str(bundle_dir), **overrides)
    return LearnedPlannerCore.from_bundle(config)


def _scan(stamp: float, ranges: np.ndarray, pose=None) -> LaserScanData:
    return LaserScanData(stamp, np.asarray(ranges, dtype=np.float32), SPEC.angle_min, SPEC.angle_increment,
                         SPEC.range_min, SPEC.range_max, pose)


def _odometry(stamp: float, env: WarehouseEnv) -> Odometry:
    robot = env.robot
    return Odometry(stamp, robot.x, robot.y, robot.theta, robot.v, robot.w)


def _gazebo_env(scenario: str, **overrides) -> WarehouseEnv:
    return WarehouseEnv(resolve_env_config("opticore_gazebo", {"shield_enabled": False, **overrides}), scenario)


@pytest.mark.parametrize("scenario", ["S3", "D0", "P-PINCH"])
def test_core_observation_matches_lite_environment(bundle_dir, scenario):
    env = _gazebo_env(scenario)
    observation, _ = env.reset(seed=21)
    core = _core(bundle_dir)
    core.on_path(list(env.global_path))
    planner = RuleBasedPlanner(env.config)
    last_scan = None
    for step in range(60):
        if env.last_raw_scan is not last_scan:
            last_scan = env.last_raw_scan
            core.on_scan(_scan(env.elapsed, last_scan, env.last_scan_pose))
        core.on_odometry(_odometry(env.elapsed, env))
        robot = RobotState(env.robot.x, env.robot.y, env.robot.theta, env.robot.v, env.robot.w)
        mine, _ = core.build_observation(robot)
        assert np.max(np.abs(mine - observation)) <= 1e-6, f"step {step}"
        observation, _, terminated, truncated, _ = env.step(planner.predict(observation))
        if terminated or truncated:
            break


def test_closed_loop_core_drives_the_lite_robot_to_the_goal(bundle_dir):
    env = _gazebo_env("S0")
    env.reset(seed=4)
    core = _core(bundle_dir)
    core.on_path(list(env.global_path))
    events, statuses, last_scan = [], [], None
    for _ in range(600):
        if env.last_raw_scan is not last_scan:
            last_scan = env.last_raw_scan
            core.on_scan(_scan(env.elapsed, last_scan, env.last_scan_pose))
        core.on_odometry(_odometry(env.elapsed, env))
        output = core.step(env.elapsed)
        events += output.events
        statuses.append(output.status)
        if output.status == "REACHED":
            break
        env.step(normalize_command(output.v, output.w, env.config["v_max"], env.config["w_max"]))
    assert events.count("GOAL_REACHED") == 1 and statuses[-1] == "REACHED"
    assert not env.collision
    assert math.dist((env.robot.x, env.robot.y), env.scenario.goal) <= 0.25 + 0.03 + 1e-9
    assert set(statuses[:-1]) <= {"NORMAL"}
    after = core.step(env.elapsed + 0.05)
    assert (after.v, after.w, after.status, after.events) == (0.0, 0.0, "REACHED", [])


def test_status_vocabulary_and_input_faults(bundle_dir):
    core = _core(bundle_dir)
    assert core.step(0.0).status == "WAITING_ODOM"
    core.on_odometry(Odometry(0.0, 1.0, 4.0, 0.0, 0.0, 0.0))
    assert core.step(0.1).status == "STOPPED"  # no path yet
    core.on_path([(1.0, 4.0), (11.0, 4.0)])
    output = core.step(0.1)
    assert output.status == "NORMAL" and "SCAN_STALE" in output.health["faults"]
    assert (output.v, output.w) == (0.0, 0.0)
    core.on_scan(_scan(0.1, np.full(SPEC.samples, np.inf)))
    output = core.step(0.15)
    assert output.health["healthy"] and output.v > 0.0
    core.on_path([])
    assert core.step(0.2).status == "STOPPED"
    core.on_path([(1.0, 4.0), (11.0, 4.0)])
    core.on_odometry(Odometry(0.2, 1.0, 7.0, 0.0, 0.0, 0.0))  # 3 m off the path
    core.on_scan(_scan(0.2, np.full(SPEC.samples, np.inf)))
    first, second = core.step(0.25), core.step(0.3)
    assert first.status == second.status == "PATH_LOST"
    assert first.events == ["PATH_LOST"] and second.events == []  # rate limited
    output = core.step(1.5)  # odometry 1.3 s old
    assert output.status == "STOPPED" and "ODOM_STALE" in output.health["faults"]


@pytest.mark.parametrize("fault, code", [("nan_action", "NON_FINITE_ACTION"), ("slow_inference", "DEADLINE_MISS")])
def test_fault_injection_reports_health_faults(bundle_dir, fault, code):
    core = _core(bundle_dir, fault_injection=fault)
    core.on_path([(1.0, 4.0), (11.0, 4.0)])
    outputs = []
    for tick in range(4):
        now = 0.05 * tick
        core.on_odometry(Odometry(now, 1.0, 4.0, 0.0, 0.0, 0.0))
        core.on_scan(_scan(now, np.full(SPEC.samples, np.inf)))
        outputs.append(core.step(now))
    assert code in outputs[-1].health["faults"] and not outputs[-1].health["healthy"]
    if fault == "nan_action":
        assert (outputs[0].v, outputs[0].w) == (0.0, 0.0)


def test_shield_saturation_raises_emergency_for_a_replan(bundle_dir):
    core = _core(bundle_dir)
    core.on_path([(1.0, 4.0), (11.0, 4.0)])
    beams = SPEC.beam_angles()
    wall = np.where(np.abs(beams) < 1.2, 0.2 / np.maximum(np.cos(beams), 0.3), np.inf).astype(np.float32)
    events, statuses = [], []
    for tick in range(80):
        now = 0.05 * tick
        core.on_odometry(Odometry(now, 1.0, 4.0, 0.0, 0.0, 0.0))
        core.on_scan(_scan(now, wall))
        output = core.step(now)
        events += output.events
        statuses.append(output.status)
    assert statuses[0] == "NORMAL" and statuses[-1] == "EMERGENCY"
    assert events.count("EMERGENCY") == 2  # at ~1 s and again after emergency_repeat_sec


def test_reached_holds_against_replans_of_the_same_goal(bundle_dir):
    core = _core(bundle_dir)
    core.on_path([(0.0, 0.0), (1.0, 0.0)])
    core.on_odometry(Odometry(0.0, 0.95, 0.0, 0.0, 0.0, 0.0))
    core.on_scan(_scan(0.0, np.full(SPEC.samples, np.inf)))
    assert core.step(0.0).events == ["GOAL_REACHED"]
    core.on_path([(0.95, 0.0), (1.0, 0.0)])
    assert core.step(0.05).status == "REACHED"
    core.on_path([(0.95, 0.0), (5.0, 0.0)])
    assert core.step(0.1).status == "NORMAL"


def test_bundle_and_profile_faults(bundle_dir, tmp_path):
    missing = LearnedPlannerCore.from_bundle(PlannerConfig(bundle_path=str(tmp_path / "nothing")))
    report = missing.health.report()
    assert report["faults"] == ["BUNDLE_LOAD_FAILED"] and not report["healthy"]
    assert missing.step(0.0).status == "WAITING_ODOM"
    strict = _core(bundle_dir, w_max=1.8)
    assert "PROFILE_MISMATCH" in strict.health.report()["faults"]
    assert "w_max" in json.loads(strict.health.report()["details"]["PROFILE_MISMATCH"])
    warn = _core(bundle_dir, w_max=1.8, profile_check="warn")
    assert warn.health.healthy and "w_max" in warn.profile_mismatches


def test_core_reproduces_golden_observation_vectors(bundle_dir):
    """Same file as packages/amr_lite/tests/test_observation_golden.py, from the ROS side."""
    from amr_lite.config import ROOT
    from amr_lite.learning.observation import OBSERVATION_CONTRACT_VERSION

    golden = np.load(ROOT / "tests" / "golden" / "observation_v1.npz")
    assert str(golden["contract_version"]) == OBSERVATION_CONTRACT_VERSION
    core = _core(bundle_dir)
    core.on_path([tuple(point) for point in golden["path"]])
    for index, expected in enumerate(golden["observation"]):
        stamp = 0.05 * index
        if golden["refresh"][index]:
            core.on_scan(_scan(stamp, golden["raw"][index], tuple(golden["scan_pose"][index])))
        mine, _ = core.build_observation(RobotState(*golden["robot"][index]))
        assert np.max(np.abs(mine - expected)) <= 1e-6, f"index {index}"
