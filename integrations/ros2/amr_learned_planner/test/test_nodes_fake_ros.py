"""Node wiring (topics, parameters, callbacks, timers) on the fake ROS layer.

Skipped when real rclpy is importable; there the nodes run under ROS itself.
"""
import csv
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

import fake_ros

INSTALLED = fake_ros.install()
pytestmark = pytest.mark.skipif(not INSTALLED, reason="real rclpy available")


@pytest.fixture(autouse=True)
def _reset_fakes():
    fake_ros.Node.parameter_overrides = {}
    fake_ros.Clock.time = 10.0
    fake_ros.Buffer.transform = None
    yield


def _odometry(x, y, yaw=0.0, v=0.0):
    message = fake_ros.Odometry()
    message.pose.pose.position.x, message.pose.pose.position.y = x, y
    message.pose.pose.orientation.z, message.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    message.twist.twist.linear.x = v
    return message


def _path(points, frame):
    message = fake_ros.Path()
    message.header.frame_id = frame
    for x, y in points:
        pose = fake_ros.PoseStamped()
        pose.pose.position.x, pose.pose.position.y = x, y
        message.poses.append(pose)
    return message


def _scan():
    message = fake_ros.LaserScan()
    message.angle_min, message.angle_increment = -3.14159, 6.28318 / 719
    message.range_min, message.range_max = 0.12, 25.0
    message.ranges = [float("inf")] * 720
    return message


def _string(text):
    message = fake_ros.String()
    message.data = text
    return message


def test_learned_planner_node_publishes_commands_status_and_health(bundle_dir):
    from amr_learned_planner.learned_planner_node import LearnedPlannerNode

    fake_ros.Node.parameter_overrides = {"bundle_path": str(bundle_dir)}
    node = LearnedPlannerNode()
    assert node.core.health.healthy
    node.deliver("/odometry/filtered", _odometry(1.0, 4.0))
    node.deliver("/global_path", _path([(1.0, 4.0), (11.0, 4.0)], "odom_filtered"))
    node.deliver("/lidar", _scan())
    node.fire(0.05)
    command = node.publishers["learned_planner/cmd_vel"].messages[-1]
    assert command.linear.x > 0.0
    health = json.loads(node.publishers["learned_planner/health"].messages[-1].data)
    assert health["healthy"] and health["status"] == "NORMAL" and health["shield_mode"] == "PASS"
    node.fire(1.0)
    assert node.publishers["learned_planner/status"].messages[-1].data == "NORMAL"
    # A map-frame path is transformed once into odom_filtered.
    fake_ros.Buffer.transform = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=1.0, y=2.0, z=0.0), rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
    node.deliver("/global_path", _path([(0.0, 0.0), (5.0, 0.0)], "map"))
    assert node.core.path.points == [(1.0, 2.0), (6.0, 2.0)]
    node.deliver("/global_path", _path([(0.0, 0.0), (5.0, 0.0)], "base_link"))
    assert node.core.path.points == [(1.0, 2.0), (6.0, 2.0)]  # unsupported frame ignored
    # The /odom fallback is ignored while the EKF output is fresh.
    node.deliver("/odom", _odometry(9.0, 9.0))
    assert node.core.odometry.x == 1.0
    # Runtime fault injection for the M5 fallback tests.
    assert node.set_parameter("fault_injection", "nan_action").successful
    node.deliver("/global_path", _path([(1.0, 4.0), (11.0, 4.0)], "odom_filtered"))
    node.fire(0.05)
    health = json.loads(node.publishers["learned_planner/health"].messages[-1].data)
    assert "NON_FINITE_ACTION" in health["faults"]
    assert node.publishers["learned_planner/cmd_vel"].messages[-1].linear.x == 0.0


def test_learned_planner_node_without_bundle_has_no_authority(tmp_path):
    from amr_learned_planner.learned_planner_node import LearnedPlannerNode

    fake_ros.Node.parameter_overrides = {"bundle_path": str(tmp_path / "missing")}
    node = LearnedPlannerNode()
    node.fire(0.05)
    health = json.loads(node.publishers["learned_planner/health"].messages[-1].data)
    assert not health["healthy"] and health["faults"] == ["BUNDLE_LOAD_FAILED"]
    assert any(level == "error" for level, _ in node.get_logger().messages)


def test_cmd_arbiter_node_hands_over_and_forwards_status():
    from amr_learned_planner.cmd_arbiter_node import CmdArbiterNode

    fake_ros.Node.parameter_overrides = {"mode": "learned", "min_fallback_sec": 0.0, "recover_after_sec": 0.0}
    node = CmdArbiterNode()
    twist = fake_ros.Twist()
    twist.linear.x, twist.angular.z = 0.7, 0.2
    node.deliver("learned_planner/health", _string(json.dumps({"healthy": True, "faults": []})))
    node.deliver("learned_planner/cmd_vel", twist)
    node.fire(0.05)
    assert node.arbiter.state == "LEARNED"
    forwarded = node.publishers["cmd_vel"].messages[-1]
    assert (forwarded.linear.x, forwarded.angular.z) == (0.7, 0.2)
    node.deliver("learned_planner/status", _string("GOAL_REACHED"))
    node.deliver("dwa_fallback/status", _string("REJOIN"))
    assert [message.data for message in node.publishers["dwa/status"].messages] == ["GOAL_REACHED"]
    state = json.loads(node.publishers["cmd_arbiter/state"].messages[-1].data)
    assert state["state"] == "LEARNED" and state["mode"] == "learned"


def test_goal_sequence_runner_sends_legs_and_records(tmp_path):
    from amr_learned_planner.goal_sequence_runner_node import GoalSequenceRunnerNode

    output = tmp_path / "paired.csv"
    fake_ros.Node.parameter_overrides = {"paired_set": "warehouse_v1", "start_delay_sec": 0.0,
                                         "settle_sec": 0.0, "output_csv": str(output), "label": "dwa"}
    node = GoalSequenceRunnerNode()
    node.fire(0.2)
    goals = node.publishers["/goal_pose"].messages
    assert (goals[-1].pose.position.x, goals[-1].pose.position.y, goals[-1].header.frame_id) == (20.0, 3.0, "map")
    node.deliver("/odometry/filtered", _odometry(0.0, -2.0))
    fake_ros.Clock.time = 40.0
    node.deliver("/dwa/status", _string("GOAL_REACHED"))
    node.fire(0.2)
    assert (goals[-1].pose.position.x, goals[-1].pose.position.y) == (36.0, -3.0)
    row = next(csv.DictReader(output.open()))
    assert (row["policy"], row["scenario_id"], row["seed"], row["success"]) == ("dwa", "WH-warehouse_v1-00", "0", "1")


def test_nav_episode_recorder_node_writes_rows(tmp_path):
    from amr_learned_planner.nav_episode_recorder_node import NavEpisodeRecorderNode

    output = tmp_path / "episodes.csv"
    fake_ros.Node.parameter_overrides = {"output_csv": str(output)}
    node = NavEpisodeRecorderNode()
    goal = fake_ros.PoseStamped()
    goal.pose.position.x = 3.0
    node.deliver("/goal_pose", goal)
    node.deliver("/lidar", _scan())
    fake_ros.Clock.time = 15.0
    node.deliver("/dwa/status", _string("REACHED"))
    node.fire(0.5)
    rows = list(csv.DictReader(output.open()))
    assert len(rows) == 1
    row = rows[0]
    assert (row["success"], row["collision"], row["truncated"]) == ("1", "0", "0")
    assert float(row["episode_time"]) == pytest.approx(5.0) and float(row["goal_x"]) == 3.0
    assert np.isnan(float(row["minimum_clearance"]))  # no LiDAR return: clearance unknown
