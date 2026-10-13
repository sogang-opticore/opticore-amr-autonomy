"""Small rclpy helpers shared by the nodes (kept free of planning logic)."""
from __future__ import annotations

import json
import math
from dataclasses import fields
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

SENSOR_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
RELIABLE_QOS = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE)


def yaw_from_quaternion(q: Any) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def now_seconds(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def declare_dataclass_parameters(node: Node, instance: Any) -> dict[str, Any]:
    """Declare one ROS parameter per dataclass field (default = field value)."""
    return {item.name: node.declare_parameter(item.name, getattr(instance, item.name)).value
            for item in fields(instance)}


def declare_parameters(node: Node, defaults: dict[str, Any]) -> dict[str, Any]:
    return {name: node.declare_parameter(name, default).value for name, default in defaults.items()}


def string_message(text: str) -> String:
    message = String()
    message.data = text
    return message


def json_message(payload: dict[str, Any]) -> String:
    return string_message(json.dumps(payload, default=str, sort_keys=True))


RECORDER_TOPICS = {
    "goal_pose_topic": "/goal_pose",
    "pose_topic": "/odometry/filtered",
    "scan_topic": "/lidar",
    "status_topic": "/dwa/status",
    "arbiter_state_topic": "cmd_arbiter/state",
    "learned_health_topic": "learned_planner/health",
}


def subscribe_recorder(node: Node, recorder: Any, topics: dict[str, str], *, on_goal: bool) -> None:
    """Feed an ``EpisodeRecorder`` from the navigation topics.

    ``on_goal`` starts an episode for every received goal (passive recorder);
    the goal sequence runner starts episodes itself.
    """
    def goal(message: PoseStamped) -> None:
        recorder.start(now_seconds(node), (message.pose.position.x, message.pose.position.y))

    def pose(message: Odometry) -> None:
        position = message.pose.pose.position
        recorder.on_pose(now_seconds(node), position.x, position.y)

    def scan(message: LaserScan) -> None:
        recorder.on_scan(now_seconds(node), np.asarray(message.ranges, dtype=np.float32), message.angle_min,
                         message.angle_increment, message.range_min, message.range_max)

    def status(message: String) -> None:
        recorder.on_status(now_seconds(node), message.data)

    def arbiter(message: String) -> None:
        recorder.on_arbiter(now_seconds(node), json.loads(message.data))

    def health(message: String) -> None:
        recorder.on_health(now_seconds(node), json.loads(message.data))

    if on_goal:
        node.create_subscription(PoseStamped, topics["goal_pose_topic"], goal, RELIABLE_QOS)
    node.create_subscription(Odometry, topics["pose_topic"], pose, RELIABLE_QOS)
    node.create_subscription(LaserScan, topics["scan_topic"], scan, SENSOR_QOS)
    node.create_subscription(String, topics["status_topic"], status, RELIABLE_QOS)
    node.create_subscription(String, topics["arbiter_state_topic"], arbiter, RELIABLE_QOS)
    node.create_subscription(String, topics["learned_health_topic"], health, RELIABLE_QOS)
