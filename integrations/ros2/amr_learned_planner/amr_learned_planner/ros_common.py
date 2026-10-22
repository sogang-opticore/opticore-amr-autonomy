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
from nav_msgs.msg import Odometry, Path as NavPath
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from pathlib import Path

from amr_learned_planner.core.recorder import EpisodeRecorder

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
    "path_topic": "/global_path",
    "contact_topic": "m2/contacts",
    "global_frame": "map",
    "run_id": "",
    "goal_tolerance": 0.25,
    "max_pose_age": 0.5,
    "contact_max_gap": 0.25,
    "pose_topic": "/odometry/filtered",
    "scan_topic": "/lidar",
    "status_topic": "/dwa/status",
    "arbiter_state_topic": "cmd_arbiter/state",
    "learned_health_topic": "learned_planner/health",
}


def recorder_from_parameters(params: dict[str, Any], timeout: float) -> EpisodeRecorder:
    output = Path(params["output_csv"])
    if output.exists():
        raise FileExistsError(f"episode output already exists: {output}; use a fresh run path")
    return EpisodeRecorder(
        str(params["label"]), float(params["robot_radius"]), float(params["collision_clearance"]), timeout,
        (float(params["lidar_mount_x"]), float(params["lidar_mount_y"]), float(params["lidar_mount_yaw"])),
        run_id=str(params["run_id"]), goal_tolerance=float(params["goal_tolerance"]),
        max_pose_age=float(params["max_pose_age"]), contact_max_gap=float(params["contact_max_gap"]),
        events_path=output.with_suffix(".events.jsonl"),
    )


def stamp_seconds(stamp: Any) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def subscribe_recorder(node: Node, recorder: EpisodeRecorder, topics: dict[str, Any], *, on_goal: bool) -> None:
    global_frame = str(topics["global_frame"])
    buffer = Buffer()
    node.recorder_tf_listener = TransformListener(buffer, node)

    def goal(message: PoseStamped) -> None:
        now = now_seconds(node)
        stamp = stamp_seconds(message.header.stamp)
        if (message.header.frame_id != global_frame or stamp > now + 1e-6
                or (recorder.current is not None and stamp < recorder.current.started)):
            recorder.event(now, "invalid_goal", frame=message.header.frame_id, stamp=stamp)
            return
        goal_id = f"{global_frame}:{message.header.stamp.sec}.{message.header.stamp.nanosec:09d}"
        recorder.start(stamp, (message.pose.position.x, message.pose.position.y), str(topics["scenario_id"]),
                       int(topics["seed"]), goal_id=goal_id, frame=global_frame)

    def pose(message: Odometry) -> None:
        if recorder.current is None:
            return
        source, timestamp = message.header.frame_id, message.header.stamp
        stamp = stamp_seconds(timestamp)
        if not source or stamp > now_seconds(node) + 1e-6:
            return
        position = message.pose.pose.position
        x, y = position.x, position.y
        if source != global_frame:
            try:
                tr = buffer.lookup_transform(global_frame, source, Time.from_msg(timestamp)).transform
            except TransformException:
                recorder.event(stamp, "missing_tf", source=source, target=global_frame)
                return
            yaw = yaw_from_quaternion(tr.rotation)
            x, y = (tr.translation.x + math.cos(yaw) * x - math.sin(yaw) * y,
                    tr.translation.y + math.sin(yaw) * x + math.cos(yaw) * y)
        recorder.on_pose(stamp, x, y, frame=global_frame)

    def path(message: NavPath) -> None:
        stamp = stamp_seconds(message.header.stamp)
        if stamp > now_seconds(node) + 1e-6:
            return
        point = message.poses[-1].pose.position if message.poses else None
        recorder.on_path(stamp, (point.x, point.y) if point else None, message.header.frame_id)

    def scan(message: LaserScan) -> None:
        recorder.on_scan(stamp_seconds(message.header.stamp), np.asarray(message.ranges, dtype=np.float32),
                         message.angle_min, message.angle_increment, message.range_min, message.range_max)

    def contact(message: String) -> None:
        try:
            payload = json.loads(message.data)
            stamp = float(payload["stamp"])
            if stamp <= now_seconds(node) + recorder.contact_max_gap:
                recorder.on_contact(stamp, payload["collision"], payload["source"], payload.get("pairs"),
                                    payload.get("collision_events"), payload.get("coverage_epoch"))
        except (ValueError, KeyError, TypeError):
            recorder.event(now_seconds(node), "malformed_contact")

    def status(message: String) -> None:
        recorder.on_status(now_seconds(node), message.data)

    def arbiter(message: String) -> None:
        try:
            payload = json.loads(message.data)
        except ValueError:
            payload = None
        recorder.on_arbiter(now_seconds(node), payload)

    def health(message: String) -> None:
        try:
            payload = json.loads(message.data)
        except ValueError:
            payload = None
        recorder.on_health(now_seconds(node), payload)

    if on_goal:
        node.create_subscription(PoseStamped, topics["goal_pose_topic"], goal, RELIABLE_QOS)
    node.create_subscription(Odometry, topics["pose_topic"], pose, RELIABLE_QOS)
    node.create_subscription(NavPath, topics["path_topic"], path, RELIABLE_QOS)
    node.create_subscription(LaserScan, topics["scan_topic"], scan, SENSOR_QOS)
    node.create_subscription(String, topics["contact_topic"], contact, SENSOR_QOS)
    node.create_subscription(String, topics["status_topic"], status, RELIABLE_QOS)
    node.create_subscription(String, topics["arbiter_state_topic"], arbiter, RELIABLE_QOS)
    node.create_subscription(String, topics["learned_health_topic"], health, RELIABLE_QOS)
