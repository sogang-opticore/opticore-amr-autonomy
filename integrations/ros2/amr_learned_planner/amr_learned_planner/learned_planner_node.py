#!/usr/bin/env python3
"""ROS2 node of the learned local planner.

Inputs follow the dwa_node contract: EKF odometry (``/odom`` fallback),
``/global_path`` transformed once from ``map`` into ``odom_filtered`` when
received, ``/lidar`` and ``/goal_pose``. Outputs go to the command arbiter:
``learned_planner/cmd_vel``, ``learned_planner/status`` (DWA status
vocabulary) and ``learned_planner/health`` (JSON). All decisions are made by
``LearnedPlannerCore``.
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
import tf2_ros
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult
from tf2_ros import TransformException

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry as OdometryMessage
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from amr_learned_planner.core.planner_core import LaserScanData, LearnedPlannerCore, Odometry, PlannerConfig
from amr_learned_planner.ros_common import (
    RELIABLE_QOS,
    SENSOR_QOS,
    declare_dataclass_parameters,
    declare_parameters,
    json_message,
    now_seconds,
    string_message,
    yaw_from_quaternion,
)

NODE_PARAMETERS = {
    "odom_topic": "/odometry/filtered",
    "odom_fallback_topic": "/odom",
    "global_path_topic": "/global_path",
    "goal_pose_topic": "/goal_pose",
    "scan_topic": "/lidar",
    "cmd_topic": "learned_planner/cmd_vel",
    "status_topic": "learned_planner/status",
    "health_topic": "learned_planner/health",
    "local_frame": "odom_filtered",
    "global_frame": "map",
    "path_tf_timeout": 1.0,
    "status_period": 1.0,
}


class LearnedPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("learned_planner")
        config = PlannerConfig.from_mapping(declare_dataclass_parameters(self, PlannerConfig()))
        self.p = declare_parameters(self, NODE_PARAMETERS)
        self.core = LearnedPlannerCore.from_bundle(config)
        report = self.core.health.report()
        if report["healthy"]:
            manifest = self.core.runtime.manifest
            self.get_logger().info(
                f"policy bundle {config.bundle_path} loaded: {manifest['policy']['kind']} "
                f"({manifest['policy']['mode']}), profile {manifest['robot_profile']['name']}")
        else:
            self.get_logger().error(f"learned planner has no driving authority: {report['details']}")
        if getattr(self.core, "profile_mismatches", None):
            self.get_logger().warn(f"robot profile mismatches: {self.core.profile_mismatches}")
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self._last_primary_odom: float | None = None
        self._status = "WAITING_ODOM"
        self.create_subscription(OdometryMessage, self.p["odom_topic"], self._on_odom, RELIABLE_QOS)
        self.create_subscription(OdometryMessage, self.p["odom_fallback_topic"], self._on_odom_fallback,
                                 RELIABLE_QOS)
        self.create_subscription(Path, self.p["global_path_topic"], self._on_path, RELIABLE_QOS)
        self.create_subscription(LaserScan, self.p["scan_topic"], self._on_scan, SENSOR_QOS)
        self.create_subscription(PoseStamped, self.p["goal_pose_topic"], self._on_goal, RELIABLE_QOS)
        self._cmd_pub = self.create_publisher(Twist, self.p["cmd_topic"], 10)
        self._status_pub = self.create_publisher(String, self.p["status_topic"], 10)
        self._health_pub = self.create_publisher(String, self.p["health_topic"], 10)
        self.create_timer(1.0 / max(config.control_rate, 1.0), self._on_control)
        self.create_timer(float(self.p["status_period"]), self._on_status_timer)
        self.add_on_set_parameters_callback(self._on_parameters)

    def _on_parameters(self, parameters) -> SetParametersResult:
        """Runtime switches for M5 tests; everything else needs a restart."""
        for parameter in parameters:
            if parameter.name == "fault_injection":
                self.core.config.fault_injection = str(parameter.value)
                self.get_logger().warn(f"fault_injection = {parameter.value}")
            elif parameter.name == "shield_enabled":
                self.core.config.shield_enabled = bool(parameter.value)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------- inputs
    def _odometry(self, message: OdometryMessage) -> Odometry:
        pose, twist = message.pose.pose, message.twist.twist
        return Odometry(now_seconds(self), pose.position.x, pose.position.y,
                        yaw_from_quaternion(pose.orientation), twist.linear.x, twist.angular.z)

    def _on_odom(self, message: OdometryMessage) -> None:
        self._last_primary_odom = now_seconds(self)
        self.core.on_odometry(self._odometry(message))

    def _on_odom_fallback(self, message: OdometryMessage) -> None:
        now = now_seconds(self)
        if self._last_primary_odom is None or now - self._last_primary_odom > self.core.config.odom_timeout:
            self.core.on_odometry(self._odometry(message))

    def _on_scan(self, message: LaserScan) -> None:
        self.core.on_scan(LaserScanData(now_seconds(self), np.asarray(message.ranges, dtype=np.float32),
                                        message.angle_min, message.angle_increment, message.range_min,
                                        message.range_max))

    def _on_goal(self, message: PoseStamped) -> None:
        self.get_logger().info(f"goal ({message.pose.position.x:.2f}, {message.pose.position.y:.2f})")

    def _on_path(self, message: Path) -> None:
        if not message.poses:
            self.core.on_path([])
            return
        frame = message.header.frame_id or self.p["global_frame"]
        points = [(item.pose.position.x, item.pose.position.y) for item in message.poses]
        if frame == self.p["local_frame"]:
            self.core.on_path(points)
            return
        if frame != self.p["global_frame"]:
            self.get_logger().warn(f"/global_path frame '{frame}' is not supported; ignored",
                                   throttle_duration_sec=5.0)
            return
        try:
            transform = self._tf_buffer.lookup_transform(
                self.p["local_frame"], self.p["global_frame"], Time(),
                timeout=Duration(seconds=float(self.p["path_tf_timeout"])))
        except TransformException as error:
            self.get_logger().warn(f"path TF lookup failed ({error}); keeping the previous path",
                                   throttle_duration_sec=5.0)
            return
        t, q = transform.transform.translation, transform.transform.rotation
        yaw = yaw_from_quaternion(q)
        c, s = math.cos(yaw), math.sin(yaw)
        self.core.on_path([(t.x + c * x - s * y, t.y + s * x + c * y) for x, y in points])

    # ------------------------------------------------------------ outputs
    def _on_control(self) -> None:
        output = self.core.step(now_seconds(self))
        command = Twist()
        command.linear.x, command.angular.z = float(output.v), float(output.w)
        self._cmd_pub.publish(command)
        for event in output.events:
            self._status_pub.publish(string_message(event))
        self._status = output.status
        self._health_pub.publish(json_message({
            **output.health, "status": output.status,
            "shield_mode": output.diagnostics.get("shield_mode", "PASS"),
            "shield_action": output.diagnostics.get("shield_action", "PASS"),
            "risk": output.diagnostics.get("risk", "NONE"),
        }))

    def _on_status_timer(self) -> None:
        self._status_pub.publish(string_message(self._status))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LearnedPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
