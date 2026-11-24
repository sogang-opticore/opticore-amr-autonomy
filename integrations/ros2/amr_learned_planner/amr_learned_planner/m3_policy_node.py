"""Stamped M3 policy adapter for the single-robot warehouse."""
from __future__ import annotations

from collections import deque
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.stamped_io import M3_IO_ID, StampedPolicyIO
from amr_lite.learning.stamped_replay import ReplayTrace
from amr_lite.planning.scan_shield import ScanObstacleTracker, ScanSafetyShield, transform_points
from amr_lite.sensors.lidar import scan_points
from amr_learned_planner.ros_common import (
    RELIABLE_QOS, SENSOR_QOS, declare_parameters, json_message, now_seconds,
    stamp_seconds, string_message, yaw_from_quaternion,
)


DEFAULTS = {
    "bundle_path": "",
    "replay_path": "",
    "operational": False,
    "hold": False,
    "speed_cap": 0.45,
    "yaw_rate_cap": 0.6,
}


def load_policy(path, *, operational=False):
    runtime = PolicyRuntime.load(path)
    manifest = runtime.manifest
    profile = manifest["robot_profile"]
    observation = manifest["observation_contract"]
    history_spec = (observation.get("history_mode"), manifest["policy"].get("input_transform"))
    allowed_history = {("stack", None)}
    if operational:
        allowed_history.add(("newest", "repeat_latest_scan"))
    nominal = {"policy_dt": 0.05, "v_max": 1.5, "w_max": 1.5, "accel_max": 1.0,
               "decel_max": 2.0, "angular_accel_max": 1.5, "robot_radius": 0.36}
    if (manifest["policy"]["kind"] != "sb3_ppo"
            or runtime.mode != "direct"
            or observation.get("acquisition_contract") != M3_IO_ID
            or history_spec not in allowed_history
            or runtime.action_limits != (1.5, 1.5)
            or profile.get("name") != "opticore_gazebo"
            or any(profile["values"].get(key) != value for key, value in nominal.items())
            or manifest["safety"]["shield_strategy"] != "scan"
            or manifest["safety"]["scan_shield"].get("reverse_enabled") is not False
            or manifest.get("normalization") != {"observations": "fixed_contract_scales", "rewards": "none"}):
        raise ValueError("M3 requires an SB3 PPO direct-action bundle with the stamped observation contract")
    return runtime


class M3PolicyNode(Node):
    def __init__(self):
        super().__init__("m3_policy")
        self.params = declare_parameters(self, DEFAULTS)
        self.runtime = load_policy(self.params["bundle_path"], operational=self.params["operational"])
        self.speed_cap = float(self.params["speed_cap"])
        self.yaw_cap = float(self.params["yaw_rate_cap"])
        if not 0 < self.speed_cap <= 0.45 or not 0 < self.yaw_cap <= 0.6:
            raise ValueError("M3 caps must be positive and within 0.45 m/s and 0.6 rad/s")
        self.namespace = self.get_namespace().strip("/")
        self.local_frame = f"{self.namespace}/odom_filtered"
        self.base_frame = f"{self.namespace}/base_footprint"
        self.scan_frames = (f"{self.base_frame}/lidar_sensor", f"{self.namespace}/lidar_link")
        self.io = StampedPolicyIO()
        safety = self.runtime.manifest["safety"]
        self.shield = ScanSafetyShield({**self.runtime.robot_profile["values"], "scan_shield": safety["scan_shield"]})
        self.tracker = ScanObstacleTracker(safety["scan_tracker"])
        self.odom_history = deque(maxlen=40)
        self.pending_scans = deque(maxlen=16)
        self.shield_scan = None
        self.hold = bool(self.params["hold"])
        self.odom = None
        self.goal = None
        self.path = None
        self.path_revision = 0
        self.last_clock = None
        self.reached = False
        self.trace_file = (Path(self.params["replay_path"]).open("x", buffering=1)
                           if self.params["replay_path"] else None)
        self.trace = ReplayTrace(self.trace_file) if self.trace_file else None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.create_subscription(Odometry, "odometry/filtered", self.receive_odom, RELIABLE_QOS)
        self.create_subscription(LaserScan, "lidar", self.receive_scan, SENSOR_QOS)
        self.create_subscription(PoseStamped, "goal_pose", self.receive_goal, RELIABLE_QOS)
        self.create_subscription(NavPath, "global_path", self.receive_path, RELIABLE_QOS)
        self.cmd = self.create_publisher(Twist, "learned_planner/cmd_vel", 10)
        self.status = self.create_publisher(String, "learned_planner/status", 10)
        self.health = self.create_publisher(String, "learned_planner/health", 10)
        self.create_timer(0.05, self.control)
        self.add_on_set_parameters_callback(self.set_parameters)

    def reset_history(self):
        self.io.reset()
        self.pending_scans.clear()
        self.shield_scan = None
        self.tracker.reset()
        if self.trace:
            self.trace.reset()

    def set_parameters(self, parameters):
        if any(p.name != "hold" or not isinstance(p.value, bool) for p in parameters):
            return SetParametersResult(successful=False, reason="only the M3 diagnostic hold is writable")
        for parameter in parameters:
            if self.hold != parameter.value:
                self.hold = parameter.value
                self.reset_history()
        return SetParametersResult(successful=True)

    def receive_odom(self, msg):
        if msg.header.frame_id != self.local_frame or msg.child_frame_id != self.base_frame:
            self.odom = None
            self.odom_history.clear()
            return
        stamp = stamp_seconds(msg.header.stamp)
        if not self.odom_history or stamp > self.odom_history[-1][0]:
            pose, twist = msg.pose.pose, msg.twist.twist
            self.odom_history.append((stamp, RobotState(pose.position.x, pose.position.y,
                        yaw_from_quaternion(pose.orientation), twist.linear.x, twist.angular.z)))

    def receive_scan(self, msg):
        frame = msg.header.frame_id
        if frame not in self.scan_frames:
            self.reset_history()
            return
        args = (self.namespace, frame, stamp_seconds(msg.header.stamp), np.asarray(msg.ranges, dtype=np.float32),
                msg.angle_min, msg.angle_increment, msg.range_min, msg.range_max)
        self.pending_scans.append(args)

    def consume_scans(self, now):
        while self.pending_scans:
            args = self.pending_scans[0]
            stamp = args[2]
            # DDS /clock can trail a sensor by a physics tick. Carry that acquisition
            # to the next executing clock tick, while sending larger jumps to validation.
            if now < stamp <= now + 0.02:
                return
            self.pending_scans.popleft()
            accepted = self.io.ingest_scan(*args)
            if self.trace:
                self.trace.scan(*args)
            if not accepted or not self.odom_history:
                continue
            odom_stamp, robot = min(self.odom_history, key=lambda sample: abs(sample[0] - stamp))
            if abs(odom_stamp - stamp) > 0.08:
                continue
            pose = robot.x, robot.y, robot.theta
            mount = self.io.raw_scan
            points = scan_points(*args[3:], mount_x=mount.mount_x, mount_y=mount.mount_y, mount_yaw=mount.mount_yaw,
                                 max_distance=float(self.shield.settings["max_point_range"]) + 0.4)
            self.tracker.update(points, pose, stamp)
            self.shield_scan = stamp, pose, points

    def receive_goal(self, msg):
        now = self.clock_time()
        stamp = stamp_seconds(msg.header.stamp)
        point = msg.pose.position.x, msg.pose.position.y
        if msg.header.frame_id != "map" or not all(map(math.isfinite, point)) or not 0 <= stamp <= now + 0.25:
            return
        if self.goal is None or stamp > self.goal[0]:
            goal_id = f"map:{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            self.goal = stamp, point, goal_id
            self.reached = False
            self.reset_history()

    def receive_path(self, msg):
        now = self.clock_time()
        stamp = stamp_seconds(msg.header.stamp)
        if not 0 <= stamp <= now + 0.25 or (self.path is not None and stamp < self.path[0]):
            return
        points = tuple((p.pose.position.x, p.pose.position.y) for p in msg.poses)
        if msg.header.frame_id != "map" or not all(math.isfinite(v) for pair in points for v in pair):
            points = ()
        self.path = stamp, points
        self.path_revision += 1

    def path_matches_goal(self, now):
        if self.goal is None or self.path is None:
            return False
        goal_stamp, goal, _ = self.goal
        path_stamp, points = self.path
        # Reject empty and unrelated replans. Path and goal use distinct DDS publishers.
        return goal_stamp <= path_stamp <= now and len(points) >= 2 and math.dist(points[-1], goal) <= 0.4

    def active_route(self, now):
        if not self.path_matches_goal(now):
            return None
        try:
            # Operational authority requires TF at the pose acquisition. A cached
            # latest transform from a stopped localization node must not permit fallback.
            query = Time(nanoseconds=round(self.odom[0] * 1e9)) if self.params["operational"] else Time()
            transform = self.buffer.lookup_transform(self.local_frame, "map", query).transform
        except TransformException:
            return None
        translation, yaw = transform.translation, yaw_from_quaternion(transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        def to_local(pair):
            x, y = pair
            return translation.x + c * x - s * y, translation.y + s * x + c * y
        return [to_local(p) for p in self.path[1]], to_local(self.goal[1])

    def publish(self, now, status, fault=None, observation=None, action=None, target=None,
                command=(0.0, 0.0), inference_ms=None, shield=None):
        msg = Twist()
        msg.linear.x, msg.angular.z = command
        self.cmd.publish(msg)
        self.status.publish(string_message(status))
        self.health.publish(json_message({
            "stamp": now, "status": status, "healthy": fault is None,
            "faults": [fault] if fault else [], "contract": M3_IO_ID,
            "goal_id": self.goal[2] if self.goal else None, "path_revision": self.path_revision,
            "scan_stamps": list(observation.stamps) if observation is not None else [],
            "odom_stamp": self.odom[0] if self.odom else None,
            "odom_xy": [self.odom[1].x, self.odom[1].y] if self.odom else None,
            "path_stamp": self.path[0] if self.path else None,
            "path_waiting": not self.path_matches_goal(now),
            "shield_scan_stamp": self.shield_scan[0] if self.shield_scan else None,
            "support_ready": bool(observation is not None and observation.tensor is not None
                                  and observation.fault is None and self.shield_scan is not None
                                  and self.shield_scan[0] == observation.stamps[-1]),
            "action": action.tolist() if action is not None and np.isfinite(action).all() else None,
            "target": target, "command": command, "inference_ms": inference_ms,
            "shield_mode": shield.mode if shield else None, "risk": shield.risk_source if shield else None,
        }))

    def support_command(self, sample, command, robot, now):
        if command.fault is not None:
            return command.fault, (0.0, 0.0), None
        if self.shield_scan is None or self.shield_scan[0] != sample.stamps[-1]:
            return "SCAN_POSE_UNAVAILABLE", (0.0, 0.0), None
        _, capture_pose, capture_points = self.shield_scan
        pose = robot.x, robot.y, robot.theta
        points = transform_points(capture_points, capture_pose, pose)
        tracks = self.tracker.relative_tracks(pose, now, self.shield.settings["track_min_speed"])
        shield = self.shield.apply(robot, *command.target, points, tracks)
        v, w = command.command if shield.mode == "PASS" else (shield.v, shield.w)
        limited = max(0.0, min(v, self.speed_cap)), max(-self.yaw_cap, min(w, self.yaw_cap))
        return None, limited, shield

    def clock_time(self):
        now = now_seconds(self)
        if self.last_clock is not None and now < self.last_clock:
            self.goal = self.path = self.odom = None
            self.reached = False
            self.odom_history.clear()
            self.reset_history()
        self.last_clock = now
        return now

    def control(self):
        now = self.clock_time()
        self.consume_scans(now)
        if self.hold:
            self.publish(now, "STOPPED", "DIAGNOSTIC_HOLD")
            return
        self.odom = next((sample for sample in reversed(self.odom_history) if sample[0] <= now + 1e-9), None)
        if self.odom is None:
            self.publish(now, "STOPPED", "ODOM_UNAVAILABLE")
            return
        route = self.active_route(now)
        odom_stamp, robot = self.odom
        path, goal = route if route else ([], None)
        observation = self.io.observe(now, odom_stamp, robot, path)
        action = command = None
        inference_ms = 0.0
        if observation.tensor is not None and observation.fault is None:
            if self.reached or math.dist((robot.x, robot.y), goal) <= 0.4:
                self.reached = True
                self.publish(now, "REACHED", observation=observation)
            elif observation.path is not None and abs(observation.path.cross_track_error) > 1.0:
                self.publish(now, "STOPPED", "OUTSIDE_DEVELOPMENT_CORRIDOR", observation)
            else:
                started = time.perf_counter()
                try:
                    action = self.runtime.predict(observation.tensor)
                except (ValueError, RuntimeError, FloatingPointError):
                    action = np.full(2, np.nan, dtype=np.float32)
                inference_ms = 1000.0 * (time.perf_counter() - started)
                command = self.io.actuate(action, robot, inference_ms=inference_ms)
                fault, limited, shield = self.support_command(observation, command, robot, now)
                self.publish(now, "STOPPED" if fault else "NORMAL", fault, observation,
                             action, command.target, limited, inference_ms, shield)
        else:
            self.publish(now, "STOPPED", observation.fault or "OBSERVATION_UNAVAILABLE", observation)
        if self.trace:
            self.trace.tick(now, odom_stamp, robot, path, observation, action, command, inference_ms=inference_ms,
                            goal_id=self.goal[2] if self.goal else "", path_revision=self.path_revision)

    def destroy_node(self):
        if self.trace_file:
            self.trace_file.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = M3PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
