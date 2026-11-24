"""Goal-scoped authority for the post-M1 warehouse integration."""
from __future__ import annotations

import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from rclpy.node import Node
from std_msgs.msg import String

from amr_learned_planner.core.arbiter import Source
from amr_learned_planner.ros_common import (
    RELIABLE_QOS, declare_parameters, json_message, now_seconds, stamp_seconds, string_message,
)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def numeric_pair(value):
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(map(finite_number, value))


class OperationalArbiter(Node):
    """The only final Twist publisher in operational mode. Times use the ROS clock."""

    def __init__(self):
        super().__init__("cmd_arbiter")
        self.params = declare_parameters(self, {
            "report_timeout": 0.20, "cmd_timeout": 0.25, "fallback_hold_sec": 0.30,
            "recover_after_sec": 1.0, "stall_distance_m": 0.12,
            "replan_after_sec": 3.0, "blocked_timeout_sec": 15.0,
        })
        if (not all(finite_number(v) and v > 0 for v in self.params.values())
                or self.params["blocked_timeout_sec"] <= self.params["replan_after_sec"]):
            raise ValueError("invalid operational timeouts or movement threshold")
        self.cmd = self.create_publisher(Twist, "cmd_vel", 10)
        self.status = self.create_publisher(String, "dwa/status", 10)
        self.state_pub = self.create_publisher(String, "cmd_arbiter/state", 10)
        self.last_clock = None
        self.goal_stamp = -math.inf
        self.goal_id = None
        self.reset_authority()
        self.create_subscription(PoseStamped, "goal_pose", self.on_goal, RELIABLE_QOS)
        self.create_subscription(String, "cancel_goal", self.on_cancel, RELIABLE_QOS)
        self.create_subscription(String, "learned_planner/health", self.on_report, RELIABLE_QOS)
        self.create_subscription(Twist, "dwa_fallback/cmd_vel", self.on_dwa, RELIABLE_QOS)
        self.create_subscription(String, "dwa_fallback/status", self.on_dwa_status, RELIABLE_QOS)
        self.create_timer(0.05, self.tick)

    def reset_authority(self):
        self.report = None
        self.report_stamp = -math.inf
        self.dwa = Source()
        self.fallback_since = self.healthy_since = None
        self.anchor = self.anchor_stamp = self.request_stamp = self.no_path_since = None
        self.terminal = None

    def on_goal(self, msg):
        now, stamp = self.clock_time(), stamp_seconds(msg.header.stamp)
        point = (msg.pose.position.x, msg.pose.position.y)
        if msg.header.frame_id != "map" or not numeric_pair(point) or not 0 <= stamp <= now + 0.25:
            return
        if stamp <= self.goal_stamp:
            return
        self.reset_authority()
        self.goal_stamp = stamp
        self.goal_id = f"map:{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        self.emit("WAIT_INPUT", "STOPPED")

    def on_cancel(self, msg):
        if msg.data == self.goal_id and self.terminal is None:
            self.terminal = "CANCELLED"
            self.emit(self.terminal, "STOPPED")

    def on_report(self, msg):
        try:
            report = json.loads(msg.data)
        except (ValueError, TypeError):
            report = None
        if not isinstance(report, dict):
            self.report = None
            return
        if self.goal_id is None or report.get("goal_id") != self.goal_id:
            return
        stamp = report.get("stamp")
        if not finite_number(stamp) or stamp > now_seconds(self) + 0.02:
            self.report = None
            return
        if stamp < self.goal_stamp or stamp <= self.report_stamp:
            return
        self.report_stamp, self.report = stamp, report

    def on_dwa(self, msg):
        cmd = (msg.linear.x, msg.angular.z)
        now = now_seconds(self)
        if numeric_pair(cmd):
            self.dwa.cmd, self.dwa.cmd_stamp = cmd, now
        else:
            self.dwa = Source()

    def on_dwa_status(self, msg):
        self.dwa.status, self.dwa.status_stamp = msg.data, now_seconds(self)

    def emit(self, state, status, command=(0.0, 0.0)):
        msg = Twist()
        msg.linear.x, msg.angular.z = map(float, command)
        self.cmd.publish(msg)
        self.status.publish(string_message(status))
        active = {"FALLBACK": "dwa", "LEARNED": "learned"}.get(state, "stop")
        self.state_pub.publish(json_message({"mode": "learned", "state": state, "active": active,
                                             "goal_id": self.goal_id, "stamp": now_seconds(self)}))

    def clock_time(self):
        now = now_seconds(self)
        if self.last_clock is not None and now < self.last_clock:
            self.goal_stamp, self.goal_id = -math.inf, None
            self.reset_authority()
        self.last_clock = now
        return now

    def tick(self):
        now = self.clock_time()
        if self.terminal:
            self.emit(self.terminal, "REACHED" if self.terminal == "REACHED" else "STOPPED")
            return
        r = self.report
        # M3 checks the scan, odometry, their alignment, TF, goal and route on the same
        # acquisition. Check original stamps here too: a delayed report cannot extend freshness.
        sensors_fresh = (r and 0 <= now - self.report_stamp <= self.params["report_timeout"]
                         and all(finite_number(r.get(k)) and 0 <= now - r[k] <= 0.5
                                 for k in ("odom_stamp", "shield_scan_stamp")))
        if sensors_fresh and r.get("faults") == ["PATH_UNAVAILABLE"] and r.get("path_waiting") is True:
            self.fallback_since = self.healthy_since = None
            self.dwa = Source()
            if self.no_path_since is None:
                self.no_path_since = now
            elapsed = now - self.no_path_since
            if elapsed >= self.params["blocked_timeout_sec"]:
                self.terminal = "PATH_TIMEOUT"
            replan = elapsed >= self.params["replan_after_sec"] and self.terminal is None
            self.emit(self.terminal or ("WAIT_REPLAN" if replan else "WAIT_PATH"),
                      "DYNAMIC_BLOCKED" if replan else "STOPPED")
            return
        self.no_path_since = None
        if not (sensors_fresh and r.get("support_ready") is True and finite_number(r.get("path_stamp")) and self.goal_stamp <= r["path_stamp"] <= now
                and numeric_pair(r.get("odom_xy")) and numeric_pair(r.get("command"))):
            self.fallback_since = self.healthy_since = None
            self.dwa = Source()
            self.anchor = self.anchor_stamp = self.request_stamp = None
            self.emit("WAIT_INPUT", "STOPPED")
            return
        healthy = r.get("healthy") is True and r.get("faults") == []
        policy_fault = r.get("healthy") is False and r.get("faults") in (
            ["INFERENCE_DEADLINE"], ["INVALID_ACTION"])
        if not healthy and not policy_fault:
            self.fallback_since = self.healthy_since = None
            self.dwa = Source()
            self.anchor = self.anchor_stamp = self.request_stamp = None
            self.emit("WAIT_INPUT", "STOPPED")
            return
        if healthy and r.get("status") == "REACHED":
            self.terminal = "REACHED"
            self.emit(self.terminal, self.terminal)
            return

        position = r["odom_xy"]
        if self.anchor is None or math.dist(self.anchor, position) >= self.params["stall_distance_m"]:
            self.anchor, self.anchor_stamp, self.request_stamp = position, now, None
        elapsed = now - self.anchor_stamp
        if elapsed >= self.params["blocked_timeout_sec"]:
            self.terminal = "BLOCKED_TIMEOUT"
            self.emit(self.terminal, "STOPPED")
            return
        if elapsed >= self.params["replan_after_sec"] and self.request_stamp is None:
            self.request_stamp = now
        if self.request_stamp is not None and r["path_stamp"] <= self.request_stamp:
            # A* subscribes to this status and throttles duplicate requests itself.
            self.emit("WAIT_REPLAN", "DYNAMIC_BLOCKED")
            return

        if policy_fault:
            self.healthy_since = None
            if self.fallback_since is None:
                self.fallback_since = now
                self.dwa = Source()  # require a new DWA output after this handoff
        elif self.fallback_since is not None:
            if self.healthy_since is None:
                self.healthy_since = now
            if now - self.healthy_since >= self.params["recover_after_sec"]:
                self.fallback_since = self.healthy_since = None
        if self.fallback_since is not None:
            if (now - self.fallback_since < self.params["fallback_hold_sec"]
                    or not self.dwa.fresh(now, self.params["cmd_timeout"])):
                self.emit("FALLBACK_HOLD", "STOPPED")
            else:
                v, w = self.dwa.cmd
                # Upstream DWA status has no goal ID. Forward fresh navigation states
                # for A*, but arrival is certified by the goal-scoped pose check above.
                status = self.dwa.status
                if not status or status in {"GOAL_REACHED", "REACHED"} or now - self.dwa.status_stamp > 1.5:
                    status = "NORMAL"
                self.emit("FALLBACK", status, (min(max(v, 0.0), 0.45), min(max(w, -0.6), 0.6)))
            return
        v, w = r["command"]
        if not (r.get("status") == "NORMAL" and 0 <= v <= 0.45 and abs(w) <= 0.6):
            self.emit("INVALID_COMMAND", "STOPPED")
            return
        self.emit("LEARNED", "NORMAL", (v, w))


def main(args=None):
    rclpy.init(args=args)
    node = OperationalArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.emit("STOP", "STOPPED")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
