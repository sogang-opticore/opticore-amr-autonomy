#!/usr/bin/env python3
"""Single publisher of ``cmd_vel`` and the effective ``dwa/status``.

Forwards the DWA (``dwa``/``shadow`` modes) or the learned planner with DWA
fallback (``learned`` mode); see ``core/arbiter.py`` for the state machine.
"""
from __future__ import annotations

import json
from dataclasses import fields

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String

from amr_learned_planner.core.arbiter import ArbiterConfig, ArbiterOutput, CommandArbiter
from amr_learned_planner.core.planner_core import coerce
from amr_learned_planner.ros_common import (
    RELIABLE_QOS,
    declare_dataclass_parameters,
    declare_parameters,
    json_message,
    now_seconds,
    string_message,
)

NODE_PARAMETERS = {
    "learned_cmd_topic": "learned_planner/cmd_vel",
    "learned_status_topic": "learned_planner/status",
    "learned_health_topic": "learned_planner/health",
    "dwa_cmd_topic": "dwa_fallback/cmd_vel",
    "dwa_status_topic": "dwa_fallback/status",
    "cmd_topic": "cmd_vel",
    "status_topic": "dwa/status",
    "state_topic": "cmd_arbiter/state",
    "rate": 20.0,
    "state_period": 1.0,
}


class CmdArbiterNode(Node):
    def __init__(self) -> None:
        super().__init__("cmd_arbiter")
        values = declare_dataclass_parameters(self, ArbiterConfig())
        self.p = declare_parameters(self, NODE_PARAMETERS)
        defaults = ArbiterConfig()
        config = ArbiterConfig(**{item.name: coerce(getattr(defaults, item.name), values[item.name])
                                  for item in fields(ArbiterConfig)})
        self.arbiter = CommandArbiter(config)
        self.get_logger().info(f"cmd_arbiter mode={config.mode} state={self.arbiter.state}")
        self._cmd_pub = self.create_publisher(Twist, self.p["cmd_topic"], 10)
        self._status_pub = self.create_publisher(String, self.p["status_topic"], 10)
        self._state_pub = self.create_publisher(String, self.p["state_topic"], 10)
        for source in ("learned", "dwa"):
            self.create_subscription(Twist, self.p[f"{source}_cmd_topic"],
                                     lambda message, name=source: self._on_cmd(name, message), RELIABLE_QOS)
            self.create_subscription(String, self.p[f"{source}_status_topic"],
                                     lambda message, name=source: self._on_status(name, message), RELIABLE_QOS)
        self.create_subscription(String, self.p["learned_health_topic"], self._on_health, RELIABLE_QOS)
        self.create_timer(1.0 / max(float(self.p["rate"]), 1.0), self._on_tick)
        self.create_timer(float(self.p["state_period"]), self._publish_state)

    def _emit(self, output: ArbiterOutput) -> None:
        if output.cmd is not None:
            command = Twist()
            command.linear.x, command.angular.z = output.cmd
            self._cmd_pub.publish(command)
        for status in output.statuses:
            self._status_pub.publish(string_message(status))
        if output.transition is not None:
            self.get_logger().warn(f"arbiter {output.transition['from']} -> {output.transition['to']}: "
                                   f"{output.transition['reason']}")
            self._publish_state()

    def _on_cmd(self, source: str, message: Twist) -> None:
        self._emit(self.arbiter.on_cmd(source, now_seconds(self), message.linear.x, message.angular.z))

    def _on_status(self, source: str, message: String) -> None:
        self._emit(self.arbiter.on_status(source, now_seconds(self), message.data))

    def _on_health(self, message: String) -> None:
        try:
            report = json.loads(message.data)
        except json.JSONDecodeError:
            report = {"healthy": False, "faults": ["HEALTH_PARSE_ERROR"]}
        self.arbiter.on_health(now_seconds(self), bool(report.get("healthy", False)), report.get("faults"))

    def _on_tick(self) -> None:
        self._emit(self.arbiter.tick(now_seconds(self)))

    def _publish_state(self) -> None:
        self._state_pub.publish(json_message(self.arbiter.snapshot(now_seconds(self))))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CmdArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
