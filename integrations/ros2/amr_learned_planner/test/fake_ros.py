"""Minimal stand-ins for rclpy, ROS messages and tf2 so node wiring can be tested without ROS.

Only what the nodes use is implemented: parameters with overrides,
publishers that record messages, subscriptions and timers that tests call
directly, and a settable clock.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Callable


class Vector3:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0


class Quaternion:
    def __init__(self) -> None:
        self.x = self.y = self.z = 0.0
        self.w = 1.0


class Pose:
    def __init__(self) -> None:
        self.position, self.orientation = Vector3(), Quaternion()


class Header:
    def __init__(self) -> None:
        self.frame_id, self.stamp = "", None


class Twist:
    def __init__(self) -> None:
        self.linear, self.angular = Vector3(), Vector3()


class PoseStamped:
    def __init__(self) -> None:
        self.header, self.pose = Header(), Pose()


class Odometry:
    def __init__(self) -> None:
        self.header = Header()
        self.pose = SimpleNamespace(pose=Pose())
        self.twist = SimpleNamespace(twist=Twist())


class Path:
    def __init__(self) -> None:
        self.header, self.poses = Header(), []


class LaserScan:
    def __init__(self) -> None:
        self.header = Header()
        self.angle_min = self.angle_max = self.angle_increment = 0.0
        self.range_min, self.range_max = 0.0, 0.0
        self.ranges: list[float] = []


class String:
    def __init__(self) -> None:
        self.data = ""


class Clock:
    time = 0.0

    def now(self) -> Any:
        return SimpleNamespace(nanoseconds=int(Clock.time * 1e9), to_msg=lambda: None)


class Logger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def _log(self, level: str, message: str, **_: Any) -> None:
        self.messages.append((level, message))

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("info", message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> None:
        self._log("warn", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("error", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("debug", message, **kwargs)


class Publisher:
    def __init__(self, message_type: type, topic: str) -> None:
        self.message_type, self.topic, self.messages = message_type, topic, []

    def publish(self, message: Any) -> None:
        self.messages.append(message)


class Node:
    parameter_overrides: dict[str, Any] = {}

    def __init__(self, name: str) -> None:
        self.name = name
        self.publishers: dict[str, Publisher] = {}
        self.subscriptions: dict[str, list[Callable]] = {}
        self.timers: list[tuple[float, Callable]] = []
        self.logger = Logger()

    def declare_parameter(self, name: str, default: Any) -> Any:
        return SimpleNamespace(value=Node.parameter_overrides.get(name, default))

    def create_publisher(self, message_type: type, topic: str, qos: Any) -> Publisher:
        self.publishers[topic] = Publisher(message_type, topic)
        return self.publishers[topic]

    def create_subscription(self, message_type: type, topic: str, callback: Callable, qos: Any) -> None:
        self.subscriptions.setdefault(topic, []).append(callback)

    def create_timer(self, period: float, callback: Callable) -> None:
        self.timers.append((period, callback))

    def add_on_set_parameters_callback(self, callback: Callable) -> None:
        self.parameter_callback = callback

    def set_parameter(self, name: str, value: Any) -> Any:
        return self.parameter_callback([SimpleNamespace(name=name, value=value)])

    def get_logger(self) -> Logger:
        return self.logger

    def get_clock(self) -> Clock:
        return Clock()

    def destroy_node(self) -> None:
        pass

    # test helpers
    def deliver(self, topic: str, message: Any) -> None:
        for callback in self.subscriptions[topic]:
            callback(message)

    def fire(self, period: float) -> None:
        for timer_period, callback in self.timers:
            if abs(timer_period - period) < 1e-9:
                callback()


class TransformException(Exception):
    pass


class Buffer:
    transform: Any = None

    def lookup_transform(self, target: str, source: str, time: Any, timeout: Any = None) -> Any:
        if Buffer.transform is None:
            raise TransformException(f"no transform {source} -> {target}")
        return Buffer.transform


def install() -> bool:
    """Install the fakes unless real rclpy is importable. Returns True if installed."""
    try:
        import rclpy  # noqa: F401
        return False
    except ImportError:
        pass

    def module(name: str, **attributes: Any) -> types.ModuleType:
        created = types.ModuleType(name)
        created.__dict__.update(attributes)
        sys.modules[name] = created
        return created

    module("rclpy", init=lambda args=None: None, shutdown=lambda: None, spin=lambda node: None,
           spin_once=lambda node, timeout_sec=None: None, ok=lambda: True)
    module("rclpy.node", Node=Node)
    module("rclpy.qos", QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
           QoSReliabilityPolicy=SimpleNamespace(BEST_EFFORT="best_effort", RELIABLE="reliable"))
    module("rclpy.duration", Duration=lambda seconds=0.0: SimpleNamespace(seconds=seconds))
    module("rclpy.time", Time=lambda: SimpleNamespace())
    module("geometry_msgs")
    module("geometry_msgs.msg", Twist=Twist, PoseStamped=PoseStamped)
    module("nav_msgs")
    module("nav_msgs.msg", Odometry=Odometry, Path=Path)
    module("sensor_msgs")
    module("sensor_msgs.msg", LaserScan=LaserScan)
    module("std_msgs")
    module("std_msgs.msg", String=String)
    module("rcl_interfaces")
    module("rcl_interfaces.msg", SetParametersResult=lambda successful=True: SimpleNamespace(successful=successful))
    module("tf2_ros", Buffer=Buffer, TransformListener=lambda buffer, node: None,
           TransformException=TransformException)
    return True
