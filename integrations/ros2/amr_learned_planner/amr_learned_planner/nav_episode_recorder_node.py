#!/usr/bin/env python3
"""Passive recorder: one CSV row per /goal_pose, whatever planner is driving."""
from __future__ import annotations

import rclpy
from rclpy.node import Node

from amr_learned_planner.core.recorder import EpisodeRecorder
from amr_learned_planner.ros_common import RECORDER_TOPICS, declare_parameters, now_seconds, subscribe_recorder

NODE_PARAMETERS = {
    **RECORDER_TOPICS,
    "label": "ros",
    "output_csv": "nav_episodes.csv",
    "robot_radius": 0.36,
    "collision_clearance": 0.0,
    "timeout_sec": 120.0,
    "lidar_mount_x": 0.25,
    "lidar_mount_y": 0.0,
    "lidar_mount_yaw": 0.0,
}


class NavEpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("nav_episode_recorder")
        self.p = declare_parameters(self, NODE_PARAMETERS)
        self.recorder = EpisodeRecorder(
            str(self.p["label"]), float(self.p["robot_radius"]), float(self.p["collision_clearance"]),
            float(self.p["timeout_sec"]),
            (float(self.p["lidar_mount_x"]), float(self.p["lidar_mount_y"]), float(self.p["lidar_mount_yaw"])))
        subscribe_recorder(self, self.recorder, self.p, on_goal=True)
        self._written = 0
        self.create_timer(0.5, self._on_timer)

    def _on_timer(self) -> None:
        self.recorder.tick(now_seconds(self))
        if len(self.recorder.rows) != self._written:
            self._written = len(self.recorder.rows)
            path = self.recorder.write_csv(self.p["output_csv"])
            row = self.recorder.rows[-1]
            self.get_logger().info(f"episode {row['scenario_id']}: success={row['success']} "
                                   f"failure={row['failure_type'] or '-'} time={row['episode_time']:.1f}s -> {path}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = NavEpisodeRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.recorder.write_csv(node.p["output_csv"])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
