#!/usr/bin/env python3
"""Run a paired scenario set in Gazebo and record it like Lite (ROADMAP M6).

Sends the legs of ``packages/amr_lite/configs/paired/<set>.yaml`` as
``/goal_pose`` in the map frame, one after another, records each leg as an
episode named ``WH-<set>-<NN>`` and writes ``output_csv`` when finished.
"""
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

from amr_learned_planner.core.recorder import EpisodeRecorder
from amr_learned_planner.core.sequence import GoalSequence
from amr_learned_planner.ros_common import (
    RECORDER_TOPICS,
    RELIABLE_QOS,
    declare_parameters,
    now_seconds,
    subscribe_recorder,
)

NODE_PARAMETERS = {
    **RECORDER_TOPICS,
    "paired_set": "warehouse_v1",
    "runs": 1,
    "label": "ros",
    "output_csv": "paired_episodes.csv",
    "start_delay_sec": 5.0,
    "settle_sec": 2.0,
    "on_failure": "abort_run",
    "global_frame": "map",
    "robot_radius": 0.36,
    "collision_clearance": 0.0,
    "lidar_mount_x": 0.25,
    "lidar_mount_y": 0.0,
    "lidar_mount_yaw": 0.0,
}


class GoalSequenceRunnerNode(Node):
    def __init__(self) -> None:
        super().__init__("goal_sequence_runner")
        self.p = declare_parameters(self, NODE_PARAMETERS)
        from amr_lite.envs.warehouse_scenarios import paired_spec

        spec = paired_spec(str(self.p["paired_set"]))
        self.sequence = GoalSequence(spec, int(self.p["runs"]), float(self.p["settle_sec"]), str(self.p["on_failure"]))
        self.sequence.ready_at = now_seconds(self) + float(self.p["start_delay_sec"])
        self.recorder = EpisodeRecorder(
            str(self.p["label"]), float(self.p["robot_radius"]), float(self.p["collision_clearance"]),
            self.sequence.timeout_sec,
            (float(self.p["lidar_mount_x"]), float(self.p["lidar_mount_y"]), float(self.p["lidar_mount_yaw"])))
        subscribe_recorder(self, self.recorder, self.p, on_goal=False)
        self._goal_pub = self.create_publisher(PoseStamped, self.p["goal_pose_topic"], RELIABLE_QOS)
        self._rows = 0
        self.finished = False
        self.create_timer(0.2, self._on_timer)
        self.get_logger().info(f"paired set {spec['name']}: {len(spec['legs'])} legs x {self.sequence.runs} runs")

    def _on_timer(self) -> None:
        now = now_seconds(self)
        self.recorder.tick(now)
        if len(self.recorder.rows) != self._rows:
            self._rows = len(self.recorder.rows)
            row = self.recorder.rows[-1]
            self.sequence.on_result(now, bool(row["success"]))
            self.recorder.write_csv(self.p["output_csv"])
            self.get_logger().info(f"{row['scenario_id']} run {row['seed']}: success={row['success']} "
                                   f"failure={row['failure_type'] or '-'}")
        goal = self.sequence.next_goal(now)
        if goal is not None:
            self.recorder.start(now, (goal.x, goal.y), goal.scenario_id, goal.run)
            message = PoseStamped()
            message.header.frame_id = self.p["global_frame"]
            message.header.stamp = self.get_clock().now().to_msg()
            message.pose.position.x, message.pose.position.y = goal.x, goal.y
            message.pose.orientation.z, message.pose.orientation.w = math.sin(goal.yaw / 2), math.cos(goal.yaw / 2)
            self._goal_pub.publish(message)
            self.get_logger().info(f"goal {goal.scenario_id} run {goal.run}: ({goal.x:.2f}, {goal.y:.2f})")
        if self.sequence.done and not self.finished:
            self.finished = True
            path = self.recorder.write_csv(self.p["output_csv"])
            self.get_logger().info(f"paired set complete: {len(self.recorder.rows)} episodes, "
                                   f"{len(self.sequence.skipped)} legs skipped -> {path}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GoalSequenceRunnerNode()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.recorder.write_csv(node.p["output_csv"])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
