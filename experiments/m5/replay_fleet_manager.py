#!/usr/bin/env python3
"""Replay scripted ROS inputs through the unmodified pinned upstream deadlock manager."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

REVISION = "e8c9197ac2801c2770c83a7ee696b2b992d473b8"
SOURCE = "ros2_ws/src/amr_fleet/src/deadlock_manager.py"
SHA256 = "62ad7e3142c8469a806e6ef9d17d6c855c5c9390e4cd600ede5f41a085893692"

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations/ros2/amr_learned_planner/test"))
import fake_ros as ros


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_manager(upstream):
    source = subprocess.check_output(["git", "-C", str(upstream), "show", REVISION + ":" + SOURCE])
    require(hashlib.sha256(source).hexdigest() == SHA256, "upstream source hash differs")
    require(ros.install(), "run this offline replay with python3 -S to use its isolated ROS stubs")
    # Supply the parameter lookup and QoS names used by this upstream node.
    class ParameterNode(ros.Node):
        def declare_parameter(self, name, default):
            if not hasattr(self, "parameters"):
                self.parameters = {}
            value = super().declare_parameter(name, default)
            self.parameters[name] = value
            return value

        def get_parameter(self, name):
            return self.parameters[name]

    ros.Node.parameter_overrides = {"robots": ["amr1:6.6:37.0", "amr2:5.0:37.0"], "x_hold": 20.0}
    sys.modules["rclpy.node"].Node = ParameterNode
    sys.modules["rclpy"].time = sys.modules["rclpy.time"]
    qos = sys.modules["rclpy.qos"]
    qos.DurabilityPolicy = SimpleNamespace(TRANSIENT_LOCAL=1)
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST=1)
    qos.ReliabilityPolicy = qos.QoSReliabilityPolicy
    sys.modules["std_msgs.msg"].Int32MultiArray = SimpleNamespace
    context = {"__name__": "pinned_deadlock_manager", "__file__": SOURCE}
    exec(compile(source, SOURCE, "exec"), context)
    return context["DeadlockManager"]()


def replay(manager):
    poses = {"amr1/base_footprint": (3.6, 22.), "amr2/base_footprint": (2., 22.)}

    def transform(target, source, query):
        require(target == "map", "manager requested a different TF frame")
        x, y = poses[source]
        return SimpleNamespace(transform=SimpleNamespace(translation=SimpleNamespace(x=x, y=y),
                                                         rotation=SimpleNamespace(z=0., w=1.)))

    manager.tf_buffer.lookup_transform = transform
    goals = {"amr2": (8., 22.), "amr1": (10., 22.)}
    publications, transitions = [], []
    previous = None
    # Both robots face east. Move the blocker after six seconds; scripted loser
    # motion begins at 9.5 seconds, only after the manager has restored its mission.
    for tick in range(121):
        t = tick / 10
        ros.Clock.time = t
        for robot, start, speed, depart in (("amr1", 3.6, .45, 6.), ("amr2", 2., .35, 9.5)):
            poses[robot + "/base_footprint"] = (start + max(0., t - depart) * speed, 22.)
            odom = ros.Odometry()
            odom.twist.twist.linear.x = speed if t >= depart else 0.
            manager.deliver(f"/{robot}/odometry/filtered", odom)
            status = ros.String()
            status.data = "NORMAL" if t >= depart else "STOPPED"
            manager.deliver(f"/{robot}/dwa/status", status)
        if tick in (0, 60):
            robot = "amr2" if tick == 0 else "amr1"
            msg = ros.PoseStamped()
            msg.header.frame_id = "map"
            msg.pose.position.x, msg.pose.position.y = goals[robot]
            manager.deliver(f"/{robot}/goal_pose", msg)
        if tick == 95:
            require(publications and publications[-1]["xy"] == list(goals["amr2"]),
                    "manager must restore the mission before the fixture moves the yielding robot")
        if tick % 2 == 0:
            manager.fire(.2)
            for robot in goals:
                topic = f"/{robot}/goal_pose"
                messages = manager.publishers[topic].messages
                while messages:
                    msg = messages.pop(0)
                    stamp = msg.header.stamp
                    require(msg.header.frame_id == "map", "replacement frame must be map")
                    publications.append({"topic": topic, "stamp": stamp.sec + stamp.nanosec / 1e9,
                                         "xy": [msg.pose.position.x, msg.pose.position.y]})
                    manager.deliver(topic, msg)  # Own publications are also seen by its goal subscription.
        if tick % 5 == 0:
            manager.fire(.5)
            status = json.loads(manager.publishers["/fleet/deadlock_status"].messages.pop().data)
            leader, yielding = status["robots"]["amr1"], status["robots"]["amr2"]
            require(status["prio_source"] == "default" and leader["prio"] > yielding["prio"]
                    and leader["fsm"] == "NORMAL", "unexpected priority or leader override")
            if previous != yielding["fsm"]:
                transitions.append({"t": status["t"], "amr2": yielding["fsm"], "blocker": yielding["blocked_by"]})
                previous = yielding["fsm"]

    require([transition["amr2"] for transition in transitions] == ["NORMAL", "HOLD", "RESUMING", "NORMAL"],
            "did not observe the expected upstream FSM sequence on the status topic")
    require(transitions[1]["blocker"] == "amr1", "hold did not name the expected blocker")
    targets = []
    for event in publications:
        require(event["topic"] == "/amr2/goal_pose", "upstream overrode the priority robot")
        if not targets or targets[-1] != event["xy"]:
            targets.append(event["xy"])
    require(targets == [[2., 22.], list(goals["amr2"])], "manager failed to HOLD and restore original goal")
    require(all(a["stamp"] < b["stamp"] for a, b in zip(publications, publications[1:])),
            "replacement goals need fresh increasing timestamps")
    return {"scope": "upstream callbacks with scripted TF, odometry, status and ROS clock",
            "upstream_revision": REVISION, "source_sha256": SHA256, "duration_s": 12,
            "manager_parameters": ros.Node.parameter_overrides,
            "fixture": {"amr1": {"start": [3.6, 22.], "mission_at_s": 6., "goal": goals["amr1"],
                                 "move_at_s": 6., "east_mps": .45},
                        "amr2": {"start": [2., 22.], "mission_at_s": 0., "goal": goals["amr2"],
                                 "move_at_s": 9.5, "east_mps": .35}},
            "fsm_samples_at_transitions": transitions, "manager_goal_publications": publications,
            "status": "verified"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="opticore-amr git checkout")
    args = parser.parse_args()
    print(json.dumps(replay(load_manager(args.upstream)), indent=2))
