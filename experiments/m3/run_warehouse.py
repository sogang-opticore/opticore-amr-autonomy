#!/usr/bin/env python3
"""Run one bounded, train-only M3 warehouse shadow or learned diagnostic."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "m2"))
from mission_spec import digest, resolve
from run_mission import capture_params, check_evidence, child, ownership_error, parameter_response, seconds, startup, write_status

ROOT = Path(__file__).resolve().parents[2]
DRIVER = "/m3_mission_driver"
CASE_ID = "m3-short-train-dock1"
TIME_LIMIT = 30.0


def diagnostic_case():
    case = resolve("train-static")
    return {**case, "id": CASE_ID, "goal": [case["start"][0] + 4.0, case["start"][1]],
            "timeout_sec": TIME_LIMIT, "parent_mission": case["id"]}


def spin_until(node, launch, condition, wall_timeout):
    import rclpy

    deadline = time.monotonic() + wall_timeout
    while time.monotonic() < deadline:
        if launch.poll() is not None:
            raise RuntimeError("warehouse exited; inspect launch.log")
        rclpy.spin_once(node, timeout_sec=0.05)
        reason = condition()
        if reason is None:
            return
    raise RuntimeError(f"warehouse deadline: {reason}")


def hold(node, enabled):
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter

    result = parameter_response(node, SetParameters, "/amr1/m3_policy/set_parameters",
                                SetParameters.Request(parameters=[Parameter("hold", value=enabled).to_parameter_msg()]))
    if len(result.results) != 1 or not result.results[0].successful:
        raise RuntimeError("M3 hold command was rejected")


def run_world(out, args, case, run, stack):
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=[])
    stack.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)
    node = Node(DRIVER.lstrip("/"), parameter_overrides=[Parameter("use_sim_time", value=True)])
    stack.callback(node.destroy_node)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if any(ns == "/amr1" for _, ns in node.get_node_names_and_namespaces()):
        raise RuntimeError("ROS domain already has an amr1")
    tf = Buffer()
    listener = TransformListener(tf, node)
    goal_pub = node.create_publisher(PoseStamped, "/amr1/goal_pose", 10)
    witness = stack.enter_context((out / "control.jsonl").open("x", buffering=1))
    seen = {"heartbeat": None, "scan_stamp": None, "ground_truth": None}
    signals = {}

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def record(kind, **payload):
        print(json.dumps({"kind": kind, "time": now(), **payload}, allow_nan=False), file=witness)

    def contact(msg):
        sample = json.loads(msg.data)
        seen["heartbeat"] = float(sample["stamp"]), sample["collision"]
        record("contact", sample=sample)

    def motion(msg):
        seen["ground_truth"] = msg
        p, v = msg.pose.pose.position, msg.twist.twist.linear
        signals["pose"] = (p.x, p.y, math.hypot(v.x, v.y))
        record("motion", stamp=seconds(msg.header.stamp), xy=[p.x, p.y], speed=signals["pose"][2])

    def command(kind, msg):
        signals[kind] = now(), (msg.linear.x, msg.angular.z)
        record(kind, physical=signals[kind][1])

    def report(kind, msg):
        signals[kind] = json.loads(msg.data)
        record(kind, payload=signals[kind])

    node.create_subscription(String, "/amr1/m2/contacts", contact, qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/amr1/lidar",
                             lambda m: seen.update(scan_stamp=seconds(m.header.stamp)), qos_profile_sensor_data)
    node.create_subscription(Odometry, "/amr1/ground_truth", motion, qos_profile_sensor_data)
    for kind, topic in (("cmd", "cmd_vel"), ("candidate", "learned_planner/cmd_vel"),
                        ("dwa", "dwa_fallback/cmd_vel")):
        node.create_subscription(Twist, "/amr1/" + topic, lambda msg, k=kind: command(k, msg), 10)
    for kind, topic in (("arbiter", "cmd_arbiter/state"), ("health", "learned_planner/health")):
        node.create_subscription(String, "/amr1/" + topic, lambda msg, k=kind: report(k, msg), 10)

    x, y = case["spawn_world"]
    csv_path = out / "episodes.csv"
    cmd = ["ros2", "launch", "amr_learned_planner", "m2_baseline.launch.py", "planner:=b1",
           f"command:={args.mode}", f"spawn_x:={x}", f"spawn_y:={y}", f"record_bag:={out / 'bag'}",
           f"record_csv:={csv_path}", f"scenario_id:={CASE_ID}", f"run_id:={run['run_id']}",
           f"timeout_sec:={TIME_LIMIT}", f"m3_bundle:={args.bundle}", f"m3_trace:={out / 'replay.jsonl'}"]
    run["commands"] = [cmd]
    write_status(out, run)
    launch_log = out / "launch.log"
    launch = stack.enter_context(child(cmd, launch_log))

    def readiness():
        why = ownership_error(node, "b1", DRIVER) or startup(node, goal_pub, tf=tf, case=case, **seen)
        if why:
            return why
        if goal_pub.get_subscription_count() < 4:
            return "M3 goal input has not matched"
        log = launch_log.read_text(errors="replace")
        # The pinned upstream A* discards goals before its first occupancy map callback.
        if not any("[amr1.astar_planner]: 맵 수신:" in line for line in log.splitlines()):
            return "A* has not acknowledged its initial map"
        if "health" not in signals or "arbiter" not in signals:
            return "M3 node and command arbiter have not reported"
        if "Subscribed to topic '/amr1/goal_pose'" not in log:
            return "rosbag has not subscribed to the goal"
        return None

    spin_until(node, launch, readiness, 180)
    run["effective_parameter_sha256"] = capture_params(node, out, "b1", extra_nodes=("m3_policy",))
    spin_until(node, launch, readiness, 10)
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x, msg.pose.position.y = case["goal"]
    msg.pose.orientation.w = 1.0
    started = seconds(msg.header.stamp)
    run["goal_stamp"] = started
    write_status(out, run)
    record("goal", stamp=started, map_xy=case["goal"])
    goal_pub.publish(msg)

    phase = "first_motion"
    last = started
    next_graph_check = started
    stopped_since = None
    resume_x = None
    observed = {}

    def observe():
        nonlocal phase, last, next_graph_check, stopped_since, resume_x
        current = now()
        if current + 1e-6 < last:
            raise RuntimeError("simulation clock moved backwards")
        last = current
        if current >= next_graph_check:
            why = ownership_error(node, "b1", DRIVER)
            if why:
                hold(node, True)
                raise RuntimeError(why)
            next_graph_check = current + 0.5
        if seen["heartbeat"] is None or current - seen["heartbeat"][0] > 0.5 or seen["heartbeat"][1]:
            hold(node, True)
            raise RuntimeError("contact watchdog failed or physical contact occurred")
        px, py, speed = signals["pose"]
        forward = px - x
        if abs(py - y) > 1.0 or not -0.25 <= forward <= 4.8:
            hold(node, True)
            raise RuntimeError("robot left the bounded M3 diagnostic corridor")
        active = signals.get("arbiter", {}).get("active")
        health = signals.get("health", {})
        command_stamp, final = signals.get("cmd", (0, (0, 0)))
        if args.mode == "learned":
            if active == "dwa" or not (0.0 <= final[0] <= 0.45 and abs(final[1]) <= 0.6):
                hold(node, True)
                raise RuntimeError("learned authority or command cap violated")
            if (phase == "first_motion" and active == "learned" and health.get("healthy")
                    and final[0] >= 0.035 and forward >= 0.15 and speed >= 0.035):
                observed["learned_motion"] = {"stamp": current, "x_delta_m": forward, "speed_mps": speed}
                hold(node, True)
                record("hold_requested")
                phase = "holding"
            elif phase == "holding":
                is_stopped = (active == "stop" and "DIAGNOSTIC_HOLD" in health.get("faults", [])
                              and current - command_stamp <= 0.25 and final == (0, 0) and speed < 0.02)
                stopped_since = (stopped_since if stopped_since is not None else current) if is_stopped else None
                if stopped_since is not None and current - stopped_since >= 0.4:
                    observed["verified_stop"] = {"stamp": current, "x_delta_m": forward, "speed_mps": speed,
                                                 "dwell_s": current - stopped_since}
                    resume_x = px
                    hold(node, False)
                    record("hold_released")
                    phase = "resume"
            elif (phase == "resume" and active == "learned" and health.get("healthy") and px - resume_x >= 0.10
                  and final[0] >= 0.035 and speed >= 0.035):
                observed["resumed_motion"] = {"stamp": current, "x_delta_m": px - resume_x, "speed_mps": speed}
                phase = "verified"
        elif (active == "dwa" and health.get("healthy") and health.get("action") is not None
              and forward >= 0.15 and speed >= 0.035):
            observed.setdefault("shadow_motion", {"stamp": current, "x_delta_m": forward, "speed_mps": speed})

        if csv_path.is_file():
            with csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            if rows:
                if len(rows) != 1 or rows[0]["run_id"] != run["run_id"]:
                    raise RuntimeError("recorder emitted an unexpected run identity")
                hold(node, True)
                run.update(control_witness=observed, outcome=rows[0], final_world_xy=[px, py])
                if args.mode == "learned" and phase != "verified" or args.mode == "shadow" and not observed:
                    raise RuntimeError(f"insufficient M3 authority/motion evidence: {phase}, {observed}")
                if rows[0]["collision"] != "0" or rows[0]["collision_known"] != "1":
                    raise RuntimeError("M3 diagnostic did not maintain verified clear contact coverage")
                return None
        return f"{phase}, elapsed {current - started:.1f} sim s"

    spin_until(node, launch, observe, 3 * TIME_LIMIT + 15)


def verify_bag(out):
    check_evidence(out)
    metadata = yaml.safe_load((out / "bag" / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    counts = {row["topic_metadata"]["name"]: row["message_count"] for row in metadata["topics_with_message_count"]}
    required = ("global_path", "dwa_fallback/cmd_vel", "cmd_arbiter/state",
                "learned_planner/cmd_vel", "learned_planner/health", "lidar", "odometry/filtered")
    if counts["/amr1/goal_pose"] != 1 or any(not counts.get("/amr1/" + topic) for topic in required):
        raise RuntimeError("M3 control bag lacks the unique goal or required policy/arbiter signals")
    if not (out / "replay.jsonl").stat().st_size:
        raise RuntimeError("M3 replay inputs missing")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("shadow", "learned"))
    parser.add_argument("--bundle", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--out", required=True, type=lambda value: Path(value).resolve())
    parser.add_argument("--domain", type=int, default=86, choices=range(102), metavar="0..101")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    case = diagnostic_case()
    token = uuid.uuid4().hex
    run = {"run_id": token, "status": "running", "mode": args.mode, "case": case, "ros_domain_id": args.domain,
           "ign_partition": "m3_" + token}
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((Path(tempfile.gettempdir()) / f"opticore-m2-domain-{args.domain}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1", IGN_PARTITION=run["ign_partition"])
            from ament_index_python.packages import get_package_share_directory

            share = Path(get_package_share_directory("amr_bringup"))
            if digest(share / "maps" / "warehouse_map.pgm") != case["map_sha256"]:
                raise RuntimeError("installed warehouse map differs from the diagnostic case")
            run.update(
                bundle_sha256={name: digest(args.bundle / name) for name in ("bundle.json", "weights.npz", "golden.npz")},
                autonomy_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
            )
            run_world(args.out, args, case, run, stack)
        run.update(status="verified", bag_counts=verify_bag(args.out))
    except BaseException as error:
        run.update(status="invalid", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_status(args.out, run)


if __name__ == "__main__":
    main()
