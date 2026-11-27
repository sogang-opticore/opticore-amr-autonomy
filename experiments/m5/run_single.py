#!/usr/bin/env python3
"""Run a development-only M5 trial in a fresh, instrumented Fortress world."""
from __future__ import annotations

import argparse
from collections import Counter
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/m3"))
from run_warehouse import spin_until
from mission_spec import digest, resolve
from run_mission import capture_params, check_evidence, child, ownership_error, parameter_response, seconds, startup, write_status
from scene import Actor, GazeboActor, world_xy

DRIVER = "/m5_single_driver"
CASES = ("baseline", "blockage", "sequence", "transport", "faults")


def scenario(name, cycles):
    parent = resolve("train-static")
    x, y = parent["start"]
    short = [x + 4.0, y]
    goals = [short]
    if name == "sequence":
        goals = [short, [x + 6.0, y], parent["start"]] * cycles
    elif name == "transport":
        goals = [parent["goal"], parent["start"]] * cycles
    elif name == "faults":
        goals = [parent["goal"], short]
    actor = {"knots": [[0, x + 1.6, y]], "size": [0.5, 1.8, 1.0], "remove_at": 8.0}
    return {**parent, "id": f"m5-train-{name}", "parent_mission": parent["id"], "goals": goals,
            "actor": actor if name == "blockage" else None, "timeout_sec": 180.0 if name == "transport" else 45.0}


def candidate_identity(bundle):
    manifest = json.loads((bundle / "bundle.json").read_text())
    seed = manifest["provenance"]["seed"]
    score = json.loads((ROOT / "experiments/m4/followup/mixed.json").read_text())
    approved = {entry["seed"]: entry["bundle_sha256"] for entry in score["training"]}
    hashes = {n: digest(bundle / n) for n in ("bundle.json", "weights.npz", "golden.npz")}
    if not score["mixed"]["gate"]["eligible"] or approved.get(seed) != hashes["bundle.json"]:
        raise ValueError("expected an M4 candidate from the promoted three-seed configuration")
    if any(hashes[name] != h for name, h in manifest["files"].items()):
        raise ValueError("candidate files differ from bundle manifest")
    return seed, hashes


def stopped_sample(t, signal):
    """A stationary base needs a recent zero command and independent ground truth."""
    return (signal.get("cmd") == (0.0, 0.0)
            and 0 <= t - signal.get("cmd_stamp", -math.inf) <= 0.25
            and -0.02 <= t - signal.get("motion_stamp", -math.inf) <= 0.25
            and signal.get("speed", math.inf) < 0.02 and abs(signal.get("yaw_rate", math.inf)) < 0.03)


def set_policy_parameter(node, key, value):
    from rcl_interfaces.srv import SetParameters
    from rclpy.parameter import Parameter

    request = SetParameters.Request(parameters=[Parameter(key, value=value).to_parameter_msg()])
    result = parameter_response(node, SetParameters, "/amr1/m3_policy/set_parameters", request)
    if len(result.results) != 1 or not result.results[0].successful:
        raise RuntimeError(f"diagnostic parameter {key} was rejected: {result.results}")


def run_world(args, case, run, stack):
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry, Path as NavPath
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    out = args.out
    rclpy.init(args=[])
    stack.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)
    node = Node(DRIVER.lstrip("/"), parameter_overrides=[Parameter("use_sim_time", value=True)])
    stack.callback(node.destroy_node)
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if any(ns == "/amr1" for _, ns in node.get_node_names_and_namespaces()):
        raise RuntimeError("another amr1 already occupies this ROS domain")
    tf = Buffer()
    listener = TransformListener(tf, node)
    goal = node.create_publisher(PoseStamped, "/amr1/goal_pose", 10)
    cancel = node.create_publisher(String, "/amr1/cancel_goal", 10)
    log = stack.enter_context((out / "control.jsonl").open("x", buffering=1))
    seen = {"heartbeat": None, "scan_stamp": None, "ground_truth": None}
    signal = {}
    states = Counter()
    faults = Counter()
    count = Counter()

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def record(kind, **kwargs):
        print(json.dumps({"kind": kind, "time": now(), **kwargs}, allow_nan=False), file=log)

    def contact(msg):
        data = json.loads(msg.data)
        if data.get("source") != "gazebo_contacts" or type(data.get("collision")) is not bool:
            return
        seen["heartbeat"] = data["stamp"], data["collision"]
        if data.get("actor_pose_valid") is True:
            signal["actor"] = data["stamp"], data.get("actor_world_xy")
        record("contact", sample=data)

    def motion(msg):
        seen["ground_truth"] = msg
        p, v = msg.pose.pose.position, msg.twist.twist.linear
        signal["xy"] = (p.x, p.y)
        signal["speed"] = math.hypot(v.x, v.y)
        signal["yaw_rate"] = msg.twist.twist.angular.z
        signal["motion_stamp"] = seconds(msg.header.stamp)
        record("motion", stamp=signal["motion_stamp"], xy=signal["xy"], speed=signal["speed"],
               yaw_rate=signal["yaw_rate"])

    def command(msg):
        v, w = msg.linear.x, msg.angular.z
        if not (0.0 <= v <= 0.45 and abs(w) <= 0.6):
            raise ValueError(f"final command outside operational cap: {v}, {w}")
        signal["cmd"] = (v, w)
        signal["cmd_stamp"] = now()
        record("cmd", physical=[v, w])

    def report(kind, msg):
        data = json.loads(msg.data)
        signal[kind] = data
        if data.get("goal_id") and data["goal_id"] == signal.get("goal_id"):
            if kind == "arbiter":
                states[data["state"]] += 1
            elif kind == "health":
                faults.update(data["faults"])
        record(kind, payload=data)

    def route(msg):
        count["paths"] += 1
        end = msg.poses[-1].pose.position if msg.poses else None
        record("path", stamp=seconds(msg.header.stamp), frame=msg.header.frame_id,
               endpoint=[end.x, end.y] if end else None)

    ns = "/amr1/"
    node.create_subscription(Odometry, ns+"ground_truth", motion, qos_profile_sensor_data)
    node.create_subscription(String, ns+"m2/contacts", contact, qos_profile_sensor_data)
    node.create_subscription(LaserScan, ns+"lidar", lambda m: seen.update(scan_stamp=seconds(m.header.stamp)),
                             qos_profile_sensor_data)
    node.create_subscription(Twist, ns+"cmd_vel", command, 10)
    node.create_subscription(NavPath, ns+"global_path", route, 10)
    for k, topic in (("health", "learned_planner/health"), ("arbiter", "cmd_arbiter/state")):
        node.create_subscription(String, ns+topic, lambda msg, kind=k: report(kind, msg), 10)

    csv_path = out / "episodes.csv"
    x, y = case["spawn_world"]
    cmd = ["ros2", "launch", "amr_learned_planner", "m2_baseline.launch.py", "planner:=b1",
           "command:=operational", f"spawn_x:={x}", f"spawn_y:={y}", f"record_bag:={out / 'bag'}",
           f"record_csv:={csv_path}", f"scenario_id:={case['id']}", f"run_id:={run['run_id']}",
           f"timeout_sec:={case['timeout_sec']}", f"seed:={run['training_seed']}",
           f"m3_bundle:={args.bundle}", f"m3_trace:={out / 'replay.jsonl'}"]
    run["commands"] = [cmd]
    write_status(out, run)
    launch_log = out / "launch.log"
    launch = stack.enter_context(child(cmd, launch_log))

    def ready():
        reason = ownership_error(node, "b1", DRIVER) or startup(node, goal, tf=tf, case=case, **seen)
        if reason:
            return reason
        text = launch_log.read_text(errors="replace")
        if "[amr1.astar_planner]: 맵 수신:" not in text:
            return "waiting for A* first map"
        goal_listeners = {e.node_name for e in node.get_subscriptions_info_by_topic(ns+"goal_pose")
                          if e.node_namespace == "/amr1"}
        if ("health" not in signal or "arbiter" not in signal or
                not {"m3_policy", "cmd_arbiter"}.issubset(goal_listeners)):
            return "waiting for matched policy and operational arbiter"
        if args.case == "faults" and not any(
            e.node_name == "cmd_arbiter" and e.node_namespace == "/amr1"
            for e in node.get_subscriptions_info_by_topic(ns+"cancel_goal")
        ):
            return "waiting for goal-cancel subscription"
        if "Subscribed to topic '/amr1/goal_pose'" not in text:
            return "waiting for rosbag goal subscription"
        return None

    spin_until(node, launch, ready, 180)
    run["effective_parameter_sha256"] = capture_params(node, out, "b1", extra_nodes=("m3_policy",))
    spin_until(node, launch, ready, 15)
    obstacle = None
    if case["actor"]:
        trajectory = Actor.from_dict(case["actor"])
        obstacle = GazeboActor(trajectory)
        target = world_xy(trajectory.position(0))
        def placed():
            stamp, position = signal.get("actor", (0, None))
            if position is None or now() - stamp > 0.5 or math.dist(target, position) > 0.08:
                return "awaiting physical obstacle pose"
            return None
        spin_until(node, launch, placed, 15)
        record("actor_placed", world_xy=target)

    run["goals"] = []
    run["legs"] = []
    for index, destination in enumerate(case["goals"]):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = map(float, destination)
        msg.pose.orientation.w = 1.0
        gid = f"map:{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
        start = seconds(msg.header.stamp)
        run["goals"].append({"id": gid, "goal": destination, "stamp": start})
        states.clear()
        faults.clear()
        count.clear()
        signal["goal_id"] = gid
        record("goal", goal_id=gid, map_xy=destination, stamp=start)
        goal.publish(msg)
        write_status(out, run)
        begin_xy = signal["xy"]
        previous, next_graph = start, start
        phase, dwell, anchor = "drive", None, None
        events = {}
        actor_remove_request = None
        phase_start = start
        row = None
        def event(kind, **data):
            events[kind] = {"stamp": now(), "world_xy": list(signal["xy"]), **data}
            record(kind, **events[kind])
        def observe():
            nonlocal phase, dwell, anchor, actor_remove_request, previous, next_graph, phase_start, row
            t = now()
            if t < previous - 1e-6:
                raise RuntimeError("simulation clock went backwards")
            previous = t
            elapsed = t - start
            if t >= next_graph:
                reason = ownership_error(node, "b1", DRIVER)
                if reason:
                    raise RuntimeError(reason)
                next_graph = t + 0.5
            if seen["heartbeat"] is None or t - seen["heartbeat"][0] > 0.55:
                raise RuntimeError("lost contact heartbeat")
            if t - seconds(seen["ground_truth"].header.stamp) > 0.6:
                raise RuntimeError("lost physical ground truth")
            arb = signal["arbiter"]
            scoped = arb.get("goal_id") == gid and abs(t - arb["stamp"]) <= 0.25
            state = arb["state"] if scoped else None
            pos, speed = signal["xy"], signal["speed"]
            command = signal.get("cmd")
            fresh = abs(t - signal.get("cmd_stamp", -100)) <= 0.25
            moving = state == "LEARNED" and fresh and command[0] > 0.02 and speed >= 0.03
            stopped = stopped_sample(t, signal)
            stop_evidence = {"speed_mps": speed, "yaw_rate_rad_s": signal["yaw_rate"],
                             "cmd": command, "cmd_stamp": signal.get("cmd_stamp"),
                             "motion_stamp": signal["motion_stamp"]}

            if obstacle:
                actor_stamp, actor_xy = signal["actor"]
                if t - actor_stamp > 0.6:
                    raise RuntimeError("obstacle position observation went stale")
                if actor_remove_request is None:
                    if actor_xy is None:
                        raise RuntimeError("obstacle disappeared before removal")
                    if elapsed >= case["actor"]["remove_at"]:
                        obstacle.update(elapsed)
                        actor_remove_request = t
                        event("actor_remove_requested")
                    elif scoped and stopped and "moving_before_obstacle" in events:
                        if dwell is None:
                            dwell = t
                        if t-dwell >= 0.4:
                            if "stopped_before_release" not in events:
                                event("stopped_before_release", dwell_s=t-dwell, **stop_evidence)
                    else:
                        dwell = None
                        if moving and "moving_before_obstacle" not in events:
                            event("moving_before_obstacle")
                elif "actor_removed" not in events:
                    if actor_stamp >= actor_remove_request and actor_xy is None:
                        event("actor_removed", observed_stamp=actor_stamp)
                        anchor = pos
                    elif t - actor_remove_request > 1.0:
                        raise RuntimeError("Gazebo removal was not observed")
                elif moving and math.dist(anchor, pos) >= 0.25 and "advanced_after_release" not in events:
                    event("advanced_after_release", distance_m=math.dist(anchor, pos))

            if args.case == "faults" and index == 0:
                if phase == "drive" and moving and math.dist(begin_xy, pos) >= 0.15:
                    event("drop_scans")
                    set_policy_parameter(node, "diagnostic_drop_scans", True)
                    phase, phase_start = "stale", t
                elif phase == "stale" and state == "WAIT_INPUT" and stopped and faults["SCAN_STALE"] > 0:
                    dwell = t if dwell is None else dwell
                    if t - dwell >= 0.35:
                        event("stale_stop", dwell_s=t-dwell, **stop_evidence)
                        set_policy_parameter(node, "diagnostic_drop_scans", False)
                        phase, phase_start, anchor, dwell = "resume", t, pos, None
                elif phase == "stale":
                    dwell = None
                elif phase == "resume" and moving and math.dist(anchor, pos) >= 0.12:
                    event("restored_scans_motion")
                    set_policy_parameter(node, "diagnostic_inference_delay_ms", 35.0)
                    phase, phase_start = "fallback", t
                elif (phase == "fallback" and state == "FALLBACK" and faults["INFERENCE_DEADLINE"] > 0
                      and states["FALLBACK_HOLD"] > 0 and fresh and command[0] > 0.02 and speed >= 0.03):
                    event("fallback_motion")
                    set_policy_parameter(node, "diagnostic_inference_delay_ms", 0.0)
                    phase, phase_start, anchor = "recover", t, pos
                elif phase == "recover" and moving and math.dist(anchor, pos) >= 0.12:
                    event("learned_recovery")
                    req = String()
                    req.data = gid
                    cancel.publish(req)
                    event("cancel_sent")
                    phase, phase_start = "cancel", t
                elif phase == "cancel" and state == "CANCELLED" and stopped:
                    dwell = t if dwell is None else dwell
                    if t-dwell >= 0.35:
                        event("cancelled_stop", dwell_s=t-dwell, **stop_evidence)
                        phase = "done"
                elif phase == "cancel":
                    dwell = None

            if row is None:
                try:
                    with csv_path.open(newline="") as stream:
                        rows = list(csv.DictReader(stream))
                except FileNotFoundError:
                    # A Docker bind mount can briefly hide the CSV during atomic replacement.
                    rows = []
                if len(rows) > index:
                    if len(rows) != index+1 or rows[-1]["goal_id"] != gid or rows[-1]["run_id"] != run["run_id"]:
                        raise RuntimeError("recorder row belongs to a different goal or run")
                    row = rows[-1]
            if row is not None:
                waiting_cancel = phase == "cancel" and row["termination_reason"] == "CANCELLED"
                if not waiting_cancel or t - phase_start > 3:
                    return None
            if elapsed > case["timeout_sec"] + 4:
                raise RuntimeError("recorder did not terminate in bounded time")
            return f"goal {index+1}, phase {phase}, elapsed {elapsed:.2f}s"

        spin_until(node, launch, observe, 8 * case["timeout_sec"] + 10)
        run["legs"].append({"row": row, "world_xy": signal["xy"], "physical_goal_error_m":
                            math.dist(signal["xy"], world_xy(destination)), "paths": count["paths"],
                            "states": dict(states), "faults": dict(faults), "events": events})
        write_status(out, run)
        if args.case == "faults" and index == 0 and row["termination_reason"] == "CANCELLED" and phase == "done":
            continue
        if row["success"] != "1":
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--bundle", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--out", type=lambda x: Path(x).resolve(), required=True)
    parser.add_argument("--domain", type=int, default=88, choices=range(102), metavar="0..101")
    parser.add_argument("--cycles", type=int, default=2)
    args = parser.parse_args()
    if args.cycles < 1:
        parser.error("cycles must be positive")
    case = scenario(args.case, args.cycles)
    seed, hashes = candidate_identity(args.bundle)
    args.out.mkdir(parents=True, exist_ok=False)
    token = uuid.uuid4().hex
    run = {"run_id": token, "case": case, "training_seed": seed, "bundle_sha256": hashes,
           "status": "starting", "ros_domain_id": args.domain, "ign_partition": "m5_" + token,
           "autonomy_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
           "driver_sha256": digest(Path(__file__)), "wall_started_unix": time.time()}
    write_status(args.out, run)
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((Path(tempfile.gettempdir()) / f"opticore-m2-domain-{args.domain}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1", IGN_PARTITION=run["ign_partition"])
            from ament_index_python.packages import get_package_share_directory
            share = Path(get_package_share_directory("amr_bringup"))
            if case["map_sha256"] != digest(share / "maps/warehouse_map.pgm"):
                raise RuntimeError("installed warehouse map differs from reference")
            run_world(args, case, run, stack)
        import yaml
        check_evidence(args.out)
        meta = yaml.safe_load((args.out/"bag/metadata.yaml").read_text())["rosbag2_bagfile_information"]
        counts = {r["topic_metadata"]["name"]: r["message_count"] for r in meta["topics_with_message_count"]}
        if counts["/amr1/goal_pose"] != len(run["goals"]) or any(not counts.get("/amr1/"+k) for k in
                ("cmd_arbiter/state", "learned_planner/health", "global_path", "lidar", "odometry/filtered")):
            raise RuntimeError("incomplete goal or navigation signals in rosbag")
        if not (args.out/"replay.jsonl").stat().st_size:
            raise RuntimeError("missing policy trace")
        run.update(status="recorded", bag_counts=counts)
    except BaseException as exc:
        run.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        run["wall_finished_unix"] = time.time()
        write_status(args.out, run)


if __name__ == "__main__":
    main()
