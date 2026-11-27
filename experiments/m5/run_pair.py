#!/usr/bin/env python3
"""Exercise two operational controllers and the fleet goal-replacement interface in Fortress."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
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
sys.path.insert(0, str(ROOT / "experiments/m2"))
from mission_spec import digest, load_cases
from run_mission import child, seconds, write_status
from scene import world_xy

START = {"amr1": (2.0, 22.0), "amr2": (2.0, 16.0)}
GOALS = {r: (x + 8.0, y) for r, (x, y) in START.items()}


class ProtocolFailure(RuntimeError):
    """Measured behavior contradicted the fleet protocol after both robots became ready."""


def authority_graph(node, goal_owners=("/m5_fleet_probe",)):
    def owners(topic):
        return sorted(e.node_namespace.rstrip("/") + "/" + e.node_name
                      for e in node.get_publishers_info_by_topic(topic))

    result = {"/clock": owners("/clock")}
    inputs = {}
    if result["/clock"] != ["/amr1/ros_gz_bridge"]:
        return None
    for robot in START:
        prefix = f"/{robot}/"
        topics = {
            "cmd_vel": prefix + "cmd_arbiter", "dwa/status": prefix + "cmd_arbiter",
            "cmd_arbiter/state": prefix + "cmd_arbiter", "goal_pose": goal_owners,
            "global_path": prefix + "astar_planner", "dwa_fallback/cmd_vel": prefix + "dwa_planner",
            "dwa_fallback/status": prefix + "dwa_planner", "learned_planner/health": prefix + "m3_policy",
            "learned_planner/cmd_vel": prefix + "m3_policy", "m2/contacts": prefix + "m2_contact_bridge",
        }
        for channel, owner in topics.items():
            topic = prefix + channel
            result[topic] = owners(topic)
            if result[topic] != (sorted(owner) if channel == "goal_pose" else [owner]):
                return None
        for channel, required in (("goal_pose", {"cmd_arbiter", "m3_policy", "dwa_planner", "astar_planner"}),
                                  ("cmd_vel", {"ros_gz_bridge"}), ("dwa/status", {"astar_planner"})):
            topic = prefix + channel
            listeners = node.get_subscriptions_info_by_topic(topic)
            if any(e.node_namespace in {"/" + r for r in START if r != robot} for e in listeners):
                return None
            names = {e.node_name for e in listeners if e.node_namespace == "/" + robot}
            if not required <= names:
                return None
            inputs[topic] = sorted(e.node_namespace.rstrip("/") + "/" + e.node_name for e in listeners)
    return {"publishers": result, "subscribers": inputs}


def near_zero(command):
    return all(abs(component) < 1e-6 for component in command)


def hold_acknowledged(row, goal_id, issued):
    # Each source must have produced a sample since the replacement.
    return (row["state"]["goal_id"] == goal_id and row["state"]["active"] == "stop"
            and min(row["state"]["stamp"], row["motion"][4], row["cmd_received"]) >= issued
            and near_zero(row["cmd"]) and row["motion"][2] < .02 and abs(row["motion"][3]) < .05)


def check_telemetry(robot, row, t, epoch, phase):
    c = row["contact"]
    if c["collision"] or c["collision_events"] != 0:
        raise ProtocolFailure(f"{robot}: physical contact during {phase}")
    stamps = (c["stamp"], row["motion"][4], row["scan"][0], row["health"]["stamp"],
              row["state"]["stamp"], row["cmd_received"], row["status_received"])
    if c["coverage_epoch"] != epoch or any(not -.10 <= t - stamp <= .6 for stamp in stamps):
        raise RuntimeError(f"{robot}: telemetry gap during {phase}")


def run(args, report, stack):
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformException, TransformListener

    rclpy.init(args=[])
    stack.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)
    node = Node("m5_fleet_probe", parameter_overrides=[Parameter("use_sim_time", value=True)])
    stack.callback(node.destroy_node)
    tf = Buffer()
    listener = TransformListener(tf, node)
    log = stack.enter_context((args.out / "witness.jsonl").open("x", buffering=1))
    seen = {r: {} for r in START}
    pubs = {}
    expected = {}
    allowed_ids = {r: set() for r in START}
    monitoring = False
    safety_snapshot = {}
    held_since = None

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def record(robot, key, value):
        print(json.dumps({"time": now(), "robot": robot, "kind": key, "value": value},
                         allow_nan=False), file=log)

    def receive(r, key, value):
        record(r, key, value)
        if key in {"health", "state"} and monitoring:
            gid = value.get("goal_id")
            if gid is not None and gid not in allowed_ids[r]:
                raise ProtocolFailure(f"{r} accepted foreign goal {gid}")
        if key == "state" and held_since is not None and value["goal_id"] != expected[r]:
            raise ProtocolFailure(f"{r} lost its assigned goal during fleet hold")
        if key == "cmd" and monitoring and r not in expected and not near_zero(value):
            raise ProtocolFailure(f"unassigned {r} emitted a motion command")
        if key == "cmd" and held_since is not None and r == "amr1" and not near_zero(value):
            raise ProtocolFailure("amr1 emitted a motion command during fleet hold")
        seen[r][key] = value
        seen[r][f"{key}_received"] = now()

    for robot in START:
        path = "/" + robot + "/"
        pubs[robot] = node.create_publisher(PoseStamped, path + "goal_pose", 10)
        node.create_subscription(String, path + "cmd_arbiter/state",
                                 lambda m, r=robot: receive(r, "state", json.loads(m.data)), 10)
        node.create_subscription(String, path + "dwa/status", lambda m, r=robot: receive(r, "status", m.data), 10)
        node.create_subscription(String, path + "learned_planner/health",
                                 lambda m, r=robot: receive(r, "health", json.loads(m.data)), 10)
        node.create_subscription(Twist, path + "cmd_vel",
                                 lambda m, r=robot: receive(r, "cmd", (m.linear.x, m.angular.z)), 10)
        node.create_subscription(Odometry, path + "ground_truth",
                                 lambda m, r=robot: receive(r, "motion", (m.pose.pose.position.x,
                                     m.pose.pose.position.y, math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y),
                                     m.twist.twist.angular.z,
                                     seconds(m.header.stamp), m.header.frame_id)), qos_profile_sensor_data)
        node.create_subscription(LaserScan, path + "lidar",
                                 lambda m, r=robot: receive(r, "scan", [seconds(m.header.stamp), m.header.frame_id]),
                                 qos_profile_sensor_data)
        node.create_subscription(String, path + "m2/contacts",
                                 lambda m, r=robot: receive(r, "contact", json.loads(m.data)), qos_profile_sensor_data)

    def mapped(r):
        pos = tf.lookup_transform("map", f"{r}/base_footprint", Time()).transform.translation
        return pos.x, pos.y

    next_ready_check = 0.0

    def ready():
        nonlocal next_ready_check
        if time.monotonic() < next_ready_check:
            return False
        next_ready_check = time.monotonic() + .3
        if not authority_graph(node):
            return False
        # The upstream A* silently discards goals until its first map callback.
        launch_log = (args.out / "launch.log").read_text(errors="replace")
        for r, row in seen.items():
            if (f"[{r}.astar_planner]: 맵 수신:" not in launch_log or
                    not {"motion", "scan", "health", "state", "status", "cmd", "contact"} <= row.keys()):
                return False
            m, scan, c = row["motion"], row["scan"], row["contact"]
            if (m[5].lstrip("/") != "world" or math.dist(m[:2], world_xy(START[r])) > .2
                    or scan[1].lstrip("/") not in {f"{r}/lidar_link", f"{r}/base_footprint/lidar_sensor"}
                    or c.get("source") != "gazebo_contacts" or c["collision"] or c["collision_events"] != 0
                    or any(not -.05 <= now() - t <= .5 for t in (m[4], scan[0], c["stamp"],
                           row["health"]["stamp"], row["state"]["stamp"], row["cmd_received"], row["status_received"]))):
                return False
            try:
                if math.dist(mapped(r), START[r]) > .5:
                    return False
            except TransformException:
                return False
        return True

    until = time.monotonic() + 1.5
    while time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=.05)
    if any(ns in {"/amr1", "/amr2"} for _, ns in node.get_node_names_and_namespaces()):
        raise RuntimeError("ROS domain already contains a robot")
    launch = stack.enter_context(child([
        "ros2", "launch", "amr_learned_planner", "m5_pair.launch.py", f"bundle:={args.bundle}",
        f"output:={args.out / 'policies'}"], args.out / "launch.log"))

    def wait(phase, predicate, wall_budget, sim_budget=None):
        report["phase"] = phase
        write_status(args.out, report)
        t0, last, wall_end = now(), now(), time.monotonic() + wall_budget
        next_graph = last + 1
        while time.monotonic() < wall_end:
            if launch.poll() is not None:
                raise RuntimeError(f"warehouse exited with code {launch.returncode}")
            rclpy.spin_once(node, timeout_sec=.05)
            t = now()
            if monitoring:
                if t < last - 1e-7:
                    raise RuntimeError("simulation clock moved backwards")
                for r in START:
                    check_telemetry(r, seen[r], t, safety_snapshot[r], phase)
                if t >= next_graph:
                    if not authority_graph(node):
                        raise RuntimeError("command, goal or status endpoint changed ownership")
                    next_graph = t + 1
            last = t
            if predicate():
                report["phases_passed"].append(phase)
                record("pair", "phase_passed", phase)
                return
            if sim_budget and t - t0 > sim_budget:
                raise ProtocolFailure(f"{phase} missed its {sim_budget}s simulation deadline")
        raise RuntimeError(f"{phase} missed its {wall_budget}s wall-clock acquisition deadline")

    def send(r, xy):
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = node.get_clock().now().to_msg()
        goal.pose.position.x, goal.pose.position.y = xy
        goal.pose.orientation.w = 1.
        stamp = goal.header.stamp
        gid = f"map:{stamp.sec}.{stamp.nanosec:09d}"
        if any(gid in ids for ids in allowed_ids.values()):
            raise RuntimeError("goal stamps must distinguish both the robots and phases")
        allowed_ids[r].add(gid)
        expected[r] = gid
        record(r, "published_goal", {"goal_id": gid, "map_xy": xy})
        report["goals"].append({"robot": r, "goal_id": gid, "map_xy": xy})
        pubs[r].publish(goal)
        return gid, seconds(stamp)

    def active(r):
        return (seen[r]["state"]["goal_id"] == expected[r] and seen[r]["state"]["active"] == "learned"
                and seen[r]["cmd"][0] > .02)

    wait("both_ready", ready, 200)
    report["initial_graph"] = authority_graph(node)
    if not report["initial_graph"]:
        raise RuntimeError("authority graph changed at startup")
    safety_snapshot = {r: seen[r]["contact"]["coverage_epoch"] for r in START}
    monitoring = True
    origins = {r: seen[r]["motion"][:2] for r in START}
    send("amr2", GOALS["amr2"])
    def one_assigned():
        if math.dist(origins["amr1"], seen["amr1"]["motion"][:2]) > .05:
            raise ProtocolFailure("unassigned amr1 changed position")
        return (active("amr2") and seen["amr1"]["state"]["goal_id"] is None
                and near_zero(seen["amr1"]["cmd"]))

    wait("only_amr2_assigned", one_assigned, 100, 25)
    send("amr1", GOALS["amr1"])
    wait("both_moving", lambda: all(active(r) and math.dist(origins[r], seen[r]["motion"][:2]) >= .12
                                    for r in START), 100, 25)
    held_id, issued = send("amr1", mapped("amr1"))
    wait("amr1_hold_acknowledged", lambda: hold_acknowledged(seen["amr1"], held_id, issued), 60, 8)
    held_since = now()
    held_at = seen["amr1"]["motion"][:2]
    other_at = seen["amr2"]["motion"][:2]

    def isolated_hold():
        if seen["amr1"]["motion"][2] >= .02 or abs(seen["amr1"]["motion"][3]) >= .05:
            raise ProtocolFailure("amr1 regained speed during hold")
        if math.dist(held_at, seen["amr1"]["motion"][:2]) > .05:
            raise ProtocolFailure("amr1 changed position during hold")
        return (now() - held_since >= 1. and active("amr2")
                and math.dist(other_at, seen["amr2"]["motion"][:2]) >= .12
                and seen["amr1"]["state"]["goal_id"] == held_id)

    wait("amr1_stopped_while_amr2_continued", isolated_hold, 80, 12)
    report["hold"] = {"duration_s": now()-held_since, "amr1_displacement_m": math.dist(held_at, seen["amr1"]["motion"][:2]),
                      "amr2_displacement_m": math.dist(other_at, seen["amr2"]["motion"][:2])}
    held_since = None
    send("amr1", GOALS["amr1"])
    wait("amr1_resumed_original_goal", lambda: active("amr1") and
         math.dist(held_at, seen["amr1"]["motion"][:2]) >= .12 and
         seen["amr2"]["state"]["goal_id"] == expected["amr2"], 100, 25)
    report["final_graph"] = authority_graph(node)
    if not report["final_graph"]:
        raise RuntimeError("authority graph changed at completion")
    report["final_contact"] = {r: seen[r]["contact"] for r in START}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", type=int, required=True, choices=range(102), metavar="0..101")
    args = parser.parse_args()
    args.bundle, args.out = args.bundle.resolve(), args.out.resolve()
    actors = json.loads((ROOT / "experiments/m4/followup/mixed.json").read_text())["training"]
    bundle_hash = digest(args.bundle / "bundle.json")
    candidate = next((x for x in actors if x["bundle_sha256"] == bundle_hash), None)
    if candidate is None:
        parser.error("expected one of the three selected M4 bundles")
    args.out.mkdir(parents=True, exist_ok=False)
    partition = "m5_pair_" + uuid.uuid4().hex
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1", IGN_PARTITION=partition)
    report = {"status": "starting", "phase": "launch", "goals": [], "phases_passed": [],
              "ros_domain": args.domain, "ign_partition": partition,
              "fleet_input": "stamped goal replacement issued by probe", "seed": candidate["seed"],
              "bundle_sha256": bundle_hash, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "autonomy_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                                        text=True).strip()}
    write_status(args.out, report)
    try:
        with ExitStack() as stack:
            from ament_index_python.packages import get_package_share_directory

            lock_path = Path(tempfile.gettempdir()) / f"opticore-m2-domain-{args.domain}.lock"
            lock = stack.enter_context(lock_path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            share = Path(get_package_share_directory("amr_bringup"))
            report["map_sha256"] = digest(share / "maps/warehouse_map.pgm")
            if load_cases()["map_sha256"] != report["map_sha256"]:
                raise RuntimeError("installed warehouse map differs from the pinned development map")
            run(args, report, stack)
        report["status"] = "verified"
    except ProtocolFailure as error:
        report.update(status="failed", error=str(error))
        raise
    except BaseException as error:
        report.update(status="unverified", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_status(args.out, report)


if __name__ == "__main__":
    main()
