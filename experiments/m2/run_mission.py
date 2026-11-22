#!/usr/bin/env python3
"""Start a fresh upstream warehouse, issue one manifest goal and preserve its evidence."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid

from mission_spec import MAP, digest, resolve
from scene import Actor, GazeboActor, world_xy


@contextmanager
def child(command, log):
    with log.open("x") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield proc
        finally:
            # ros2 launch handles SIGINT and normally stops its own children, incl Gazebo.
            for sig, grace in [(signal.SIGINT, 15), (signal.SIGTERM, 4), (signal.SIGKILL, 4)]:
                try:
                    os.killpg(proc.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    continue
                # A failed launcher can leave descendants after its own process exits.
                try:
                    os.killpg(proc.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)


def write_status(out, run):
    temp = out / "run.json.tmp"
    temp.write_text(json.dumps(run, indent=2) + "\n")
    temp.replace(out / "run.json")


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def ownership_error(node, planner, driver="/m2_mission_driver"):
    expected = {"/amr1/cmd_vel": "/amr1/" + ("dwa_planner" if planner == "b0" else "cmd_arbiter"),
                "/amr1/goal_pose": driver}
    for topic, owner in expected.items():
        pubs = node.get_publishers_info_by_topic(topic)
        names = [e.node_namespace.rstrip("/") + "/" + e.node_name for e in pubs]
        if names != [owner]:
            return f"{topic} expected exactly {owner}, found {names}"
    return None


def parameter_response(node, service, name, request):
    """Query the live node directly and bound discovery and response in wall time."""
    import rclpy

    client = node.create_client(service, name)
    try:
        if not client.wait_for_service(timeout_sec=10):
            raise RuntimeError(f"parameter service unavailable: {name}")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=10)
        if not future.done():
            raise RuntimeError(f"parameter service response timeout: {name}")
        return future.result()
    finally:
        node.destroy_client(client)


def capture_params(node, out, planner, extra_nodes=()):
    from rcl_interfaces.srv import GetParameters, ListParameters
    from ros2param.api import get_value
    import yaml

    nodes = ["astar_planner", "dwa_planner", "amcl", "nav_episode_recorder"]
    if planner == "b1":
        nodes.append("cmd_arbiter")
    nodes.extend(extra_nodes)
    target = out / "parameters"
    target.mkdir()
    hashes = {}
    for name in nodes:
        key = "/amr1/" + name
        listed = parameter_response(node, ListParameters, key + "/list_parameters", ListParameters.Request())
        names = sorted(listed.result.names)
        if not names:
            raise RuntimeError(f"no effective parameters listed for {key}")
        response = parameter_response(node, GetParameters, key + "/get_parameters", GetParameters.Request(names=names))
        if len(response.values) != len(names):
            raise RuntimeError(f"incomplete parameter response for {key}")
        params = {n: get_value(parameter_value=v) for n, v in zip(names, response.values)}
        path = target / (name + ".yaml")
        path.write_text(yaml.safe_dump({key: {"ros__parameters": params}}))
        hashes[key] = digest(path)
    return hashes


def startup(node, runner, heartbeat, scan_stamp, ground_truth, tf, case):
    """Check actual endpoints and sensors before the only /goal_pose publication."""
    from rclpy.time import Time
    from tf2_ros import TransformException
    now = node.get_clock().now().nanoseconds * 1e-9
    listeners = {e.node_name for e in node.get_subscriptions_info_by_topic("/amr1/goal_pose")
                 if e.node_namespace == "/amr1"}
    missing = {"astar_planner", "dwa_planner", "nav_episode_recorder"} - listeners
    if missing or runner.get_subscription_count() < 3:
        return f"unmatched goal subscriptions: {sorted(missing)}"
    if scan_stamp is None or not 0 <= now - scan_stamp <= 1:
        return "awaiting fresh LiDAR"
    if heartbeat is None or heartbeat[1] or not 0 <= now - heartbeat[0] <= 0.5:
        return "awaiting fresh clear collision measurement"
    if ground_truth is None or not 0 <= now - seconds(ground_truth.header.stamp) <= 1:
        return "awaiting fresh world ground truth"
    p = ground_truth.pose.pose.position
    if (ground_truth.header.frame_id.lstrip("/") != "world" or not all(map(math.isfinite, (p.x, p.y)))
            or math.dist((p.x, p.y), case["spawn_world"]) > 0.20):
        return "ground truth does not match requested world spawn"
    try:
        tr = tf.lookup_transform("map", "amr1/base_footprint", Time())
    except TransformException as exc:
        return str(exc)
    q = tr.transform.translation
    if not all(map(math.isfinite, (q.x, q.y))) or math.dist((q.x, q.y), case["start"]) > 0.5:
        return "map localization does not match requested start"
    return None


def verify_actor_position(trajectory, elapsed, observed):
    """Allow up to 0.5 s actuation lag and 0.08 m pose error against timed knots."""
    expected = trajectory.position(elapsed)
    if expected is None:
        if observed is None or elapsed - trajectory.remove_at <= 0.5:
            return
        raise RuntimeError("Gazebo still reports removed actor")
    if observed is None or len(observed) != 2 or not all(map(math.isfinite, observed)):
        raise RuntimeError("missing Gazebo actor pose")
    # Linear segment(s) travelled in the tolerated update window.
    begin = max(0., elapsed - 0.5)
    times = [begin] + [k[0] for k in trajectory.knots if begin < k[0] < elapsed] + [elapsed]
    points = [world_xy(trajectory.position(t)) for t in times]
    for (x, y), (u, v) in zip(points, points[1:]):
        dx, dy = u - x, v - y
        squared = dx * dx + dy * dy
        a = 0 if squared == 0 else max(0, min(1, ((observed[0]-x)*dx+(observed[1]-y)*dy)/squared))
        if math.dist((x + a * dx, y + a * dy), observed) <= 0.08:
            return
    raise RuntimeError(f"actor missed trajectory at t={elapsed}: observed world {observed}")


def observe(out, case, run, stack):
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from tf2_ros import Buffer, TransformListener

    rclpy.init(args=[])
    stack.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)
    node = Node("m2_mission_driver", parameter_overrides=[Parameter("use_sim_time", value=True)])
    stack.callback(node.destroy_node)
    # Refuse an already-active ROS graph: a second robot must not consume this trial's goal.
    until = time.monotonic() + 1.5
    while time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=0.1)
    if any(ns == "/amr1" for _, ns in node.get_node_names_and_namespaces()):
        raise RuntimeError("ROS domain already has robot amr1; choose another --domain")
    tf = Buffer()
    listener = TransformListener(tf, node)  # retain subscriptions for the lifetime of this run
    goal = node.create_publisher(PoseStamped, "/amr1/goal_pose", 10)
    seen = {"heartbeat": None, "scan_stamp": None, "ground_truth": None}
    trajectory = Actor.from_dict(case["actor"]) if case["actor"] else None
    actor_sample = None
    def on_contact(msg):
        nonlocal actor_sample
        v = json.loads(msg.data)
        if v.get("source") in ("gazebo_contacts", "ground_truth_geometry") and isinstance(v.get("collision"), bool):
            stamp = float(v["stamp"])
            seen["heartbeat"] = (stamp, v["collision"])
            if v.get("actor_pose_valid") is True and "actor_world_xy" in v:
                actor_sample = (stamp, v["actor_world_xy"])
    node.create_subscription(String, "/amr1/m2/contacts", on_contact, qos_profile_sensor_data)
    node.create_subscription(LaserScan, "/amr1/lidar", lambda m: seen.update(scan_stamp=seconds(m.header.stamp)),
                             qos_profile_sensor_data)
    node.create_subscription(Odometry, "/amr1/ground_truth", lambda m: seen.update(ground_truth=m),
                             qos_profile_sensor_data)
    planner = run["planner"]
    x, y = case["spawn_world"]
    bag = out / "bag"
    cmd = ["ros2", "launch", "amr_learned_planner", "m2_baseline.launch.py", f"planner:={planner}",
           "robot:=amr1", f"spawn_x:={x}", f"spawn_y:={y}", f"record_bag:={bag}"]
    csv_path = out / "episodes.csv"
    cmd.extend([f"record_csv:={csv_path}", f"scenario_id:={case['id']}", f"run_id:={run['run_id']}",
                f"seed:={run['seed']}", f"timeout_sec:={case['timeout_sec']}"])
    run["commands"] = [cmd]
    write_status(out, run)
    launch = stack.enter_context(child(cmd, out / "launch.log"))
    def spin():
        if launch.poll() is not None:
            raise RuntimeError("baseline launch exited before trial completion; inspect launch.log")
        rclpy.spin_once(node, timeout_sec=0.05)
        return node.get_clock().now().nanoseconds * 1e-9

    def await_start(verify_actor=False):
        deadline = time.monotonic() + 180
        while True:
            spin()
            why = ownership_error(node, planner) or startup(node, goal, tf=tf, case=case, **seen)
            if why is None and verify_actor:
                if (actor_sample is None or actor_sample[1] is None or
                        math.dist(world_xy(trajectory.position(0)), actor_sample[1]) > .05):
                    why = "Gazebo did not observe the challenge actor at its initial pose"
            if why is None:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"startup deadline: {why}")

    await_start()
    run["effective_parameter_sha256"] = capture_params(node, out, planner)
    obstacle = GazeboActor(trajectory) if trajectory else None
    await_start(verify_actor=bool(obstacle))
    # Stamp and dispatch only after acknowledged, measured placement in the world.
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.pose.position.x, msg.pose.position.y = map(float, case["goal"])
    msg.pose.orientation.w = 1.0
    start = seconds(msg.header.stamp)
    run["goal_stamp"] = start
    wall_start = time.monotonic()
    goal.publish(msg)
    next_update, last, next_graph_check = 0.0, 0.0, 0.0
    last_observed = actor_sample[0] if actor_sample else start
    # Bound wall time even if Gazebo stalls or its clock resets.
    deadline = time.monotonic() + 8 * case["timeout_sec"]
    with (out / "actor.jsonl").open("x", buffering=1) as actor_log:
        if obstacle:
            print(json.dumps({"elapsed":0.0, "map_xy":obstacle.last, "action":"create"}), file=actor_log)
        while time.monotonic() < deadline:
            elapsed = spin() - start
            if elapsed < last - 1e-6:
                raise RuntimeError("simulation clock moved backwards")
            if elapsed >= next_graph_check:
                error = ownership_error(node, planner)
                if error:
                    raise RuntimeError(error)
                next_graph_check = elapsed + 0.5
            if obstacle and actor_sample and actor_sample[0] > last_observed:
                observed_at, measured = actor_sample
                measured_elapsed = observed_at - start
                if measured_elapsed >= 0:
                    verify_actor_position(trajectory, measured_elapsed, measured)
                    print(json.dumps({"action": "observed", "elapsed": measured_elapsed,
                                      "world_xy": measured}), file=actor_log)
                last_observed = observed_at
            if obstacle and start + elapsed - last_observed > 0.75:
                raise RuntimeError("lost actor ground-truth observations")
            if csv_path.exists():
                with csv_path.open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                if rows:
                    row = rows[-1]
                    if len(rows) != 1 or row["scenario_id"] != case["id"] or row["run_id"] != run["run_id"]:
                        raise RuntimeError("unexpected recorder identity or additional goal")
                    wall_sec = time.monotonic() - wall_start
                    run.update(wall_seconds=wall_sec, simulation_seconds=elapsed,
                               real_time_factor=elapsed / wall_sec)
                    return
            if obstacle and elapsed >= next_update:
                if elapsed - next_update > 0.5:
                    raise RuntimeError("missed actor simulation-time deadline; discard trial")
                event = obstacle.update(elapsed)
                if event:
                    print(json.dumps(event), file=actor_log)
                next_update = elapsed + 0.10
            if elapsed > case["timeout_sec"] + 5:
                raise RuntimeError("missing terminal recorder row after simulation timeout")
            last = elapsed
    raise RuntimeError("wall deadline while waiting for Gazebo / recorder")


def check_evidence(out):
    """Perform after ros2 launch shutdown has flushed the bag and recorder."""
    import yaml

    bag = out / "bag"
    metadata = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    files = metadata["relative_file_paths"]
    if not files or any(not (bag / path).is_file() or not (bag / path).stat().st_size for path in files):
        raise RuntimeError("raw bag data missing")
    counts = {row["topic_metadata"]["name"]: row["message_count"] for row in metadata["topics_with_message_count"]}
    required = ["/clock", "/amr1/goal_pose", "/amr1/ground_truth", "/amr1/m2/contacts", "/amr1/cmd_vel"]
    missing = [name for name in required if not counts.get(name)]
    if missing:
        raise RuntimeError(f"required signals absent from bag: {missing}")
    if not (out / "episodes.events.jsonl").stat().st_size:
        raise RuntimeError("recorder event log missing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mission", help="mission ID from missions.json")
    parser.add_argument("--describe", action="store_true", help="print resolved condition without starting ROS")
    parser.add_argument("--planner", choices=("b0", "b1"), default="b0")
    parser.add_argument("--out", type=Path, help="new directory for this trial")
    parser.add_argument("--domain", type=int, default=81, choices=range(0, 102), metavar="0..101")
    parser.add_argument("--seed", type=int, default=0, help="repetition ID for pairing; Fortress RNG is uncontrolled")
    args = parser.parse_args()
    case = resolve(args.mission)
    if args.describe:
        print(json.dumps(case, indent=2))
        return
    if case["split"] == "test":
        parser.error("test missions are reserved for the frozen final evaluator")
    if args.out is None or args.seed < 0:
        parser.error("--out NEW_DIRECTORY and a nonnegative --seed are required")
    if digest(MAP) != case["map_sha256"]:
        raise RuntimeError("warehouse map digest mismatch")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    instance = uuid.uuid4().hex
    run = {**case, "status": "running", "planner": args.planner, "seed": args.seed,
           "scenario_id": case["id"], "run_id": instance, "ros_domain_id": args.domain,
           "ign_partition": "m2_" + instance}
    write_status(out, run)
    try:
        with ExitStack() as stack:
            lock = stack.enter_context((Path(tempfile.gettempdir()) / f"opticore-m2-domain-{args.domain}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1", IGN_PARTITION=run["ign_partition"])
            from ament_index_python.packages import get_package_prefix, get_package_share_directory
            share = Path(get_package_share_directory("amr_bringup"))
            live_map = digest(share / "maps/warehouse_map.pgm")
            if live_map != case["map_sha256"]:
                raise RuntimeError("installed amr_bringup map differs from the mission map")
            run["installed_hashes"] = {str(p.relative_to(share)):digest(p) for p in
                                       (share/"maps/warehouse_map.pgm", share/"maps/warehouse_map.yaml",
                                        share/"urdf/amr_robot.urdf.xacro", share/"worlds/warehouse.world")}
            run["installed_hashes"]["dwa_params.yaml"] = digest(
                Path(get_package_share_directory("amr_navigation"))/"config/dwa_params.yaml")
            for package, files in {
                "amr_navigation": ["lib/amr_navigation/dwa_node.py", "lib/amr_navigation/astar_planner"],
                "amr_m2_contact_monitor": ["lib/libm2_contact_monitor.so"],
                "amr_bringup": ["share/amr_bringup/launch/multi_robot.launch.py"],
                "amr_learned_planner": ["share/amr_learned_planner/launch/m2_baseline.launch.py"],
            }.items():
                prefix = Path(get_package_prefix(package))
                for path in files + [f"share/{package}/package.xml"]:
                    run["installed_hashes"][path] = digest(prefix / path)
            run["autonomy_commit"] = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True).stdout.strip()
            observe(out, case, run, stack)
        check_evidence(out)
        run["status"] = "recorded"
    except BaseException as exc:
        run.update(status="invalid", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_status(out, run)


if __name__ == "__main__":
    main()
