#!/usr/bin/env python3
"""Observe HOLD and mission restoration by the pinned upstream deadlock manager in Fortress."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from run_pair import ProtocolFailure, authority_graph, child, digest, load_cases, near_zero, seconds, world_xy, write_status

HERE = Path(__file__).resolve().parent
DRIVER = "/m5_fleet_driver"
MANAGER = "/deadlock_manager"
PINNED_MANAGER = "62ad7e3142c8469a806e6ef9d17d6c855c5c9390e4cd600ede5f41a085893692"
START = {"amr1": (3.6, 22.), "amr2": (2., 22.)}
MISSIONS = {"amr1": (10., 22.), "amr2": (8., 22.)}
# A 0.40 m loss of the initial 1.60 m longitudinal spacing ends the trial early.
MIN_SEPARATION = 1.20


def endpoint_gids(node, topic):
    writers = {}
    for ep in node.get_publishers_info_by_topic(topic):
        owner = ep.node_namespace.rstrip("/") + "/" + ep.node_name
        gid = bytes(ep.endpoint_gid)
        if owner in writers or not any(gid) or gid.hex() in writers.values():
            raise RuntimeError(f"missing or ambiguous DDS writer on {topic}")
        writers[owner] = gid.hex()
    return writers


def writer_snapshot(node):
    graph = authority_graph(node, (DRIVER, MANAGER))
    if not graph:
        return None
    if set(endpoint_gids(node, "/fleet/deadlock_status")) != {MANAGER}:
        return None
    if set(endpoint_gids(node, "/m5/observed_goals")) != {"/m5_goal_witness"}:
        return None
    writers = {}
    for r in START:
        topic = f"/{r}/goal_pose"
        writers[r] = endpoint_gids(node, topic)
        if set(writers[r]) != {DRIVER, MANAGER}:
            return None
        if not {"/m5_goal_witness", MANAGER} <= set(graph["subscribers"][topic]):
            return None
    if len({gid for endpoints in writers.values() for gid in endpoints.values()}) != 4:
        raise RuntimeError("conflicting writer identities across robot namespaces")
    return {"graph": graph, "goal_writers": writers}


class GoalLedger:
    def __init__(self, writers, hold_xy):
        self.writers = writers
        self.hold_xy = hold_xy
        self.issued = {}
        self.observed = {r: {} for r in START}
        self.kinds = {"hold": set(), "resume": set()}

    def observe(self, sample):
        r = sample["robot"]
        gid = sample["writer"]
        goal = f"map:{sample['sec']}.{sample['nsec']:09d}"
        xy = sample["xy"]
        q = sample["orientation"]
        if (not sample["is_map"] or len(xy) != 2 or len(q) != 4
                or not all(math.isfinite(v) for v in [*xy, *q]) or abs(sum(v*v for v in q) - 1) > .01):
            raise ProtocolFailure(f"invalid {r} goal frame or pose")
        if goal in self.observed[r]:
            raise ProtocolFailure(f"{r} reused a goal timestamp")
        if gid == self.writers[r][DRIVER]:
            if goal != self.issued.get(r) or math.dist(xy, MISSIONS[r]) > 1e-5 or abs(q[3]) < .99999:
                raise ProtocolFailure(f"unexpected driver goal for {r}")
            kind = "mission"
        elif gid == self.writers[r][MANAGER]:
            if r != "amr2" or r not in self.issued:
                raise ProtocolFailure(f"manager overrode the wrong robot or an unassigned robot: {r}")
            if math.dist(xy, self.hold_xy) <= .30 and not self.kinds["resume"]:
                kind = "hold"
            elif self.kinds["hold"] and math.dist(xy, MISSIONS[r]) < 1e-5 and abs(q[3]) > .99999:
                kind = "resume"
            else:
                raise ProtocolFailure(f"unexpected manager goal on {r}: {xy}")
            self.kinds[kind].add(goal)
        else:
            raise RuntimeError(f"unregistered DDS goal writer on {r}: {gid}")
        self.observed[r][goal] = kind
        return kind, goal


def check_distance(positions):
    distance = math.dist(positions["amr1"], positions["amr2"])
    if not math.isfinite(distance):
        raise RuntimeError("nonfinite Gazebo position")
    if distance <= MIN_SEPARATION:
        raise ProtocolFailure(f"ground-truth separation {distance:.3f} m reached the {MIN_SEPARATION} m stop boundary")
    return distance


def physical_stop(row):
    return near_zero(row["cmd"]) and row["motion"][2] < .02 and abs(row["motion"][3]) < .05


def run(args, result, stack):
    import rclpy
    from ament_index_python.packages import get_package_prefix, get_package_share_directory
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry
    from rcl_interfaces.srv import SetParameters
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    from run_mission import parameter_response

    manager = Path(get_package_prefix("amr_fleet")) / "lib/amr_fleet/deadlock_manager.py"
    result["manager_sha256"] = digest(manager)
    if result["manager_sha256"] != PINNED_MANAGER:
        raise RuntimeError(f"installed fleet manager differs from pinned upstream: {manager}")
    warehouse = Path(get_package_share_directory("amr_bringup")) / "maps/warehouse_map.pgm"
    result["map_sha256"] = digest(warehouse)
    if result["map_sha256"] != load_cases()["map_sha256"]:
        raise RuntimeError("unexpected development map")
    scratch = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="m5-goal-witness-")))
    with (args.out / "witness-build.log").open("x") as log:
        subprocess.run(["cmake", "-S", str(HERE / "fleet_witness"), "-B", str(scratch)], check=True,
                       stdout=log, stderr=subprocess.STDOUT, timeout=120)
        subprocess.run(["cmake", "--build", str(scratch)], check=True, stdout=log, stderr=subprocess.STDOUT,
                       timeout=240)
    rclpy.init(args=[])
    stack.callback(lambda: rclpy.shutdown() if rclpy.ok() else None)
    node = Node(DRIVER[1:], parameter_overrides=[Parameter("use_sim_time", value=True)])
    stack.callback(node.destroy_node)
    until = time.monotonic() + 1.5
    while time.monotonic() < until:
        rclpy.spin_once(node, timeout_sec=.05)
    if any(ns in {"/amr1", "/amr2"} or name in {"deadlock_manager", "m5_goal_witness"}
           for name, ns in node.get_node_names_and_namespaces()):
        raise RuntimeError("ROS domain already contains fleet nodes")
    witness = stack.enter_context(child([str(scratch / "m5_goal_witness")], args.out / "witness-node.log"))
    launch = stack.enter_context(child(["ros2", "launch", "amr_learned_planner", "m5_pair.launch.py",
                                        "fleet_manager:=true", f"bundle:={args.bundle}",
                                        f"output:={args.out / 'policies'}"], args.out / "launch.log"))
    log = stack.enter_context((args.out / "witness.jsonl").open("x", buffering=1))
    rows = {r: {} for r in START}
    accepted = {r: set() for r in START}
    pubs = {}
    fleet = {}
    ledger = None
    snapshot = None
    epochs = {}
    holding = False
    origins = {}

    def now():
        return node.get_clock().now().nanoseconds * 1e-9

    def record(r, kind, value):
        print(json.dumps({"time": now(), "robot": r, "kind": kind, "value": value}, allow_nan=False), file=log)

    def receive(r, key, value):
        rows[r][key] = value
        rows[r][key + "_at"] = now()
        record(r, key, value)
        if key == "state" and value.get("goal_id") is not None:
            accepted[r].add(value["goal_id"])
        if ledger is not None and r not in ledger.issued and key == "cmd" and not near_zero(value):
            raise ProtocolFailure(f"unassigned {r} emitted a nonzero command")
        if holding and r == "amr2" and key in {"motion", "cmd"}:
            if not physical_stop(rows[r]) or math.dist(origins[r], rows[r]["motion"][:2]) > .05:
                raise ProtocolFailure("AMR2 moved during the manager's HOLD")

    def goal_observed(msg):
        sample = json.loads(msg.data)
        if ledger is None:
            raise RuntimeError("unsolicited goal before capturing endpoint identities")
        kind, goal_id = ledger.observe(sample)
        result["goals"].append({**sample, "kind": kind, "goal_id": goal_id})
        record(sample["robot"], kind, result["goals"][-1])

    def fleet_status(msg):
        nonlocal fleet
        fleet = json.loads(msg.data)
        record("fleet", "status", fleet)
        if ledger is not None and (fleet["robots"]["amr1"]["fsm"] != "NORMAL" or
                                   any(v["livelock"] for v in fleet["robots"].values())):
            raise ProtocolFailure("manager yielded the winner or declared a livelock")

    node.create_subscription(String, "/m5/observed_goals", goal_observed, 32)
    node.create_subscription(String, "/fleet/deadlock_status", fleet_status, 10)

    def subscribe_robot(r):
        p = f"/{r}/"
        pubs[r] = node.create_publisher(PoseStamped, p + "goal_pose", 10)
        node.create_subscription(String, p + "cmd_arbiter/state", lambda m: receive(r, "state", json.loads(m.data)), 10)
        node.create_subscription(String, p + "learned_planner/health", lambda m: receive(r, "health", json.loads(m.data)), 10)
        node.create_subscription(Twist, p + "cmd_vel", lambda m: receive(r, "cmd", (m.linear.x, m.angular.z)), 10)
        node.create_subscription(Odometry, p + "ground_truth", lambda m: receive(r, "motion", (
            m.pose.pose.position.x, m.pose.pose.position.y, math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y),
            m.twist.twist.angular.z, seconds(m.header.stamp), m.header.frame_id)), qos_profile_sensor_data)
        node.create_subscription(LaserScan, p + "lidar", lambda m: receive(r, "scan", (
            seconds(m.header.stamp), m.header.frame_id)), qos_profile_sensor_data)
        node.create_subscription(String, p + "m2/contacts", lambda m: receive(r, "contact", json.loads(m.data)),
                                 qos_profile_sensor_data)

    for r in START:
        subscribe_robot(r)

    def ready():
        if set(fleet.get("robots", {})) != set(START):
            return False
        launch_log = (args.out / "launch.log").read_text(errors="replace")
        for r, row in rows.items():
            if f"[{r}.astar_planner]: 맵 수신:" not in launch_log or not {"cmd", "motion", "scan", "state", "health", "contact"} <= row.keys():
                return False
            m, c, scan = row["motion"], row["contact"], row["scan"]
            robot = fleet["robots"][r]
            if (robot["pose"] is None or robot["fsm"] != "NORMAL" or robot["has_goal"]
                    or math.dist(robot["pose"][:2], START[r]) > .45 or abs(robot["pose"][2]) > .28
                    or m[5].lstrip("/") != "world" or math.dist(m[:2], world_xy(START[r])) > .15
                    or scan[1].lstrip("/") not in {f"{r}/lidar_link", f"{r}/base_footprint/lidar_sensor"}
                    or c["source"] != "gazebo_contacts" or c["collision"] or c["collision_events"]
                    or any(not -.1 <= now()-v <= .6 for v in (c["stamp"], m[4], scan[0]))):
                return False
        return (fleet["robots"]["amr1"]["prio"] > fleet["robots"]["amr2"]["prio"]
                and writer_snapshot(node) is not None)

    def poll_safety(t):
        for r, row in rows.items():
            c, m = row["contact"], row["motion"]
            if c["collision"] or c["collision_events"]:
                raise ProtocolFailure(f"{r} made physical contact")
            if c["coverage_epoch"] != epochs[r] or c["source"] != "gazebo_contacts" or not all(
                -.1 <= t-stamp <= .7 for stamp in (c["stamp"], m[4], row["scan"][0], row["cmd_at"],
                                                     row["state"]["stamp"], row["health"]["stamp"])):
                raise RuntimeError(f"{r} telemetry coverage interrupted")
        if not -.1 <= t - fleet["t"] <= 1.25:
            raise RuntimeError("fleet status telemetry interrupted")
        distance = check_distance({r: row["motion"][:2] for r, row in rows.items()})
        result["min_separation_m"] = min(distance, result.get("min_separation_m", distance))

    def wait(phase, predicate, sim_budget=None):
        result["phase"] = phase
        write_status(args.out, result)
        started, last, wall_end = now(), now(), time.monotonic() + (210 if sim_budget is None else 100)
        next_graph, next_ready = started + 1, 0.
        while time.monotonic() < wall_end:
            if launch.poll() is not None or witness.poll() is not None:
                raise RuntimeError("Fortress launch or DDS witness exited")
            rclpy.spin_once(node, timeout_sec=.05)
            t = now()
            if ledger is not None:
                if t < last:
                    raise RuntimeError("simulation clock moved backwards")
                poll_safety(t)
                if t >= next_graph:
                    current = writer_snapshot(node)
                    if not current or current["goal_writers"] != snapshot["goal_writers"]:
                        raise RuntimeError("fleet endpoints or pinned DDS writers changed")
                    next_graph = t + 1
            elif time.monotonic() < next_ready:
                continue
            else:
                next_ready = time.monotonic() + .25
            last = t
            if predicate():
                result["phases_passed"].append(phase)
                record("fleet", "phase_passed", phase)
                return
            if sim_budget is not None and t-started > sim_budget:
                raise ProtocolFailure(f"{phase} missed its {sim_budget:g}s simulation deadline")
        raise RuntimeError(f"{phase} exhausted its wall-clock acquisition budget")

    def scan_fault(enable):
        p = Parameter("diagnostic_drop_scans", value=enable).to_parameter_msg()
        answer = parameter_response(node, SetParameters, "/amr2/m3_policy/set_parameters", SetParameters.Request(parameters=[p]))
        if len(answer.results) != 1 or not answer.results[0].successful:
            raise RuntimeError("AMR2 scan fault parameter was rejected")
        record("amr2", "scan_drop_enabled", enable)

    def issue(r):
        m = PoseStamped()
        m.header.frame_id, m.header.stamp = "map", node.get_clock().now().to_msg()
        m.pose.position.x, m.pose.position.y = MISSIONS[r]
        m.pose.orientation.w = 1.
        ledger.issued[r] = f"map:{m.header.stamp.sec}.{m.header.stamp.nanosec:09d}"
        pubs[r].publish(m)

    wait("both_ready", ready)
    snapshot = writer_snapshot(node)
    if snapshot is None:
        raise RuntimeError("fleet endpoints disappeared during startup")
    result["initial_graph"] = snapshot
    epochs = {r: row["contact"]["coverage_epoch"] for r, row in rows.items()}
    origins = {r: row["motion"][:2] for r, row in rows.items()}
    ledger = GoalLedger(snapshot["goal_writers"], fleet["robots"]["amr2"]["pose"][:2])
    scan_fault(True)
    armed = now()
    wait("stationary_encounter_armed", lambda: now()-armed >= .6 and all(physical_stop(row) for row in rows.values()), 3)
    issue("amr2")
    wait("follower_mission_observed", lambda: ledger.observed["amr2"].get(ledger.issued["amr2"]) == "mission", 3)

    def assigned_hold():
        r = fleet["robots"]["amr2"]
        return (r["fsm"] == "HOLD" and r["stuck"] and r["blocked_by"] == "amr1"
                and {"stuck": "amr2", "by": "amr1"} in fleet["pairs"]
                and ledger.observed["amr2"].get(ledger.issued["amr2"]) == "mission"
                and rows["amr2"]["state"]["goal_id"] in ledger.kinds["hold"]
                and rows["amr2"]["state"]["active"] == "stop" and physical_stop(rows["amr2"]))

    wait("manager_hold_acknowledged", assigned_hold, 10)
    result["hold_status"] = fleet
    holding = True
    scan_fault(False)
    restored = now()
    wait("hold_with_live_scans", lambda: now()-restored >= 1. and assigned_hold()
         and (rows["amr2"]["health"].get("shield_scan_stamp") or 0) > restored, 5)
    result["hold_duration_live_scans_s"] = now()-restored
    result["hold_displacement_m"] = math.dist(rows["amr2"]["motion"][:2], origins["amr2"])
    holding = False
    issue("amr1")

    def winner_cleared():
        winner = rows["amr1"]
        return (winner["state"]["goal_id"] == ledger.issued["amr1"] and winner["state"]["active"] == "learned"
                and ledger.observed["amr1"].get(ledger.issued["amr1"]) == "mission"
                and winner["cmd"][0] > .02 and winner["motion"][0] - origins["amr1"][0] > .45)

    wait("winner_cleared", winner_cleared, 15)
    wait("manager_restored_original_goal", lambda: bool(ledger.kinds["resume"] & accepted["amr2"]), 10)
    wait("loser_resumed_motion", lambda: rows["amr2"]["state"]["goal_id"] in ledger.kinds["resume"]
         and rows["amr2"]["state"]["active"] == "learned" and rows["amr2"]["cmd"][0] > .02
         and rows["amr2"]["motion"][0] - origins["amr2"][0] > .18
         and fleet["robots"]["amr2"]["fsm"] == "NORMAL" and fleet["robots"]["amr2"]["blocked_by"] is None, 20)
    # Arbiter and writer witness use different DDS topics. Let late observations
    # catch up after resumption before comparing all goal IDs seen on each robot.
    completed_at = now()
    wait("all_accepted_goals_authenticated", lambda: now()-completed_at >= .35 and all(
        ids <= ledger.observed[r].keys() for r, ids in accepted.items()), 2)
    result["final_graph"] = writer_snapshot(node)
    if result["final_graph"] is None or result["final_graph"]["goal_writers"] != snapshot["goal_writers"]:
        raise RuntimeError("endpoint identity changed before completion")
    result["final_contact"] = {r: row["contact"] for r, row in rows.items()}
    result["final_status"] = fleet
    result["forward_progress_m"] = {r: row["motion"][0]-origins[r][0] for r, row in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", type=int, choices=range(102), metavar="0..101", required=True)
    args = parser.parse_args()
    args.out, args.bundle = args.out.resolve(), args.bundle.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    partition = "m5_fleet_" + uuid.uuid4().hex
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), ROS_LOCALHOST_ONLY="1", IGN_PARTITION=partition)
    result = {"status": "starting", "phase": "launch", "phases_passed": [], "goals": [],
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "ros_domain": args.domain,
              "ign_partition": partition, "encounter": "stationary follower; induced scan loss; moving leader",
              "fleet_input": "pinned upstream deadlock manager", "separation_stop_m": MIN_SEPARATION}
    write_status(args.out, result)
    try:
        from run_single import candidate_identity
        seed, hashes = candidate_identity(args.bundle)
        result.update(seed=seed, bundle_sha256=hashes["bundle.json"], candidate_hashes=hashes,
                      autonomy_commit=subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip())
        with ExitStack() as stack:
            lock = stack.enter_context((Path(tempfile.gettempdir()) / f"opticore-m2-domain-{args.domain}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(args, result, stack)
        result["status"] = "verified"
    except ProtocolFailure as error:
        result.update(status="failed", error=str(error))
        raise
    except BaseException as error:
        result.update(status="unverified", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_status(args.out, result)


if __name__ == "__main__":
    main()
