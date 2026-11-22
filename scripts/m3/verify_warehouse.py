#!/usr/bin/env python3
"""Verify the M3 warehouse trial against its raw rosbag and stamped policy replay."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import closing
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.learning.stamped_replay import replay


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        fingerprint = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            fingerprint.update(block)
    return fingerprint.hexdigest()


def read_rows(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def stamp(value):
    return value.sec + value.nanosec * 1e-9


def bag_signals(out):
    """Read the subset used for independent validation via ROS message types."""
    from geometry_msgs.msg import PoseStamped, Twist
    from nav_msgs.msg import Odometry, Path as NavPath
    from rclpy.serialization import deserialize_message
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import String
    import yaml

    topics = {
        "/clock": ("clock", Clock),
        "/amr1/goal_pose": ("goal", PoseStamped),
        "/amr1/global_path": ("path", NavPath),
        "/amr1/lidar": ("scan", LaserScan),
        "/amr1/ground_truth": ("motion", Odometry),
        "/amr1/m2/contacts": ("contact", String),
        "/amr1/cmd_vel": ("final", Twist),
        "/amr1/learned_planner/cmd_vel": ("learned", Twist),
        "/amr1/dwa_fallback/cmd_vel": ("dwa", Twist),
        "/amr1/learned_planner/health": ("health", String),
        "/amr1/cmd_arbiter/state": ("arbiter", String),
    }
    bag = out / "bag"
    meta = yaml.safe_load((bag / "metadata.yaml").read_text())["rosbag2_bagfile_information"]
    require(meta["storage_identifier"] == "sqlite3", "M3 verifier expects a SQLite rosbag")
    clock = None
    signals = defaultdict(list)
    variables = ", ".join("?" for _ in topics)
    query = f"""SELECT topics.name, messages.data FROM messages JOIN topics
                ON topics.id=messages.topic_id WHERE topics.name IN ({variables})
                ORDER BY messages.timestamp, messages.id"""
    for filename in meta["relative_file_paths"]:
        with closing(sqlite3.connect(f"{(bag / filename).resolve().as_uri()}?mode=ro", uri=True)) as db:
            for topic, raw in db.execute(query, tuple(topics)):
                kind, message_type = topics[topic]
                value = deserialize_message(raw, message_type)
                if kind == "clock":
                    clock = stamp(value.clock)
                elif kind == "motion":
                    p, v = value.pose.pose.position, value.twist.twist.linear
                    require(value.header.frame_id.lstrip("/") == "world", "unexpected ground-truth frame")
                    signals[kind].append((stamp(value.header.stamp), p.x, p.y, math.hypot(v.x, v.y)))
                elif kind in {"goal", "path", "scan"}:
                    signals[kind].append(value)
                elif kind in {"contact", "health", "arbiter"}:
                    signals[kind].append((clock, json.loads(value.data)))
                elif clock is not None:
                    signals[kind].append((clock, (value.linear.x, value.angular.z)))
    require(all(signals[kind] for kind, _ in topics.values() if kind != "clock"), "missing raw evidence topic")
    return signals


def command_matches(final, source):
    """Match all but delivery-boundary samples to the asynchronous candidate stream."""
    samples = sorted(source)
    times = [t for t, _ in samples]
    supported = 0
    for tick, command in final:
        # Each stream has its own DDS reception; the bag uses prior /clock as a
        # bound for unstamped Twist messages. Permit one normal watchdog window.
        begin = bisect_left(times, tick - 0.25)
        end = bisect_left(times, tick + 0.10)
        supported += any(command == proposal for _, proposal in samples[begin:end])
    return supported


def record_outcome(out, run):
    with (out / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1 and rows[0]["run_id"] == run["run_id"], "episode identity differs from run")
    row = rows[0]
    require(row == run["outcome"] and row["collision_known"] == "1" and row["collision"] == "0",
            "episode or contact coverage differs from the verified run")
    return row


def verified_scan_inputs(trace, bag_scans, needed):
    scans = {(msg.header.frame_id, stamp(msg.header.stamp)): msg for msg in bag_scans}
    traced = {}
    for row in read_rows(trace):
        if row["event"] == "scan" and row["stamp"] in needed:
            traced[(row["frame"], row["stamp"])] = row
    # LiDAR is best-effort. The bag and policy subscriptions can independently
    # miss an acquisition. Verify the content of each shared source stamp.
    matched = traced.keys() & scans.keys()
    require({t for _, t in traced} == needed, "policy health reports a scan absent from its trace")
    require(len(matched) / len(traced) >= 0.95, "bag lacks most of the sensor input used by policy")
    for key in matched:
        row, msg = traced[key], scans[key]
        fields = ("angle_min", "angle_increment", "range_min", "range_max")
        require(all(row[name] == getattr(msg, name) for name in fields)
                and len(row["ranges"]) == len(msg.ranges)
                and all(a == b or (math.isnan(a) and math.isnan(b))
                        for a, b in zip(map(float, row["ranges"]), msg.ranges)),
                f"trace and bag LiDAR samples differ for {key}")
    return {"unique_policy_acquisitions": len(needed), "matched_raw_bag_acquisitions": len(matched)}


def verify(out, bundle, expected_commit=None):
    run = json.loads((out / "run.json").read_text())
    require(run["status"] == "verified" and run["case"]["split"] == "train", "expected a verified train trial")
    require(expected_commit is None or run["autonomy_commit"] == expected_commit, "unexpected autonomy commit")
    for name, expected in run["bundle_sha256"].items():
        require(digest(bundle / name) == expected, f"bundle file differs from run digest: {name}")
    for node, expected in run["effective_parameter_sha256"].items():
        name = node.rsplit("/", 1)[-1]
        require(digest(out / "parameters" / f"{name}.yaml") == expected, f"parameter digest differs: {node}")
    outcome = record_outcome(out, run)
    events = read_rows(out / "control.jsonl")
    raw = bag_signals(out)
    goal = raw["goal"]
    started, ended = float(outcome["started"]), float(outcome["ended"])
    require(len(goal) == 1 and goal[0].header.frame_id == "map", "bag must contain one map goal")
    point = (goal[0].pose.position.x, goal[0].pose.position.y)
    require(math.isclose(stamp(goal[0].header.stamp), started, abs_tol=1e-6) and list(point) == run["case"]["goal"],
            "bag goal differs from the mission goal")
    require(any(path.header.frame_id == "map" and started <= stamp(path.header.stamp) <= ended
                and path.poses and math.dist(point, (path.poses[-1].pose.position.x,
                                                     path.poses[-1].pose.position.y)) <= 0.4
                for path in raw["path"]), "bag does not show a matching route")
    contact = sorted((s for _, s in raw["contact"] if started - 0.25 <= s["stamp"] <= ended + 0.25),
                     key=lambda s: s["stamp"])
    require(contact and contact[0]["stamp"] <= started and contact[-1]["stamp"] >= ended
            and all(not s["collision"] and s["source"] == "gazebo_contacts" for s in contact)
            and len({(s["coverage_epoch"], s["collision_events"]) for s in contact}) == 1
            and all(b["stamp"] - a["stamp"] <= 0.25 for a, b in zip(contact, contact[1:])),
            "raw contact witness is missing, changed or has a gap")

    mode = run["mode"]
    arbiter = Counter(s["active"] for t, s in raw["arbiter"] if t is not None and started - 0.25 <= t <= ended)
    require(mode in ("shadow", "learned"), "unknown authority mode")
    states = {"dwa"} if mode == "shadow" else {"learned", "stop"}
    require(set(arbiter) == states, f"unexpected {mode} command authority: {dict(arbiter)}")
    # Verify that moving final commands actually appeared on the active input.
    final = [(t, c) for t, c in raw["final"] if started <= t <= ended]
    moving = [(t, c) for t, c in final if abs(c[0]) >= 0.035 or abs(c[1]) >= 0.035]
    source = "dwa" if mode == "shadow" else "learned"
    matched = command_matches(moving, raw[source])
    require(len(moving) >= 5 and matched / len(moving) >= 0.95, "final command did not match the active source")
    if mode == "learned":
        require(all(0 <= v <= 0.45 and abs(w) <= 0.6 for _, (v, w) in final), "learned command cap violated")

    motion = sorted(raw["motion"])
    origin_x, origin_y = run["case"]["spawn_world"]
    moving_pose = [pose for pose in motion if started <= pose[0] <= ended]
    require(moving_pose and all(-0.25 <= x - origin_x <= 4.8 and abs(y - origin_y) <= 1.0
                                for _, x, y, _ in moving_pose), "ground truth outside the diagnostic corridor")
    closed_loop = {}
    if mode == "learned":
        request = [e["time"] for e in events if e["kind"] == "hold_requested"]
        release = [e["time"] for e in events if e["kind"] == "hold_released"]
        require(len(request) == len(release) == 1 and started < request[0] < release[0] < ended,
                "missing or unordered hold request and release")
        before = [pose for pose in moving_pose if pose[0] <= request[0]]
        require(before and max(x for _, x, _, _ in before) - origin_x >= 0.14, "no pre-hold learned motion")
        stop_time = run["control_witness"]["verified_stop"]["stamp"]
        dwell_start = stop_time - run["control_witness"]["verified_stop"]["dwell_s"]
        stopped = [pose for pose in moving_pose if dwell_start + 0.04 <= pose[0] <= stop_time]
        stop_commands = [c for t, c in final if dwell_start + 0.04 <= t <= stop_time]
        require(request[0] <= dwell_start < stop_time <= release[0] + 0.1 and stopped
                and stopped[-1][0] - stopped[0][0] >= 0.30
                and all(v < 0.02 for *_, v in stopped) and len(stop_commands) >= 3
                and all(c == (0., 0.) for c in stop_commands), "raw bag does not confirm the physical stop")
        after = [pose for pose in moving_pose if pose[0] >= release[0]]
        require(after and max(x for _, x, _, _ in after) - after[0][1] >= 0.09, "no raw evidence of resumed motion")
        closed_loop = {"before_hold_m": max(x for _, x, _, _ in before) - origin_x,
                       "stop_coverage_s": stopped[-1][0] - stopped[0][0],
                       "stop_speed_max_mps": max(v for *_, v in stopped),
                       "after_release_m": max(x for _, x, _, _ in after) - after[0][1]}
    predictions = [s for _, s in raw["health"] if started <= s["stamp"] <= ended
                   and s["healthy"] and s.get("action") is not None]
    require(predictions and all(len(s["scan_stamps"]) == len(set(s["scan_stamps"])) == 4
                                and s["scan_stamps"] == sorted(s["scan_stamps"])
                                and 0 <= s["stamp"] - s["scan_stamps"][-1] <= 0.5 + 1e-9
                                for s in predictions), "bag contains no healthy four-acquisition predictions")
    trace = out / "replay.jsonl"
    scans = verified_scan_inputs(trace, raw["scan"], {t for s in predictions for t in s["scan_stamps"]})
    with trace.open() as stream:
        reconstructed = replay(stream, PolicyRuntime.load(bundle))
    tracked = [out / "run.json", out / "launch.log", out / "control.jsonl", out / "replay.jsonl",
               out / "episodes.csv", out / "episodes.events.jsonl", *sorted((out / "bag").glob("*"))]
    return {"run_id": run["run_id"], "autonomy_commit": run["autonomy_commit"], "mode": mode,
            "termination": outcome["termination_reason"], "elapsed_sim_s": ended - started,
            "final_map_goal_error_m": float(outcome["final_goal_distance"]),
            "contact_samples": len(contact), "arbiter_states": dict(arbiter), "moving_commands": len(moving),
            "active_source_matches": matched, "healthy_commands": len(predictions), "closed_loop": closed_loop,
            "replay": reconstructed, "source_scan_comparison": scans,
            "evidence_sha256": {str(path.relative_to(out)): digest(path) for path in tracked if path.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    try:
        report = verify(args.run, args.bundle, args.expected_commit)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"M3 verification failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
