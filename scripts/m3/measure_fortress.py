#!/usr/bin/env python3
"""Measure the static-transfer sensor and motion envelope from finalized Fortress bags."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median

import numpy as np

THRESHOLD = 0.05


def seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def read(connection, topic, message_type, deserialize):
    query = (
        "SELECT timestamp,data FROM messages JOIN topics ON topics.id=messages.topic_id "
        "WHERE topics.name=? ORDER BY timestamp"
    )
    for timestamp, data in connection.execute(query, (topic,)):
        yield timestamp, deserialize(data, message_type)


def trial(directory):
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from rclpy.serialization import deserialize_message
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import LaserScan
    import yaml

    run = json.loads((directory / "run.json").read_text())
    if run["status"] != "recorded" or run["scenario_id"] != "train-static":
        raise ValueError(f"{directory} is not a finalized static-train run")
    goal_stamp = run["goal_stamp"]
    metadata = yaml.safe_load((directory / "bag/metadata.yaml").read_text())
    bag_files = metadata["rosbag2_bagfile_information"]["relative_file_paths"]
    if len(bag_files) != 1:
        raise ValueError(f"{directory} requires one SQLite bag")
    path = directory / "bag" / bag_files[0]
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        clocks = [(t, seconds(m.clock)) for t, m in read(connection, "/clock", Clock, deserialize_message)]
        clock_times = [t for t, _ in clocks]
        commands = []
        for wall_time, message in read(connection, "/amr1/cmd_vel", Twist, deserialize_message):
            clock_index = bisect_right(clock_times, wall_time) - 1
            if clock_index >= 0 and clocks[clock_index][1] >= goal_stamp:
                commands.append((clocks[clock_index][1], message.linear.x))
        motion = []
        for _, message in read(connection, "/amr1/ground_truth", Odometry, deserialize_message):
            speed = message.twist.twist.linear
            motion.append((seconds(message.header.stamp), math.hypot(speed.x, speed.y)))
        scans = [(seconds(m.header.stamp), np.asarray(m.ranges)) for _, m in
                 read(connection, "/amr1/lidar", LaserScan, deserialize_message)]
    motion = [(t, v) for t, v in sorted(motion) if t >= goal_stamp]
    command_time = next(t for t, v in commands if v >= THRESHOLD)
    motion_time = next(t for t, v in motion if v >= THRESHOLD)
    onset_end = next(t for t, v in motion if v >= 1.0)
    # Estimate the ramp between the first observed 0.05 and 1.0 m/s samples.
    rise = onset_end - motion_time
    acceleration = (1.0 - THRESHOLD) / rise
    times = [t for t, _ in motion]
    last_cruise = max(i for i, (_, v) in enumerate(commands) if v >= 1.0)
    slow_time = next(t for t, _ in commands[last_cruise + 1:])
    slow_index = bisect_left(times, slow_time)
    # Peak deceleration over a 100 ms window during final approach.
    brakes = []
    for i in range(slow_index, len(motion)):
        start, speed = motion[i]
        if start > slow_time + 3:
            break
        j = bisect_left(times, start + 0.1 - 1e-9)
        if j < len(motion):
            brakes.append((speed - motion[j][1]) / (motion[j][0] - start))
    before_goal = [v for t, v in scans if goal_stamp - 1 <= t < goal_stamp]
    stationary = np.stack(before_goal)
    previous, current = stationary[:-1], stationary[1:]
    finite = np.isfinite(previous) & np.isfinite(current)
    noise = float(np.std(current[finite] - previous[finite]) / math.sqrt(2))
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {
        "trial": directory.name,
        "autonomy_commit": run["autonomy_commit"],
        "bag_sha256": checksum.hexdigest(),
        "lidar_period_s": median(np.diff([t for t, _ in scans if t >= goal_stamp])),
        "ground_truth_period_s": median(np.diff(times)),
        "clock_period_s": median(np.diff([t for _, t in clocks])),
        "scan_beams": len(scans[0][1]),
        "forward_response_s": motion_time - command_time,
        "forward_ramp_mps2": acceleration,
        "approach_peak_deceleration_mps2": max(brakes) if brakes else None,
        "stationary_range_noise_std_m": noise,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    evidence = [trial(path) for path in args.runs]
    print(json.dumps({"speed_threshold_mps": THRESHOLD, "runs": evidence}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
