"""Write and replay the inputs that reached the M3 observation/command boundary."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import TextIO

import numpy as np

from amr_lite.deploy.runtime import GOLDEN_TOLERANCE, PolicyRuntime
from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.stamped_io import M3_IO_ID, CommandSample, ObservationSample, StampedPolicyIO

OBSERVATION_TOLERANCE = 1e-7
COMMAND_TOLERANCE = 1e-7


def _number(value: float) -> float | str:
    """Retain non-finite raw inputs (including valid LiDAR no-returns) in valid JSON."""
    return float(value) if math.isfinite(value) else str(float(value))


def _action(action):
    if action is None:
        return None
    array = np.asarray(action)
    if array.ndim == 0:
        return _number(array)
    return [_action(value) for value in array]


def _pair_error(actual, expected) -> float:
    if actual is None or expected is None:
        return 0.0 if actual is expected else math.inf
    if len(expected) != 2:
        return math.inf
    return max(abs(a - b) for a, b in zip(actual, expected))


class ReplayTrace:
    """An ordered JSONL stream of scans, resets, path inputs and policy ticks."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self._path = None
        self._write("contract", id=M3_IO_ID, version=1)

    def _write(self, event: str, **data) -> None:
        self.stream.write(json.dumps({"event": event, **data}, allow_nan=False, separators=(",", ":")) + "\n")

    def scan(self, namespace: str, frame: str, stamp: float, ranges,
             angle_min: float, angle_increment: float, range_min: float, range_max: float) -> None:
        self._write("scan", namespace=namespace, frame=frame, stamp=_number(stamp),
                    ranges=[_number(v) for v in ranges], angle_min=_number(angle_min),
                    angle_increment=_number(angle_increment), range_min=_number(range_min),
                    range_max=_number(range_max))

    def reset(self) -> None:
        self._path = None
        self._write("reset")

    def tick(self, now: float, odom_stamp: float, robot: RobotState, path,
             observation: ObservationSample | None, action, command: CommandSample | None, *,
             goal_id, path_revision, segment_range=None, inference_ms: float = 0.0,
             deadline_ms: float = 25.0) -> None:
        points = tuple((x, y) for x, y in path)
        if points != self._path:
            self._write("path", points=[[_number(x), _number(y)] for x, y in points])
            self._path = points
        self._write("tick", now=_number(now), odom_stamp=_number(odom_stamp),
                    robot={k: _number(v) for k, v in asdict(robot).items()},
                    goal_id=goal_id, path_revision=path_revision, segment_range=segment_range,
                    inference_ms=_number(inference_ms), deadline_ms=_number(deadline_ms),
                    action=_action(action), command=asdict(command) if command else None,
                    observation=(
                        {"fault": observation.fault, "stamps": observation.stamps,
                         "tensor": observation.tensor.tolist() if observation.tensor is not None else None}
                        if observation else None))


def replay(stream: TextIO, runtime: PolicyRuntime | None = None) -> dict:
    """Rebuild each observation and physical command and compare to the recorded values."""
    first = json.loads(stream.readline())
    if first != {"event": "contract", "id": M3_IO_ID, "version": 1}:
        raise ValueError("unsupported M3 replay contract")
    sensor, path = StampedPolicyIO(), []
    counts, faults = Counter(), Counter()
    max_observation_error = 0.0
    max_action_error = 0.0
    max_target_error = max_command_error = 0.0
    for line_number, line in enumerate(stream, start=2):
        row = json.loads(line)
        event = row["event"]
        counts[event] += 1
        if event == "reset":
            sensor.reset()
            path = []
        elif event == "path":
            path = [tuple(map(float, point)) for point in row["points"]]
        elif event == "scan":
            sensor.ingest_scan(row["namespace"], row["frame"], float(row["stamp"]),
                               [float(v) for v in row["ranges"]], float(row["angle_min"]),
                               float(row["angle_increment"]), float(row["range_min"]), float(row["range_max"]))
        elif event == "tick":
            robot = RobotState(**{k: float(v) for k, v in row["robot"].items()})
            expected = row["observation"]
            action = row["action"]
            if expected is not None:
                actual = sensor.observe(float(row["now"]), float(row["odom_stamp"]), robot, path,
                                        segment_range=row["segment_range"])
                tensor = expected["tensor"]
                if (actual.fault != expected["fault"] or list(actual.stamps) != expected["stamps"]
                        or (actual.tensor is None) != (tensor is None)):
                    raise ValueError(f"line {line_number} has a different observation or scan history")
                if actual.tensor is not None:
                    expected_tensor = np.asarray(tensor, dtype=np.float32)
                    if expected_tensor.shape != actual.tensor.shape:
                        raise ValueError(f"line {line_number} has a different observation shape")
                    error = float(np.max(np.abs(actual.tensor - expected_tensor)))
                    if not error <= OBSERVATION_TOLERANCE:
                        raise ValueError(f"line {line_number} has a different observation")
                    max_observation_error = max(max_observation_error, error)
                    counts["valid_observation"] += 1
                elif actual.fault is not None:
                    faults[actual.fault] += 1
                if runtime is not None and actual.tensor is not None and action is not None:
                    recorded_action = np.asarray(action, dtype=np.float32)
                    if recorded_action.shape != (2,):
                        raise ValueError(f"line {line_number} has a different policy output shape")
                    error = float(np.max(np.abs(runtime.predict(actual.tensor) - recorded_action)))
                    if not error <= GOLDEN_TOLERANCE:
                        raise ValueError(f"line {line_number} has a different policy output")
                    max_action_error = max(max_action_error, error)
                    counts["policy_prediction"] += 1
            if row["command"] is not None:
                expected = row["command"]
                result = sensor.actuate(action, robot, inference_ms=float(row["inference_ms"]),
                                        deadline_ms=float(row["deadline_ms"]))
                target_error = _pair_error(result.target, expected["target"])
                command_error = _pair_error(result.command, expected["command"])
                if (result.fault != expected["fault"] or not target_error <= COMMAND_TOLERANCE
                        or not command_error <= COMMAND_TOLERANCE):
                    raise ValueError(f"line {line_number} has a different physical command")
                max_target_error = max(max_target_error, target_error)
                max_command_error = max(max_command_error, command_error)
                counts["physical_command"] += 1
                if result.fault:
                    faults[result.fault] += 1
        else:
            raise ValueError(f"line {line_number} has unknown event {event!r}")
    if not counts["valid_observation"] or not counts["physical_command"]:
        raise ValueError("replay did not exercise both an observation and a physical command")
    if runtime is not None and not counts["policy_prediction"]:
        raise ValueError("replay did not exercise a bundle prediction")
    return {"contract": M3_IO_ID, "counts": dict(counts), "faults": dict(faults),
            "max_observation_error": max_observation_error, "max_target_error": max_target_error,
            "max_command_error": max_command_error,
            "max_action_error": max_action_error if runtime is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--bundle", type=Path, help="also verify predictions with the saved NumPy bundle")
    args = parser.parse_args()
    runtime = PolicyRuntime.load(args.bundle) if args.bundle else None
    with args.trace.open() as stream:
        print(json.dumps(replay(stream, runtime), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
