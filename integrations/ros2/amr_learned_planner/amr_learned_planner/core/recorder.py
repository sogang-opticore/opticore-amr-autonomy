"""Goal-scoped, time-aligned ROS navigation evidence. All sample times use the ROS clock."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUCCESS_STATES = {"GOAL_REACHED", "REACHED"}
ACTIVE_STATES = {"NORMAL", "ALIGN", "REJOIN", "SPIN", "FORWARD_ONLY", "EMERGENCY", "AVOID"}
CONTACT_SOURCES = {"gazebo_contacts", "ground_truth_geometry"}


@dataclass
class _Episode:
    scenario_id: str
    seed: int
    goal: tuple[float, float]
    started: float
    goal_id: str
    episode_id: str
    frame: str
    last_clock: float
    wall_started: float = field(default_factory=time.monotonic)
    start_pose: tuple[float, float] | None = None
    pose: tuple[float, float] | None = None
    pose_stamp: float = -math.inf
    path_stamp: float | None = None
    armed: bool = False
    path_length: float = 0.0
    minimum_clearance: float = math.inf
    proximity_warning: bool = False
    contact_start: float | None = None
    contact_end: float | None = None
    contact_source: str = ""
    contact_count: int | None = None
    coverage_epoch: int | None = None
    contact_gap: bool = False
    contacted: bool = False
    arrived_at: float | None = None
    shield_ticks: int = 0
    ticks: int = 0
    fallbacks: int = 0
    fallback_since: float | None = None
    fallback_time: float = 0.0
    max_inference_ms: float = 0.0
    deadline_misses_at_start: int | None = None
    deadline_misses: int = 0
    faults: set[str] = field(default_factory=set)


class EpisodeRecorder:
    def __init__(self, label: str = "ros", robot_radius: float = 0.36, collision_clearance: float = 0.0,
                 timeout_sec: float = 120.0, lidar_mount: tuple[float, float, float] = (0.25, 0.0, 0.0),
                 *, run_id: str = "", goal_tolerance: float = 0.25, max_pose_age: float = 0.5,
                 contact_max_gap: float = 0.25, events_path: str | Path | None = None):
        self.label, self.robot_radius = label, robot_radius
        self.collision_clearance, self.timeout_sec, self.lidar_mount = collision_clearance, timeout_sec, lidar_mount
        self.goal_tolerance, self.max_pose_age, self.contact_max_gap = goal_tolerance, max_pose_age, contact_max_gap
        if not all(math.isfinite(x) and x > 0 for x in (timeout_sec, goal_tolerance, max_pose_age, contact_max_gap)):
            raise ValueError("timeout and evidence tolerances must be positive and finite")
        self.run_id = run_id or uuid.uuid4().hex
        self.current: _Episode | None = None
        self.rows: list[dict[str, Any]] = []
        self.last_contact: tuple[float, bool, str, int | None, int | None] | None = None
        self.goal_ids: set[str] = set()
        self.events = None
        if events_path is not None:
            path = Path(events_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self.events = path.open("x", encoding="utf-8", buffering=1)

    def event(self, now: float, kind: str, **data: Any) -> None:
        if self.events is not None:
            entry = {"version": 2, "run_id": self.run_id, "episode_id": self.current.episode_id if self.current else None,
                     "goal_id": self.current.goal_id if self.current else None,
                     "time": now, "wall_monotonic": time.monotonic(), "kind": kind, **data}
            self.events.write(json.dumps(entry, allow_nan=False) + "\n")

    def start(self, now: float, goal: tuple[float, float], scenario_id: str = "", seed: int = 0,
              *, goal_id: str = "", frame: str = "map", received_at: float | None = None) -> None:
        arrival = now if received_at is None else received_at
        if not frame or not all(math.isfinite(v) for v in (*goal, now, arrival)):
            return
        episode_id = f"{self.run_id}:{len(self.rows) + int(self.current is not None):04d}"
        goal_id = goal_id or episode_id
        if goal_id in self.goal_ids:
            return
        if self.current is not None:
            self.finish(now, "PREEMPTED")
        self.goal_ids.add(goal_id)
        self.current = e = _Episode(scenario_id or f"goal_{len(self.rows):03d}", int(seed), tuple(goal), now,
                                    goal_id, episode_id, frame, arrival)
        self.event(now, "goal", received_at=arrival, frame=frame, x=goal[0], y=goal[1], scenario_id=e.scenario_id, seed=seed,
                   contact_max_gap=self.contact_max_gap, goal_tolerance=self.goal_tolerance,
                   max_pose_age=self.max_pose_age, timeout_sec=self.timeout_sec)
        if self.last_contact is not None:
            stamp, hit, source, count, epoch = self.last_contact
            if 0 <= now - stamp <= self.contact_max_gap:
                e.contact_start = e.contact_end = stamp
                e.contact_source, e.contact_count, e.coverage_epoch = source, count, epoch
                # A pre-goal contact may have ended before dispatch. Only an in-window observation proves collision.
                if hit and stamp < now:
                    e.contact_gap = True
                if stamp == now:
                    e.contacted = hit
                self.event(now, "initial_contact", stamp=stamp, collision=hit, source=source, collision_events=count, coverage_epoch=epoch)
                if e.contacted:
                    self.finish(stamp, "COLLISION")

    def on_pose(self, stamp: float, x: float, y: float, *, frame: str = "map") -> None:
        e = self.current
        if e is None or not all(math.isfinite(v) for v in (x, y, stamp)):
            return
        if frame != e.frame or stamp < e.started or stamp <= e.pose_stamp:
            return
        if e.pose is not None:
            e.path_length += math.dist(e.pose, (x, y))
        if e.start_pose is None:
            e.start_pose = (x, y)
        e.pose, e.pose_stamp = (x, y), stamp
        self.event(stamp, "pose", x=x, y=y, frame=frame)

    def on_path(self, stamp: float, end: tuple[float, float] | None, frame: str) -> None:
        e = self.current
        if e is None or stamp < e.started or (e.path_stamp is not None and stamp < e.path_stamp):
            return
        matches = (end is not None and frame == e.frame and all(math.isfinite(v) for v in end)
                   and math.dist(end, e.goal) <= self.goal_tolerance)
        self.event(stamp, "path", matching=matches, frame=frame)
        if not matches:
            e.armed = False
        e.path_stamp = stamp if matches else None

    def on_scan(self, now: float, ranges: np.ndarray, angle_min: float, angle_increment: float,
                range_min: float, range_max: float) -> None:
        e = self.current
        if e is None or now < e.started:
            return
        values = np.asarray(ranges, dtype=np.float64)
        angles = angle_min + np.arange(values.size) * angle_increment + self.lidar_mount[2]
        valid = np.isfinite(values) & (values >= range_min) & (values <= range_max)
        if not np.any(valid):
            return
        x = self.lidar_mount[0] + values[valid] * np.cos(angles[valid])
        y = self.lidar_mount[1] + values[valid] * np.sin(angles[valid])
        clearance = float(np.min(np.hypot(x, y))) - self.robot_radius
        if math.isfinite(clearance):
            e.minimum_clearance = min(e.minimum_clearance, clearance)
            e.proximity_warning |= clearance < self.collision_clearance
            self.event(now, "scan", minimum_clearance=clearance)

    def on_contact(self, stamp: float, hit: bool, source: str, pairs: Any = None, count: int | None = None,
                   epoch: int | None = None) -> None:
        if source not in CONTACT_SOURCES or type(hit) is not bool or not math.isfinite(stamp):
            return
        if any(v is not None and (type(v) is not int or v < 0) for v in (count, epoch)):
            return
        if source == "gazebo_contacts" and (count is None or epoch is None):
            return
        if self.last_contact is not None and stamp <= self.last_contact[0]:
            return
        self.last_contact = (stamp, hit, source, count, epoch)
        e = self.current
        if e is None or stamp < e.started:
            return
        if e.contact_end is not None:
            e.contact_gap |= stamp - e.contact_end > self.contact_max_gap or e.contact_source != source
            e.contact_gap |= e.coverage_epoch != epoch
        if e.contact_start is None:
            e.contact_start = stamp
        increased = False
        if count is not None and e.contact_count is not None:
            e.contact_gap |= count < e.contact_count
            increased = count > e.contact_count
        prior_stamp = e.contact_end
        e.contact_count, e.contact_end, e.contact_source, e.coverage_epoch = count, stamp, source, epoch
        self.event(stamp, "contact", collision=hit, source=source, pairs=pairs, collision_events=count, coverage_epoch=epoch)
        until = e.arrived_at if e.arrived_at is not None else e.started + self.timeout_sec
        # A counter step spanning a trial boundary cannot locate the impact on either side.
        if increased and (prior_stamp is None or prior_stamp < e.started or stamp > until):
            e.contact_gap = True
        elif increased:
            hit = True
        if hit and stamp > until:
            e.contact_gap = True
        if hit and stamp <= until:
            e.contacted = True
            self.finish(stamp, "COLLISION")
        elif e.arrived_at is not None and stamp >= e.arrived_at:
            self.finish(e.arrived_at, "ARRIVED")

    def on_status(self, now: float, status: str) -> None:
        e = self.current
        if e is None or now < e.started or e.arrived_at is not None:
            return
        self.event(now, "status", value=status)
        if now >= e.started + self.timeout_sec or e.path_stamp is None or now < e.path_stamp:
            return
        if status in ACTIVE_STATES:
            e.armed = True
        if (status in SUCCESS_STATES and e.armed and e.pose is not None and
                0 <= now - e.pose_stamp <= self.max_pose_age and math.dist(e.pose, e.goal) <= self.goal_tolerance):
            e.arrived_at = now
            if e.contact_end is not None and e.contact_end >= now:
                self.finish(now, "ARRIVED")

    def on_arbiter(self, now: float, snapshot: dict[str, Any]) -> None:
        e = self.current
        if e is None:
            return
        if not isinstance(snapshot, dict) or not all(isinstance(snapshot.get(k), str) for k in ("state", "mode")):
            self.event(now, "malformed_arbiter")
            return
        self.event(now, "arbiter", state=snapshot["state"], mode=snapshot["mode"])
        fallback = snapshot.get("state") in {"FALLBACK_HOLD", "FALLBACK"} and snapshot.get("mode") == "learned"
        if fallback and e.fallback_since is None:
            e.fallbacks += 1
            e.fallback_since = now
        elif not fallback and e.fallback_since is not None:
            e.fallback_time += now - e.fallback_since
            e.fallback_since = None

    def on_health(self, now: float, report: dict[str, Any]) -> None:
        e = self.current
        if e is None:
            return
        if not isinstance(report, dict):
            self.event(now, "malformed_health")
            return
        duration = report.get("inference_ms", 0.0)
        misses = report.get("deadline_misses", 0)
        faults = report.get("faults", [])
        if (type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0
                or type(misses) is not int or misses < 0 or not isinstance(faults, list)
                or not all(isinstance(fault, str) for fault in faults)
                or not isinstance(report.get("shield_mode", "PASS"), (str, type(None)))):
            self.event(now, "malformed_health")
            return
        e.ticks += 1
        e.shield_ticks += int(report.get("shield_mode", "PASS") not in {"PASS", None})
        e.max_inference_ms = max(e.max_inference_ms, duration)
        if e.deadline_misses_at_start is None:
            e.deadline_misses_at_start = misses
        e.deadline_misses = max(0, misses - e.deadline_misses_at_start)
        e.faults.update(faults)
        self.event(now, "health", shield=report.get("shield_mode"), faults=report.get("faults", []),
                   deadline_misses=misses, inference_ms=duration)

    def tick(self, now: float) -> dict[str, Any] | None:
        e = self.current
        if e is None:
            return None
        if now < e.last_clock:
            self.last_contact = None
            return self.finish(max(e.started, e.last_clock), "CLOCK_RESET")
        e.last_clock = now
        if e.arrived_at is not None:
            if now - e.arrived_at >= self.contact_max_gap:
                return self.finish(e.arrived_at, "ARRIVED")
        else:
            end = e.started + self.timeout_sec
            if now >= end and ((e.contact_end is not None and e.contact_end >= end) or now >= end + self.contact_max_gap):
                return self.finish(end, "TIMEOUT")
        return None

    def finish(self, now: float, reason: str) -> dict[str, Any]:
        e = self.current
        assert e is not None
        if e.fallback_since is not None:
            e.fallback_time += max(0.0, now - e.fallback_since)
        coverage = (e.contact_start is not None and e.contact_start <= e.started
                    and e.contact_end is not None and e.contact_end >= now and not e.contact_gap)
        known = e.contacted or coverage
        arrived = reason == "ARRIVED"
        failure = "" if arrived and known and not e.contacted else "CONTACT_UNKNOWN" if arrived else reason
        final = e.pose or (math.nan, math.nan)
        row = {
            "policy": self.label, "scenario_id": e.scenario_id, "seed": e.seed,
            "success": int(arrived and known and not e.contacted), "collision": int(e.contacted) if known else None,
            "truncated": int(reason == "TIMEOUT"), "failure_type": failure,
            "episode_time": max(0.0, now - e.started), "path_length": e.path_length,
            "minimum_clearance": e.minimum_clearance if math.isfinite(e.minimum_clearance) else math.nan,
            "shield_intervention_rate": e.shield_ticks / e.ticks if e.ticks else math.nan,
            "fallback_count": e.fallbacks, "fallback_time": e.fallback_time,
            "max_inference_ms": e.max_inference_ms if e.ticks else math.nan, "deadline_misses": e.deadline_misses,
            "faults": ";".join(sorted(e.faults)), "goal_x": e.goal[0], "goal_y": e.goal[1],
            "start_x": (e.start_pose or (math.nan, math.nan))[0], "start_y": (e.start_pose or (math.nan, math.nan))[1],
            "final_x": final[0], "final_y": final[1], "final_goal_distance": math.dist(final, e.goal),
            "version": 2, "run_id": self.run_id, "goal_id": e.goal_id, "episode_id": e.episode_id, "frame": e.frame,
            "termination_reason": reason, "arrived": int(arrived), "collision_known": int(known),
            "contact_source": e.contact_source, "proximity_warning": int(e.proximity_warning),
            "started": e.started, "ended": now, "wall_elapsed": max(0.0, time.monotonic() - e.wall_started),
        }
        self.event(now, "end", reason=reason, collision=row["collision"], success=row["success"])
        self.rows.append(row)
        self.current = None
        return row

    def close(self, now: float) -> None:
        if self.current:
            self.finish(max(self.current.started, now), "INTERRUPTED")
        if self.events is not None:
            self.events.close()

    def write_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False, newline="", encoding="utf-8") as f:
                try:
                    writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
                    writer.writeheader()
                    writer.writerows(self.rows)
                    f.flush()
                    os.replace(f.name, target)
                finally:
                    Path(f.name).unlink(missing_ok=True)
        return target
