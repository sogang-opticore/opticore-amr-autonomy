"""Navigation episodes recorded from ROS topics (ROADMAP M6).

One episode is one goal: it starts when a goal is sent and ends on
``GOAL_REACHED``/``REACHED`` (success) or a timeout. Columns follow the Lite
``episodes.csv`` names so Lite and Gazebo results can be paired.

Gazebo has no contact sensor here, so ``collision`` is a proxy: the closest
LiDAR return came within the circular footprint used by Lite
(``robot_radius``) plus ``collision_clearance``.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SUCCESS_STATES = {"GOAL_REACHED", "REACHED"}


@dataclass
class _Episode:
    scenario_id: str
    seed: int
    goal: tuple[float, float]
    started: float
    start_pose: tuple[float, float] | None = None
    pose: tuple[float, float] | None = None
    path_length: float = 0.0
    minimum_clearance: float = math.inf
    collision: bool = False
    shield_ticks: int = 0
    ticks: int = 0
    fallbacks: int = 0
    fallback_since: float | None = None
    fallback_time: float = 0.0
    max_inference_ms: float = 0.0
    deadline_misses_at_start: int | None = None
    deadline_misses: int = 0
    faults: set[str] = field(default_factory=set)
    statuses: dict[str, int] = field(default_factory=dict)


class EpisodeRecorder:
    def __init__(self, label: str = "ros", robot_radius: float = 0.36, collision_clearance: float = 0.0,
                 timeout_sec: float = 120.0, lidar_mount: tuple[float, float, float] = (0.25, 0.0, 0.0)):
        self.label, self.robot_radius = label, robot_radius
        self.collision_clearance, self.timeout_sec, self.lidar_mount = collision_clearance, timeout_sec, lidar_mount
        self.current: _Episode | None = None
        self.rows: list[dict[str, Any]] = []

    def start(self, now: float, goal: tuple[float, float], scenario_id: str = "", seed: int = 0) -> None:
        if self.current is not None:
            self.finish(now, False, "PREEMPTED")
        self.current = _Episode(scenario_id or f"goal_{len(self.rows):03d}", int(seed), tuple(goal), now)

    def on_pose(self, now: float, x: float, y: float) -> None:
        episode = self.current
        if episode is None:
            return
        if episode.pose is not None:
            episode.path_length += math.dist(episode.pose, (x, y))
        if episode.start_pose is None:
            episode.start_pose = (x, y)
        episode.pose = (x, y)

    def on_scan(self, now: float, ranges: np.ndarray, angle_min: float, angle_increment: float,
                range_min: float, range_max: float) -> None:
        episode = self.current
        if episode is None:
            return
        values = np.asarray(ranges, dtype=np.float64)
        angles = angle_min + np.arange(values.size) * angle_increment + self.lidar_mount[2]
        valid = np.isfinite(values) & (values >= range_min) & (values <= range_max)
        if not np.any(valid):
            return
        x = self.lidar_mount[0] + values[valid] * np.cos(angles[valid])
        y = self.lidar_mount[1] + values[valid] * np.sin(angles[valid])
        clearance = float(np.min(np.hypot(x, y))) - self.robot_radius
        episode.minimum_clearance = min(episode.minimum_clearance, clearance)
        if clearance < self.collision_clearance:
            episode.collision = True

    def on_status(self, now: float, status: str) -> None:
        episode = self.current
        if episode is None:
            return
        episode.statuses[status] = episode.statuses.get(status, 0) + 1
        if status in SUCCESS_STATES:
            self.finish(now, True, "")

    def on_arbiter(self, now: float, snapshot: dict[str, Any]) -> None:
        episode = self.current
        if episode is None:
            return
        falling_back = snapshot.get("state") in {"FALLBACK_HOLD", "FALLBACK"} and snapshot.get("mode") == "learned"
        if falling_back and episode.fallback_since is None:
            episode.fallbacks += 1
            episode.fallback_since = now
        elif not falling_back and episode.fallback_since is not None:
            episode.fallback_time += now - episode.fallback_since
            episode.fallback_since = None

    def on_health(self, now: float, report: dict[str, Any]) -> None:
        episode = self.current
        if episode is None:
            return
        episode.ticks += 1
        episode.shield_ticks += int(report.get("shield_mode", "PASS") not in {"PASS", None})
        episode.max_inference_ms = max(episode.max_inference_ms, float(report.get("inference_ms", 0.0)))
        misses = int(report.get("deadline_misses", 0))
        if episode.deadline_misses_at_start is None:
            episode.deadline_misses_at_start = misses
        episode.deadline_misses = misses - episode.deadline_misses_at_start
        episode.faults.update(report.get("faults", []))

    def tick(self, now: float) -> dict[str, Any] | None:
        if self.current is not None and now - self.current.started > self.timeout_sec:
            return self.finish(now, False, "TIMEOUT")
        return None

    def finish(self, now: float, success: bool, failure: str) -> dict[str, Any]:
        episode = self.current
        assert episode is not None
        self.current = None
        if episode.fallback_since is not None:
            episode.fallback_time += now - episode.fallback_since
        collided = episode.collision
        if collided and success:
            failure = ""  # touched the footprint margin but still arrived; recorded in `collision`
        elif collided:
            failure = "COLLISION_PROXY"
        final = episode.pose or (math.nan, math.nan)
        row = {
            "policy": self.label, "scenario_id": episode.scenario_id, "seed": episode.seed,
            "success": int(success and not collided), "collision": int(collided),
            "truncated": int(failure == "TIMEOUT"), "failure_type": failure,
            "episode_time": now - episode.started, "path_length": episode.path_length,
            "minimum_clearance": episode.minimum_clearance if math.isfinite(episode.minimum_clearance) else math.nan,
            "shield_intervention_rate": episode.shield_ticks / max(1, episode.ticks),
            "fallback_count": episode.fallbacks, "fallback_time": episode.fallback_time,
            "max_inference_ms": episode.max_inference_ms, "deadline_misses": episode.deadline_misses,
            "faults": ";".join(sorted(episode.faults)),
            "goal_x": episode.goal[0], "goal_y": episode.goal[1],
            "start_x": (episode.start_pose or (math.nan, math.nan))[0],
            "start_y": (episode.start_pose or (math.nan, math.nan))[1],
            "final_x": final[0], "final_y": final[1],
            "final_goal_distance": math.dist(final, episode.goal) if episode.pose else math.nan,
        }
        self.rows.append(row)
        return row

    def write_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.rows:
            with target.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        return target
