"""LiDAR-only Safety Shield and obstacle tracker.

Unlike the ``legacy`` and ``predictive`` shields, nothing here reads the
simulator's ground truth: the inputs are LiDAR returns in the robot frame,
the robot's measured velocity and an odometry pose. The ROS learned planner
imports these classes unchanged, so the shield evaluated in Lite is the one
that runs on the robot (ARCHITECTURE §5.2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from amr_lite.envs.dynamics import RobotState, integrate, limit_velocity

from .safety_shield import ShieldResult

Pose = tuple[float, float, float]

SCAN_SHIELD_DEFAULTS: dict[str, Any] = {
    "horizon": 0.7,
    "sim_dt": 0.05,
    "static_margin": 0.03,
    "dynamic_horizon": 1.5,
    "dynamic_margin": 0.08,
    "slow_scales": [0.5, 0.2],
    "evade_v_fraction": 0.45,
    "evade_w_fraction": 0.65,
    "reverse_enabled": False,
    "reverse_max": 0.35,
    "reverse_w": 0.8,
    "max_point_range": 4.0,
    "use_tracks": True,
    "track_min_speed": 0.15,
}

SCAN_TRACKER_DEFAULTS: dict[str, Any] = {
    "cluster_gap": 0.35,
    "min_points": 3,
    "max_radius": 0.85,
    "association_distance": 0.9,
    "timeout": 1.0,
    "velocity_alpha": 0.5,
    "max_speed": 2.0,
}


def transform_points(points: np.ndarray, source: Pose, target: Pose) -> np.ndarray:
    """Move robot-frame points observed at pose ``source`` into the frame of ``target``."""
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    cs, ss = math.cos(source[2]), math.sin(source[2])
    wx = source[0] + cs * points[:, 0] - ss * points[:, 1]
    wy = source[1] + ss * points[:, 0] + cs * points[:, 1]
    ct, st = math.cos(target[2]), math.sin(target[2])
    dx, dy = wx - target[0], wy - target[1]
    return np.stack((ct * dx + st * dy, -st * dx + ct * dy), axis=1)


@dataclass
class ScanTrack:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    radius: float
    age: int
    last_seen: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class ScanObstacleTracker:
    """Clusters consecutive returns and tracks cluster centroids in the odom frame.

    Large clusters (walls, shelves) are ignored; an optional ``static_filter``
    callable ``(x, y) -> bool`` drops clusters that belong to the static map.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = {**SCAN_TRACKER_DEFAULTS, **(config or {})}
        self.tracks: list[ScanTrack] = []
        self._next_id = 0

    def reset(self) -> None:
        self.tracks = []
        self._next_id = 0

    def clusters(self, world_points: np.ndarray) -> list[tuple[float, float, float, int]]:
        """``(cx, cy, radius, count)`` of consecutive-point clusters."""
        if len(world_points) == 0:
            return []
        gaps = np.hypot(*np.diff(world_points, axis=0).T) > float(self.config["cluster_gap"])
        boundaries = np.flatnonzero(gaps) + 1
        output = []
        for group in np.split(world_points, boundaries):
            if len(group) < int(self.config["min_points"]):
                continue
            centre = group.mean(axis=0)
            radius = float(np.max(np.hypot(*(group - centre).T)))
            if radius <= float(self.config["max_radius"]):
                output.append((float(centre[0]), float(centre[1]), radius, len(group)))
        return output

    def update(self, robot_points: np.ndarray, pose: Pose, time: float,
               static_filter: Callable[[float, float], bool] | None = None) -> list[ScanTrack]:
        world = transform_points(np.asarray(robot_points, dtype=np.float64), pose, (0.0, 0.0, 0.0))
        detections = [item for item in self.clusters(world)
                      if static_filter is None or not static_filter(item[0], item[1])]
        gate = float(self.config["association_distance"])
        pairs = sorted(
            (math.hypot(det[0] - track.x, det[1] - track.y), d_index, t_index)
            for d_index, det in enumerate(detections) for t_index, track in enumerate(self.tracks))
        used_detections: set[int] = set()
        used_tracks: set[int] = set()
        alpha, max_speed = float(self.config["velocity_alpha"]), float(self.config["max_speed"])
        for distance, d_index, t_index in pairs:
            if distance > gate or d_index in used_detections or t_index in used_tracks:
                continue
            track, (cx, cy, radius, _) = self.tracks[t_index], detections[d_index]
            dt = time - track.last_seen
            if dt > 1e-6:
                vx = alpha * (cx - track.x) / dt + (1.0 - alpha) * track.vx
                vy = alpha * (cy - track.y) / dt + (1.0 - alpha) * track.vy
                speed = math.hypot(vx, vy)
                if speed > max_speed:
                    vx, vy = vx * max_speed / speed, vy * max_speed / speed
                track.vx, track.vy = vx, vy
            track.x, track.y, track.radius = cx, cy, max(radius, 0.05)
            track.age += 1
            track.last_seen = time
            used_detections.add(d_index)
            used_tracks.add(t_index)
        timeout = float(self.config["timeout"])
        self.tracks = [track for index, track in enumerate(self.tracks)
                       if index in used_tracks or time - track.last_seen <= timeout]
        for index, (cx, cy, radius, _) in enumerate(detections):
            if index not in used_detections:
                self.tracks.append(ScanTrack(self._next_id, cx, cy, 0.0, 0.0, max(radius, 0.05), 1, time))
                self._next_id += 1
        return self.tracks

    def relative_tracks(self, pose: Pose, time: float, min_speed: float = 0.0,
                        min_age: int = 2) -> list[tuple[float, float, float, float, float]]:
        """Moving tracks as ``(x, y, vx, vy, radius)`` in the robot frame at ``pose``."""
        c, s = math.cos(pose[2]), math.sin(pose[2])
        output = []
        for track in self.tracks:
            if track.age < min_age or track.speed < min_speed:
                continue
            ahead = time - track.last_seen
            x, y = track.x + track.vx * ahead - pose[0], track.y + track.vy * ahead - pose[1]
            output.append((c * x + s * y, -s * x + c * y, c * track.vx + s * track.vy,
                           -s * track.vx + c * track.vy, track.radius))
        return output


class ScanSafetyShield:
    """Rollout check of candidate commands against LiDAR points and track predictions.

    Candidates are tried in order PASS -> SLOW -> EVADE -> (REVERSE) -> STOP.
    Each rollout applies the robot profile's acceleration limits, so every
    returned command is reachable from the measured velocity. If no candidate
    is safe, the one with the largest predicted clearance is executed.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.settings = {**SCAN_SHIELD_DEFAULTS, **(config.get("scan_shield") or {})}

    def _limit(self, state: RobotState, target_v: float, target_w: float, dt: float,
               allow_reverse: bool) -> tuple[float, float]:
        reverse = float(self.settings["reverse_max"]) if allow_reverse else 0.0
        return limit_velocity(state.v, state.w, target_v, target_w, dt,
                              float(self.config["v_max"]), float(self.config["w_max"]),
                              float(self.config["accel_max"]), float(self.config["decel_max"]),
                              float(self.config["angular_accel_max"]), v_min=-reverse)

    def assess(self, state: RobotState, target_v: float, target_w: float, points: np.ndarray,
               tracks: Sequence[tuple[float, float, float, float, float]] = (),
               allow_reverse: bool = False) -> tuple[bool, str, float, float]:
        """``(unsafe, source, time_to_violation, minimum_clearance)`` of a target."""
        dt = float(self.settings["sim_dt"])
        static_horizon, dynamic_horizon = float(self.settings["horizon"]), float(self.settings["dynamic_horizon"])
        steps = max(1, int(math.ceil(max(static_horizon, dynamic_horizon if tracks else 0.0) / dt)))
        probe = RobotState(0.0, 0.0, 0.0, state.v, state.w)
        poses = np.empty((steps, 2), dtype=np.float64)
        for index in range(steps):
            v, w = self._limit(probe, target_v, target_w, dt, allow_reverse)
            probe = integrate(probe, v, w, dt)
            poses[index] = (probe.x, probe.y)
        times = (np.arange(steps) + 1) * dt
        radius = float(self.config["robot_radius"])
        first_source, first_time, minimum = "NONE", math.inf, math.inf
        static_steps = int(np.sum(times <= static_horizon + 1e-9))
        if len(points) and static_steps:
            # Only returns the footprint can reach within the static horizon matter.
            reach = (float(np.max(np.hypot(poses[:static_steps, 0], poses[:static_steps, 1])))
                     + radius + float(self.settings["static_margin"]) + 0.05)
            near = points[np.hypot(points[:, 0], points[:, 1])
                          <= min(reach, float(self.settings["max_point_range"]))]
            if len(near):
                distance = np.hypot(poses[:static_steps, None, 0] - near[None, :, 0],
                                    poses[:static_steps, None, 1] - near[None, :, 1]).min(axis=1) - radius
                minimum = float(distance.min())
                violated = np.flatnonzero(distance <= float(self.settings["static_margin"]))
                if violated.size:
                    first_source, first_time = "SCAN_STATIC", float(times[violated[0]])
        for x, y, vx, vy, track_radius in tracks:
            reach = times <= dynamic_horizon + 1e-9
            clearance = np.hypot(poses[reach, 0] - (x + vx * times[reach]),
                                 poses[reach, 1] - (y + vy * times[reach])) - radius - track_radius
            if clearance.size:
                minimum = min(minimum, float(clearance.min()))
                violated = np.flatnonzero(clearance <= float(self.settings["dynamic_margin"]))
                if violated.size and times[violated[0]] < first_time:
                    first_source, first_time = "SCAN_TRACK", float(times[violated[0]])
        return first_source != "NONE", first_source, first_time, minimum

    def apply(self, state: RobotState, target_v: float, target_w: float, points: np.ndarray,
              tracks: Sequence[tuple[float, float, float, float, float]] = (),
              enabled: bool = True) -> ShieldResult:
        if not math.isfinite(target_v) or not math.isfinite(target_w):
            return ShieldResult(0.0, 0.0, "STOP", True, "STOP", "INVALID_ACTION")
        policy_dt = float(self.config["policy_dt"])
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        unsafe, source, ttc, clearance = self.assess(state, target_v, target_w, points, tracks)
        first_v, first_w = self._limit(state, target_v, target_w, policy_dt, False)
        if not enabled or not unsafe:
            return ShieldResult(first_v, first_w, "PASS", unsafe, "PASS", source, ttc,
                                predicted_minimum_clearance=clearance)
        settings = self.settings
        candidates: list[tuple[str, float, float, bool]] = [
            ("SLOW", target_v * float(scale), target_w, False) for scale in settings["slow_scales"]]
        evade_v = min(max(0.0, target_v), float(settings["evade_v_fraction"]) * float(self.config["v_max"]))
        evade_w = float(settings["evade_w_fraction"]) * float(self.config["w_max"])
        candidates += [("EVADE", evade_v, direction * evade_w, False) for direction in (1.0, -1.0)]
        if bool(settings["reverse_enabled"]):
            back, turn = -float(settings["reverse_max"]), float(settings["reverse_w"])
            candidates += [("REVERSE", back, w, True) for w in (0.0, turn, -turn)]
        candidates.append(("STOP", 0.0, 0.0, False))
        chosen, rejected = None, []
        for candidate in candidates:
            risk = self.assess(state, candidate[1], candidate[2], points, tracks, candidate[3])
            if not risk[0]:
                chosen = candidate
                break
            rejected.append((risk, candidate))
        if chosen is None:
            # Nothing is safe: execute the candidate with the largest predicted
            # clearance (then latest violation), preferring STOP on ties.
            _, chosen = max(rejected, key=lambda item: (item[0][3], item[0][2], item[1][0] == "STOP"))
        action_type, candidate_v, candidate_w, reverse = chosen
        command_v, command_w = self._limit(state, candidate_v, candidate_w, policy_dt, reverse)
        mode = "STOP" if action_type == "STOP" else "CLAMP"
        return ShieldResult(command_v, command_w, mode, True, action_type, source, ttc,
                            predicted_minimum_clearance=clearance)
