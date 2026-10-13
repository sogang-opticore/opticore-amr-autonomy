"""ROS-free core of the learned local planner (ARCHITECTURE §2, §8, §10).

Observation, action scaling and the safety shield are the AMR-Lite
functions used in training; this module only adds input bookkeeping, the
status vocabulary A* understands, and health reporting.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field, fields
from typing import Any, Callable

import numpy as np

from amr_lite.deploy.runtime import BundleError, PolicyRuntime
from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.action import scale_action
from amr_lite.learning.observation import build_observation
from amr_lite.planning.path_utils import nearest_segment, path_state
from amr_lite.planning.scan_shield import ScanObstacleTracker, ScanSafetyShield, transform_points
from amr_lite.sensors.lidar import ContractScanSpec, resample_scan, scan_points

from .health import HealthMonitor

PROFILE_KEYS = ("v_max", "w_max", "accel_max", "decel_max", "angular_accel_max", "robot_radius")
STOPPING_STATES = {"WAITING_ODOM", "STOPPED", "REACHED"}


@dataclass
class PlannerConfig:
    bundle_path: str = ""
    profile_check: str = "strict"          # strict | warn
    control_rate: float = 20.0
    v_max: float = 1.5
    w_max: float = 1.5
    accel_max: float = 1.0
    decel_max: float = 2.0
    angular_accel_max: float = 1.5
    robot_radius: float = 0.36
    lidar_mount_x: float = 0.25
    lidar_mount_y: float = 0.0
    lidar_mount_yaw: float = 0.0
    goal_tolerance: float = 0.25
    goal_reached_epsilon: float = 0.03
    goal_reached_stopped_speed: float = 0.03
    reached_new_path_rearm_dist: float = 0.75
    path_lost_offset: float = 1.8
    path_search_back: int = 2
    path_search_ahead: int = 40
    odom_timeout: float = 0.5
    scan_timeout: float = 0.5
    deadline_ms: float = 25.0
    deadline_miss_limit: int = 3
    shield_enabled: bool = True
    emergency_after_sec: float = 1.0
    emergency_repeat_sec: float = 2.0
    path_lost_repeat_sec: float = 2.0
    fault_on_saturation: bool = False
    fault_injection: str = "none"          # none | nan_action | slow_inference | stale_scan

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PlannerConfig":
        config = cls()
        for key in {item.name for item in fields(cls)} & set(values):
            setattr(config, key, coerce(getattr(config, key), values[key]))
        return config


@dataclass
class Odometry:
    stamp: float
    x: float
    y: float
    theta: float
    v: float
    w: float


@dataclass
class LaserScanData:
    stamp: float
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float
    pose: tuple[float, float, float] | None = None   # odometry pose when the scan arrived


@dataclass
class PlannerOutput:
    v: float
    w: float
    status: str
    events: list[str] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


class PathTracker:
    """Projection with a progress window so folded paths do not jump backwards."""

    def __init__(self, back: int = 2, ahead: int = 40):
        self.back, self.ahead = back, ahead
        self.points: list[tuple[float, float]] = []
        self.index: int | None = None

    def set_path(self, points: list[tuple[float, float]]) -> None:
        self.points, self.index = list(points), None

    def offset(self, robot: RobotState) -> float:
        if len(self.points) < 2:
            return math.dist((robot.x, robot.y), self.points[0]) if self.points else math.inf
        first, last = self._window()
        _, qx, qy, _ = nearest_segment(self.points, robot.x, robot.y, first, last)
        return math.hypot(robot.x - qx, robot.y - qy)

    def _window(self) -> tuple[int, int]:
        if self.index is None:
            return 0, len(self.points) - 2
        return max(0, self.index - self.back), min(len(self.points) - 2, self.index + self.ahead)

    def state(self, robot: RobotState, lookahead: float):
        window = self._window() if len(self.points) >= 2 else None
        result = path_state(self.points, robot, lookahead, segment_range=window)
        self.index = result.segment_index
        return result


class LearnedPlannerCore:
    def __init__(self, config: PlannerConfig, runtime: PolicyRuntime | None = None,
                 load_error: BundleError | None = None, clock: Callable[[], float] = time.perf_counter):
        self.config, self.runtime, self.clock = config, runtime, clock
        self.health = HealthMonitor(config.deadline_miss_limit)
        self.observation: dict[str, Any] = {}
        if load_error is not None:
            self.health.set_permanent(load_error.code, str(load_error))
        self.shield_config: dict[str, Any] = {}
        if runtime is not None:
            self._configure_from_bundle(runtime)
        self.shield = ScanSafetyShield(self.shield_config) if runtime is not None else None
        self.tracker = ScanObstacleTracker((runtime.manifest.get("safety") or {}).get("scan_tracker")
                                           if runtime is not None else None)
        self.tracker_stamp: float | None = None
        self.odometry: Odometry | None = None
        self.scan: LaserScanData | None = None
        self._contract: tuple[float, np.ndarray] | None = None
        self.frames: deque[np.ndarray] = deque(maxlen=4)
        self.path = PathTracker(config.path_search_back, config.path_search_ahead)
        self.has_path = False
        self.reached = False
        self.state = "WAITING_ODOM"
        self.stop_streak = 0.0
        self.last_emergency = -math.inf
        self.last_path_lost = -math.inf

    @classmethod
    def from_bundle(cls, config: PlannerConfig, **kwargs: Any) -> "LearnedPlannerCore":
        try:
            runtime = PolicyRuntime.load(config.bundle_path)
        except BundleError as error:
            return cls(config, None, error, **kwargs)
        return cls(config, runtime, **kwargs)

    # ------------------------------------------------------------ bundle
    def _configure_from_bundle(self, runtime: PolicyRuntime) -> None:
        profile = runtime.robot_profile["values"]
        self.observation = runtime.manifest["observation_contract"]
        mismatches = {key: (profile.get(key), getattr(self.config, key)) for key in PROFILE_KEYS
                      if key in profile and abs(float(profile[key]) - float(getattr(self.config, key))) > 1e-6}
        lidar = profile.get("lidar_model") or {}
        if lidar.get("type") != "raw":
            mismatches["lidar_model"] = (lidar.get("type", "contract"), "raw")
        else:
            for key, mine in (("mount_x", self.config.lidar_mount_x), ("mount_y", self.config.lidar_mount_y),
                              ("mount_yaw", self.config.lidar_mount_yaw)):
                if abs(float(lidar.get(key, 0.0)) - mine) > 1e-6:
                    mismatches[f"lidar_{key}"] = (lidar.get(key), mine)
        self.profile_mismatches = mismatches
        if mismatches and self.config.profile_check == "strict":
            self.health.set_permanent("PROFILE_MISMATCH", json_safe(mismatches))
        safety = runtime.manifest.get("safety") or {}
        # Command limits always come from this robot's parameters; the shield
        # settings come from the bundle so they match training and evaluation.
        self.shield_config = {
            "v_max": self.config.v_max, "w_max": self.config.w_max, "accel_max": self.config.accel_max,
            "decel_max": self.config.decel_max, "angular_accel_max": self.config.angular_accel_max,
            "robot_radius": self.config.robot_radius, "policy_dt": 1.0 / self.config.control_rate,
            "scan_shield": safety.get("scan_shield") or {},
        }

    # ------------------------------------------------------------- inputs
    def on_odometry(self, odometry: Odometry) -> None:
        self.odometry = odometry

    def on_scan(self, scan: LaserScanData) -> None:
        if scan.pose is None and self.odometry is not None:
            scan.pose = (self.odometry.x, self.odometry.y, self.odometry.theta)
        self.scan = scan

    def on_path(self, points: list[tuple[float, float]] | None) -> None:
        """Path in the local (odometry) frame; an empty list means planning failed."""
        if points is None:
            return
        if not points:
            self.has_path, self.reached = False, False
            self.path.set_path([])
            return
        if self.reached and self.odometry is not None:
            end = points[-1]
            if math.dist(end, (self.odometry.x, self.odometry.y)) <= self.config.reached_new_path_rearm_dist:
                return  # periodic re-plan of the goal we already reached
        self.path.set_path(points)
        self.has_path, self.reached = True, False

    # --------------------------------------------------------------- step
    def _stop(self, status: str, events: list[str] | None = None, **diagnostics: Any) -> PlannerOutput:
        if status in STOPPING_STATES:
            self.frames.clear()
            self.stop_streak = 0.0
        self.state = status
        return PlannerOutput(0.0, 0.0, status, events or [], self.health.report(), diagnostics)

    def _contract_scan(self) -> np.ndarray:
        scan = self.scan
        assert scan is not None
        if self._contract is None or self._contract[0] != scan.stamp:
            spec = ContractScanSpec(int(self.observation["lidar_rays"]),
                                    math.radians(float(self.observation["lidar_fov_deg"])),
                                    float(self.observation["lidar_max_range"]))
            self._contract = (scan.stamp, resample_scan(
                scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
                mount_x=self.config.lidar_mount_x, mount_y=self.config.lidar_mount_y,
                mount_yaw=self.config.lidar_mount_yaw, contract=spec))
        return self._contract[1]

    def _robot_points(self, robot: RobotState) -> np.ndarray:
        scan = self.scan
        assert scan is not None
        points = scan_points(scan.ranges, scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max,
                             mount_x=self.config.lidar_mount_x, mount_y=self.config.lidar_mount_y,
                             mount_yaw=self.config.lidar_mount_yaw)
        if self.tracker_stamp != scan.stamp and scan.pose is not None:
            self.tracker.update(points, scan.pose, scan.stamp)
            self.tracker_stamp = scan.stamp
        return points if scan.pose is None else transform_points(points, scan.pose, (robot.x, robot.y, robot.theta))

    def build_observation(self, robot: RobotState) -> tuple[np.ndarray, Any]:
        """Same construction as ``WarehouseEnv._observation`` (ARCHITECTURE §4)."""
        contract = self._contract_scan()
        if not self.frames:
            self.frames.extend(contract.copy() for _ in range(4))
        else:
            self.frames.append(contract)
        scans = list(self.frames)
        if self.observation.get("history_mode", "stack") == "newest":
            scans = [scans[-1]] * len(scans)
        state = self.path.state(robot, float(self.observation["lookahead_distance"]))
        observation = build_observation(scans, state, robot, float(self.observation["lidar_max_range"]),
                                        float(self.observation["goal_distance_scale"]),
                                        float(self.runtime.action_limits[0]), float(self.runtime.action_limits[1]),
                                        float(self.observation["max_cross_track_error"]))
        return observation, state

    def step(self, now: float) -> PlannerOutput:
        config = self.config
        period = 1.0 / config.control_rate
        if self.odometry is None:
            return self._stop("WAITING_ODOM")
        if now - self.odometry.stamp > config.odom_timeout:
            self.health.set_transient("ODOM_STALE", f"odometry {now - self.odometry.stamp:.2f}s old")
            return self._stop("STOPPED")
        self.health.clear("ODOM_STALE")
        if not self.has_path:
            return self._stop("STOPPED")
        if self.reached:
            return self._stop("REACHED")
        odometry = self.odometry
        robot = RobotState(odometry.x, odometry.y, odometry.theta, odometry.v, odometry.w)
        goal = self.path.points[-1]
        distance = math.dist((robot.x, robot.y), goal)
        if distance <= config.goal_tolerance or (
                distance <= config.goal_tolerance + config.goal_reached_epsilon
                and abs(robot.v) <= config.goal_reached_stopped_speed):
            self.reached = True
            return self._stop("REACHED", ["GOAL_REACHED"], goal_distance=distance)
        offset = self.path.offset(robot)
        if offset > config.path_lost_offset:
            events = []
            if now - self.last_path_lost >= config.path_lost_repeat_sec:
                events, self.last_path_lost = ["PATH_LOST"], now
            return self._stop("PATH_LOST", events, path_offset=offset)
        stale = self.scan is None or now - self.scan.stamp > config.scan_timeout or config.fault_injection == "stale_scan"
        if stale:
            self.health.set_transient("SCAN_STALE", "no recent LaserScan")
            return self._stop(self.state if self.state not in STOPPING_STATES else "NORMAL")
        self.health.clear("SCAN_STALE")
        if self.runtime is None or self.shield is None:
            return self._stop("STOPPED", reason="no usable policy bundle")
        observation, path = self.build_observation(robot)
        if not np.isfinite(observation).all():
            self.health.set_transient("NON_FINITE_OBSERVATION", "observation contains NaN/inf")
            return self._stop("NORMAL")
        self.health.clear("NON_FINITE_OBSERVATION")
        started = self.clock()
        action = self.runtime.predict(observation)
        inference_ms = 1000.0 * (self.clock() - started)
        if config.fault_injection == "nan_action":
            action = np.array([math.nan, 0.0], dtype=np.float32)
        if config.fault_injection == "slow_inference":
            inference_ms += config.deadline_ms + 1.0
        self.health.record_inference(inference_ms, inference_ms > config.deadline_ms)
        if not np.isfinite(action).all():
            self.health.set_transient("NON_FINITE_ACTION", "policy produced NaN/inf")
            return self._stop("NORMAL", inference_ms=inference_ms)
        self.health.clear("NON_FINITE_ACTION")
        # Scale with the limits the policy was trained with; the shield then
        # limits the command to this robot's parameters.
        target_v, target_w = scale_action(action, *self.runtime.action_limits)
        points = self._robot_points(robot)
        settings = self.shield.settings
        tracks = (self.tracker.relative_tracks((robot.x, robot.y, robot.theta), now,
                                               float(settings["track_min_speed"]))
                  if bool(settings["use_tracks"]) else [])
        result = self.shield.apply(robot, target_v, target_w, points, tracks, config.shield_enabled)
        self.stop_streak = self.stop_streak + period if result.mode == "STOP" else 0.0
        status, events = "NORMAL", []
        if self.stop_streak >= config.emergency_after_sec:
            status = "EMERGENCY"
            if now - self.last_emergency >= config.emergency_repeat_sec:
                events, self.last_emergency = ["EMERGENCY"], now
            if config.fault_on_saturation:
                self.health.set_transient("SHIELD_SATURATION", f"STOP for {self.stop_streak:.1f}s")
        else:
            self.health.clear("SHIELD_SATURATION")
        self.state = status
        return PlannerOutput(result.v, result.w, status, events, self.health.report(), {
            "action": [float(action[0]), float(action[1])], "target": [target_v, target_w],
            "shield_mode": result.mode, "shield_action": result.action_type, "risk": result.risk_source,
            "inference_ms": inference_ms, "goal_distance": distance, "cross_track": path.cross_track_error,
        })


def coerce(default: Any, value: Any) -> Any:
    """Convert a parameter value to the type of its default (``"false"`` -> False)."""
    if isinstance(default, bool):
        return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, str):
        return str(value)
    return type(default)(value)


def json_safe(value: Any) -> str:
    import json

    return json.dumps(value, default=str, sort_keys=True)
