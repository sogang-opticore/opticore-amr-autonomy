"""Development warehouse for the stamped observation and direct-action contract."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np

from amr_lite.config import ROOT, load_config, resolve_env_config
from amr_lite.envs.dynamics import RobotState, integrate, limit_velocity
from amr_lite.envs.grid_map import named_map
from amr_lite.experiments.manifest import sha256_file
from amr_lite.learning.observation import OBSERVATION_SIZE
from amr_lite.learning.stamped_io import StampedPolicyIO
from amr_lite.planning.astar import astar
from amr_lite.planning.path_utils import remaining_path_distance
from amr_lite.planning.scan_shield import ScanObstacleTracker, ScanSafetyShield, transform_points
from amr_lite.sensors.lidar import RawLidarSpec, scan_points, simulate_raw_scan

MANIFEST = ROOT.parents[1] / "experiments/m2/missions.json"
MAP_ASSET = ROOT / "assets/maps/opticore_warehouse/warehouse_map.pgm"
_NAMESPACE, _FRAME = "/amr1", "amr1/lidar_link"
_TICK, _PHYSICS_DT = 0.05, 0.01
_STOP = (0.0, 0.0)


@dataclass(frozen=True)
class DynamicsProfile:
    """Actuator and acquisition settings beneath the shared command boundary."""

    scan_period: float = 0.10
    drive_delay_ticks: int = 0
    drive_accel: float = 2.0
    drive_decel: float = 2.0
    drive_angular_accel: float = 1.5
    scan_noise_std: float = 0.03

    def __post_init__(self):
        rates = (self.drive_accel, self.drive_decel, self.drive_angular_accel)
        if not all(math.isfinite(v) and v > 0 for v in rates):
            raise ValueError("drive rates must be finite and positive")
        if not 0.025 <= self.scan_period < 0.5:
            raise ValueError("scan_period must lie in [0.025, 0.5) seconds")
        if type(self.drive_delay_ticks) is not int or self.drive_delay_ticks < 0:
            raise ValueError("drive_delay_ticks must be a nonnegative integer")
        if not math.isfinite(self.scan_noise_std) or self.scan_noise_std < 0:
            raise ValueError("scan noise must be finite and nonnegative")


def environment_config():
    config = load_config("m2/ppo_start")
    return resolve_env_config(config["robot_profile"], config["environment_overrides"])


@lru_cache(maxsize=1)
def _manifest():
    spec = json.loads(MANIFEST.read_text())
    if spec["map_sha256"] != sha256_file(MAP_ASSET) or (spec["version"], spec["frame"]) != (1, "map"):
        raise ValueError("warehouse map or mission manifest does not match the v1 conditions")
    return spec


def development_missions():
    return tuple(case["id"] for case in _manifest()["missions"] if case["split"] == "train")


@lru_cache(maxsize=8)
def _route(mission_id):
    spec = _manifest()
    case = next(c for c in spec["missions"] if c["id"] == mission_id and c["split"] == "train")
    grid = named_map("opticore_warehouse")
    start, goal = (grid.to_lite(*spec["stations"][case[endpoint]]) for endpoint in ("from", "to"))
    config = environment_config()
    path = astar(grid, start, goal, config["grid_resolution"],
                 config["robot_radius"] + config["obstacle_inflation"], config["astar_clearance"])
    if len(path) < 2:
        raise RuntimeError(f"no Lite A* route for {mission_id}")
    return case, grid, start, goal, tuple(path)


class _ScanGeometry:
    """Ray cast against the same occupancy grid and the mission's timed rectangle."""

    def __init__(self, grid, actor):
        self.grid = grid
        self.actor = actor
        self.box = None
        if actor:
            self.times = np.array([t for t, _, _ in actor["knots"]])
            self.positions = np.array([grid.to_lite(x, y) for _, x, y in actor["knots"]])

    def set_time(self, elapsed):
        if self.actor is None:
            return
        removal = self.actor["remove_at"] if "remove_at" in self.actor else math.inf
        if elapsed >= removal:
            self.box = None
            return
        x = np.interp(elapsed, self.times, self.positions[:, 0])
        y = np.interp(elapsed, self.times, self.positions[:, 1])
        width, depth, _ = self.actor["size"]
        self.box = (x - width / 2, y - depth / 2, x + width / 2, y + depth / 2)

    def clearance(self, x, y, radius):
        distance = self.grid.clearance(x, y, radius)
        if self.box is None:
            return distance
        x_min, y_min, x_max, y_max = self.box
        return min(distance, math.hypot(max(x_min - x, x - x_max, 0),
                                        max(y_min - y, y - y_max, 0)) - radius)

    def ray_distances(self, ox, oy, dx, dy, max_range):
        static = self.grid.ray_distances(ox, oy, dx, dy, max_range)
        if self.box is None:
            return static
        x_min, y_min, x_max, y_max = self.box
        with np.errstate(divide="ignore", invalid="ignore"):
            near_x, far_x = self._slab(ox, dx, x_min, x_max)
            near_y, far_y = self._slab(oy, dy, y_min, y_max)
        enter = np.maximum(near_x, near_y)
        leave = np.minimum(far_x, far_y)
        hit = np.where(enter >= 0.0, enter, leave)
        return np.minimum(static, np.where((leave >= np.maximum(enter, 0.0)), hit, max_range))

    @staticmethod
    def _slab(origin, direction, low, high):
        a, b = (low - origin) / direction, (high - origin) / direction
        parallel = direction == 0.0
        if low <= origin <= high:
            return np.where(parallel, -np.inf, np.minimum(a, b)), np.where(parallel, np.inf, np.maximum(a, b))
        return np.where(parallel, np.inf, np.minimum(a, b)), np.where(parallel, -np.inf, np.maximum(a, b))


class M3WarehouseEnv(gym.Env):
    """One development mission with actual distinguishable LiDAR acquisitions."""

    metadata = {"render_modes": []}

    def __init__(self, mission_id="train-static", dynamics=None):
        if mission_id not in development_missions():
            raise ValueError(f"expected a train mission, received {mission_id!r}")
        super().__init__()
        self.mission_id = mission_id
        self.dynamics = dynamics or DynamicsProfile()
        self.config = environment_config()
        self.timeout = float(_manifest()["timeout_sec"])
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.io = StampedPolicyIO()
        self.raw_spec = RawLidarSpec.from_env_config(self.config)
        self.shield = ScanSafetyShield(self.config)
        self.tracker = ScanObstacleTracker(self.config.get("scan_tracker"))
        case, grid, self.start, self.goal, self.path = _route(mission_id)
        self.geometry = _ScanGeometry(grid, case["actor"])
        self.radius = float(self.config["robot_radius"])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.io.reset()
        self.tracker.reset()
        self.robot = RobotState(*self.start, float(_manifest()["start_yaw_rad"]))
        self.warmup = self.dynamics.scan_period * 3
        self.geometry.set_time(0.0)
        if self.geometry.clearance(self.robot.x, self.robot.y, self.radius) <= 0:
            raise RuntimeError(f"occupied initial pose for {self.mission_id}")
        for index in range(4):
            self._scan(index * self.dynamics.scan_period)
        self.clock = self.warmup
        self.next_scan = 4 * self.dynamics.scan_period
        self.elapsed = 0.0
        self.steps = 0
        self._delay = deque([_STOP] * self.dynamics.drive_delay_ticks)
        self.reward_totals = {name: 0.0 for name in ("progress", "time", "goal", "collision")}
        self.interventions = 0
        self.min_clearance = self.geometry.clearance(self.robot.x, self.robot.y, self.radius)
        self.done = False
        sample = self.io.observe(self.clock, self.clock, self.robot, self.path)
        if sample.tensor is None or sample.path is None:
            raise RuntimeError(f"unavailable reset observation: {sample.fault}")
        self.last_observation = sample.tensor
        self.last_stamps = sample.stamps
        self.remaining = remaining_path_distance(self.path, sample.path)
        return sample.tensor.copy(), self._info()

    def _scan(self, stamp):
        spec = self.raw_spec
        # Returns beyond the 8 m base-frame contract and the 4 m shield horizon
        # cannot contribute. Keep an extra margin for mount offset and LiDAR noise.
        cast_range = min(spec.range_max, max(8.0, self.shield.settings["max_point_range"])
                         + math.hypot(spec.mount_x, spec.mount_y) + 6 * self.dynamics.scan_noise_std)
        ranges = simulate_raw_scan(self.robot, self.geometry, [], spec, cast_range=cast_range)
        valid = np.isfinite(ranges)
        ranges[valid] += self.np_random.normal(0.0, self.dynamics.scan_noise_std, size=int(valid.sum()))
        # Gazebo reports noisy out-of-range values as no return at the far end.
        ranges[ranges > spec.range_max] = np.inf
        self.io.ingest_scan(_NAMESPACE, _FRAME, stamp, ranges, spec.angle_min, spec.angle_increment,
                            spec.range_min, spec.range_max)
        self.scan_pose = self._pose()
        self.scan_points = scan_points(ranges, spec.angle_min, spec.angle_increment, spec.range_min,
                                       spec.range_max, mount_x=spec.mount_x, mount_y=spec.mount_y,
                                       mount_yaw=spec.mount_yaw,
                                       max_distance=float(self.shield.settings["max_point_range"]) + 0.4)
        self.tracker.update(self.scan_points, self.scan_pose, stamp)

    def _pose(self):
        return self.robot.x, self.robot.y, self.robot.theta

    def _advance(self, command, end):
        profile = self.dynamics
        while self.clock < end - 1e-9:
            next_clock = min(self.clock + _PHYSICS_DT, end, self.next_scan)
            dt = next_clock - self.clock
            v, w = limit_velocity(self.robot.v, self.robot.w, *command, dt, 1.5, 1.5,
                                  profile.drive_accel, profile.drive_decel, profile.drive_angular_accel)
            self.robot = integrate(self.robot, v, w, dt)
            self.clock = next_clock
            self.elapsed = self.clock - self.warmup
            self.geometry.set_time(self.elapsed)
            clearance = self.geometry.clearance(self.robot.x, self.robot.y, self.radius)
            self.min_clearance = min(self.min_clearance, clearance)
            if self.clock >= self.next_scan - 1e-9:
                self._scan(self.next_scan)
                self.next_scan += profile.scan_period
            if clearance <= 0.0:
                return True
        return False

    def _info(self, **extras):
        return {"mission_id": self.mission_id, "elapsed": self.elapsed, "scan_stamps": self.last_stamps,
                "goal_error": math.dist((self.robot.x, self.robot.y), self.goal),
                "pose_lite": self._pose(), "velocity": (self.robot.v, self.robot.w),
                "min_clearance": self.min_clearance, "shield_interventions": self.interventions,
                **extras}

    def _invalid(self, fault):
        # A corrupt transition has no successor in the training MDP. The collector
        # must stop instead of letting SB3's automatic reset accept this row.
        self._delay.clear()
        self.done = True
        return self.last_observation.copy(), 0.0, True, False, self._info(
            disposition="invalid", fault=fault, target=None, command=_STOP, applied_command=_STOP,
            reward_components={}, reward_totals=self.reward_totals.copy())

    def step(self, action):
        if self.done:
            raise RuntimeError("reset the mission before taking another step")
        output = self.io.actuate(action, self.robot)
        if output.fault:
            return self._invalid(output.fault)
        points = transform_points(self.scan_points, self.scan_pose, self._pose())
        tracks = self.tracker.relative_tracks(self._pose(), self.clock, self.shield.settings["track_min_speed"])
        shield = self.shield.apply(self.robot, *output.target, points, tracks,
                                   enabled=bool(self.config["shield_enabled"]))
        command = (shield.v, shield.w) if shield.mode != "PASS" else output.command
        self.interventions += int(shield.mode != "PASS")
        self._delay.append(command)
        applied = self._delay.popleft()
        self.steps += 1
        collision = self._advance(applied, self.warmup + min(self.steps * _TICK, self.timeout))
        arrived = math.dist((self.robot.x, self.robot.y), self.goal) <= self.config["goal_tolerance"]
        sample = self.io.observe(self.clock, self.clock, self.robot, self.path)
        if sample.tensor is None or sample.path is None:
            return self._invalid(sample.fault)
        self.last_observation, self.last_stamps = sample.tensor, sample.stamps
        remaining = remaining_path_distance(self.path, sample.path)
        self.done = collision or arrived or self.steps * _TICK >= self.timeout
        disposition = "collision" if collision else "success" if arrived else "timeout" if self.done else "running"
        reward = {"progress": self.config["reward_progress"] * (self.remaining - remaining),
                  "time": self.config["reward_time"],
                  "goal": self.config["reward_goal"] if disposition == "success" else 0.0,
                  "collision": self.config["reward_collision"] if collision else 0.0}
        for name, value in reward.items():
            self.reward_totals[name] += value
        self.remaining = remaining
        info = self._info(disposition=disposition, target=output.target, command=command,
                          applied_command=applied, shield_mode=shield.mode, reward_components=reward,
                          reward_totals=self.reward_totals.copy())
        return sample.tensor.copy(), sum(reward.values()), collision or arrived, self.done and not (collision or arrived), info
