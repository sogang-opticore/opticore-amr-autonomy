from __future__ import annotations

import copy
import math
import random
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any

import numpy as np

from amr_lite.config import load_config
from amr_lite.learning.action import scale_action
from amr_lite.learning.observation import build_observation
from amr_lite.planning.astar import astar, clearance_key
from amr_lite.planning.path_utils import path_state, remaining_path_distance
from amr_lite.planning.safety_shield import SafetyShield
from amr_lite.sensors.lidar import ContractScanSpec, RawLidarSpec, resample_scan, simulate_raw_scan
from amr_lite.sensors.randomization import (
    SensorDynamicsModel,
    episode_sensor_model,
    noisy_contract_scan,
    noisy_raw_ranges,
    perturb_path,
)

from .dynamics import RobotState, integrate
from .raycast import lidar_scan
from .scenarios import DynamicObstacle, Scenario, make_scenario


# A* results keyed by scenario instance. Bounded because procedural scenarios
# create a new instance per training episode.
_PATH_CACHE: OrderedDict[tuple, tuple[tuple[float, float], ...]] = OrderedDict()
_PATH_CACHE_LIMIT = 4096
# The stop command: (a_v + 1) / 2 * v_max == 0. Pending delayed actions start here.
_STOP_ACTION = np.asarray([-1.0, 0.0], dtype=np.float32)


class Box:
    """Small gymnasium.spaces.Box-compatible fallback."""

    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32):
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    def contains(self, value: Any) -> bool:
        array = np.asarray(value)
        return array.shape == self.shape and np.isfinite(array).all() and (array >= self.low).all() and (array <= self.high).all()


class WarehouseEnv:
    """Deterministic, ROS-free 2D differential-drive navigation environment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: dict[str, Any] | None = None, scenario_id: str | None = None,
                 render_mode: str | None = None):
        self.config = copy.deepcopy(config or load_config("env"))
        self.scenario_id = scenario_id or self.config["scenario"]
        self.render_mode = render_mode
        self.action_space = Box(-1.0, 1.0, (2,))
        self.observation_space = Box(-1.0, 1.0, (296,))
        self.shield = SafetyShield(self.config)
        self.np_random = np.random.default_rng(self.config["seed"])
        self._python_random = random.Random(self.config["seed"])
        self.scan_history: deque[np.ndarray] = deque(maxlen=4)
        self.trajectory: list[tuple[float, float]] = []
        self.scenario: Scenario
        self.robot: RobotState
        self.global_path: list[tuple[float, float]]
        self.elapsed = 0.0
        self.episode_timeout = float(self.config["episode_timeout"])
        self.step_count = 0
        self.collision = False
        self.success = False
        self.shield_counts = {"PASS": 0, "CLAMP": 0, "STOP": 0}
        self.shield_action_counts = {
            "PASS": 0, "SLOW": 0, "EVADE": 0, "REVERSE": 0, "STOP": 0
        }
        self.unsafe_without_shield = 0
        self.shield_false_positive = 0
        self.shield_false_negative = 0
        self.shield_actual_false_negative = 0
        self.shield_ineffective_intervention = 0
        self.shield_risk_counts: dict[str, int] = {}
        self.tracking_speed_errors: list[float] = []
        self.minimum_predicted_ttc = math.inf
        self.shield_trace: list[dict[str, Any]] = []
        self.shield_streak = 0
        self.max_shield_streak = 0
        self.cumulative_reward = 0.0
        self.path_length = 0.0
        self.idle_time = 0.0
        self.angular_oscillation = 0
        self.total_abs_rotation = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_w_sign = 0
        self.min_clearance_seen = math.inf
        self.cross_track_samples: list[float] = []
        self.sensor_model = SensorDynamicsModel()
        self._noise_rng = np.random.default_rng(0)
        self._raw_spec = RawLidarSpec.from_env_config(self.config)
        self._contract_spec = ContractScanSpec.from_env_config(self.config)
        # Raw returns only reach the contract within max_range + mount offset.
        self._raw_cast_range = (self._contract_spec.max_range
                                + math.hypot(self._raw_spec.mount_x, self._raw_spec.mount_y) + 0.25)
        self.last_raw_scan: np.ndarray | None = None
        self.last_scan_pose: tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._held_scan: np.ndarray | None = None
        self._next_scan_time = 0.0
        self._action_queue: deque[np.ndarray] | None = None
        self._observation_queue: deque[np.ndarray] | None = None
        self.reset(seed=self.config["seed"])

    @property
    def dynamic_obstacles(self) -> list[DynamicObstacle]:
        return self.scenario.dynamic_obstacles

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        seed = self.config["seed"] if seed is None else int(seed)
        self.np_random = np.random.default_rng(seed)
        self._python_random.seed(seed)
        requested = (options or {}).get("scenario_id", self.scenario_id)
        self.scenario = make_scenario(requested, seed, self.config.get("scenario_randomization"), self.config)
        self.shield.reset()
        self.sensor_model, self._noise_rng = episode_sensor_model(self.config, seed)
        self.robot = copy.deepcopy(self.scenario.start)
        if options and "pose_perturbation" in options:
            amount = float(options["pose_perturbation"])
            self.robot.x += float(self.np_random.uniform(-amount, amount))
            self.robot.y += float(self.np_random.uniform(-amount, amount))
            self.robot.theta += float(self.np_random.uniform(-amount, amount))
        inflation = self.config["robot_radius"] + self.config["obstacle_inflation"]
        planner_clearance = clearance_key(self.config.get("astar_clearance"))
        path_key = (
            self.scenario.instance_id,
            float(self.config["grid_resolution"]),
            float(inflation),
            tuple((rect.x_min, rect.y_min, rect.x_max, rect.y_max)
                  for rect in self.scenario.warehouse_map.obstacles),
        )
        if planner_clearance is not None:
            path_key += (planner_clearance,)
        reference = self.scenario.reference_path
        if (reference and path_key not in _PATH_CACHE
                and self.scenario.reference_planner == (float(self.config["grid_resolution"]), float(inflation),
                                                        planner_clearance)
                and (self.robot.x, self.robot.y) == (self.scenario.start.x, self.scenario.start.y)):
            _PATH_CACHE[path_key] = tuple(reference)
        if path_key in _PATH_CACHE:
            _PATH_CACHE.move_to_end(path_key)
        else:
            _PATH_CACHE[path_key] = tuple(astar(
                self.scenario.warehouse_map,
                (self.robot.x, self.robot.y),
                self.scenario.goal,
                self.config["grid_resolution"],
                inflation,
                self.config.get("astar_clearance"),
            ))
            if len(_PATH_CACHE) > _PATH_CACHE_LIMIT:
                _PATH_CACHE.popitem(last=False)
        self.global_path = list(_PATH_CACHE[path_key])
        if self.sensor_model.perturbs_path and self.global_path:
            self.global_path = perturb_path(self.global_path, self.sensor_model.path_resample_spacing,
                                            self.sensor_model.path_lateral_noise, self._noise_rng)
        self.elapsed = 0.0
        self.episode_timeout = float(self.config["episode_timeout"])
        per_meter = float(self.config.get("episode_timeout_per_meter", 0.0))
        if per_meter > 0.0 and len(self.global_path) > 1:
            shortest = sum(math.dist(a, b) for a, b in zip(self.global_path, self.global_path[1:]))
            self.episode_timeout = max(self.episode_timeout, float(
                self.config.get("episode_timeout_base", 6.0)) + per_meter * shortest)
        self.step_count = 0
        self.collision = False
        self.success = False
        self.shield_counts = {"PASS": 0, "CLAMP": 0, "STOP": 0}
        self.shield_action_counts = {
            "PASS": 0, "SLOW": 0, "EVADE": 0, "REVERSE": 0, "STOP": 0
        }
        self.unsafe_without_shield = 0
        self.shield_false_positive = 0
        self.shield_false_negative = 0
        self.shield_actual_false_negative = 0
        self.shield_ineffective_intervention = 0
        self.shield_risk_counts = {}
        self.tracking_speed_errors = []
        self.minimum_predicted_ttc = math.inf
        self.shield_trace = []
        self.shield_streak = 0
        self.max_shield_streak = 0
        self.cumulative_reward = 0.0
        self.path_length = 0.0
        self.idle_time = 0.0
        self.angular_oscillation = 0
        self.total_abs_rotation = 0.0
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.previous_w_sign = 0
        self.min_clearance_seen = math.inf
        self.cross_track_samples = []
        self.trajectory = [(self.robot.x, self.robot.y)]
        initial_path_state = path_state(self.global_path, self.robot, self.config["lookahead_distance"])
        self.previous_path_remaining = remaining_path_distance(self.global_path, initial_path_state)
        model = self.sensor_model
        self._action_queue = (deque(_STOP_ACTION.copy() for _ in range(model.action_delay_steps))
                              if model.action_delay_steps > 0 else None)
        self._next_scan_time = model.scan_phase if model.scan_phase > 1e-9 else model.scan_period
        scan = self._sensor_scan(force=True)
        self.scan_history = deque((scan.copy() for _ in range(4)), maxlen=4)
        observation = self._observation()
        self._observation_queue = (deque(observation.copy() for _ in range(model.observation_delay_steps))
                                   if model.observation_delay_steps > 0 else None)
        info = self._info(seed=seed, reward_components={})
        info["sensor_model"] = model.as_dict()
        return observation, info

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_array = np.asarray(action, dtype=np.float32)
        invalid = action_array.shape != (2,) or not np.isfinite(action_array).all()
        if invalid:
            action_array = np.zeros(2, dtype=np.float32)
        executed_action = action_array
        if self._action_queue is not None:
            self._action_queue.append(action_array.copy())
            executed_action = self._action_queue.popleft()
        target_v, target_w = scale_action(executed_action, self.config["v_max"], self.config["w_max"])
        before_state = RobotState(
            self.robot.x, self.robot.y, self.robot.theta, self.robot.v, self.robot.w
        )
        result = self.shield.apply(self.robot, target_v, target_w, self.scenario.warehouse_map,
                                   self.dynamic_obstacles, bool(self.config["shield_enabled"]), self.elapsed)
        self.shield_counts[result.mode] += 1
        self.shield_action_counts[result.action_type] += 1
        self.unsafe_without_shield += int(result.unsafe_without_shield)
        self.shield_false_positive += int(result.false_positive)
        self.shield_false_negative += int(result.false_negative)
        self.shield_risk_counts[result.risk_source] = (
            self.shield_risk_counts.get(result.risk_source, 0) + 1
        )
        if math.isfinite(result.tracking_speed_error):
            self.tracking_speed_errors.append(result.tracking_speed_error)
        if math.isfinite(result.predicted_ttc):
            self.minimum_predicted_ttc = min(
                self.minimum_predicted_ttc, result.predicted_ttc
            )
        self.shield_streak = self.shield_streak + 1 if result.mode != "PASS" else 0
        self.max_shield_streak = max(self.max_shield_streak, self.shield_streak)
        before = (self.robot.x, self.robot.y)
        sub_dt = self.config["physics_dt"]
        for _ in range(int(self.config["physics_substeps"])):
            for obstacle in self.dynamic_obstacles:
                obstacle.update(self.elapsed, sub_dt)
            command_v, command_w = self._actuate(result.v, result.w, sub_dt)
            next_state = integrate(self.robot, command_v, command_w, sub_dt)
            self.elapsed += sub_dt
            if self._collides(next_state):
                self.robot = next_state
                self.collision = True
                break
            self.robot = next_state
        if self.collision and bool(self.config["shield_enabled"]):
            if result.mode == "PASS":
                self.shield_actual_false_negative += 1
            else:
                self.shield_ineffective_intervention += 1
        if bool(self.config.get("record_shield_trace", False)):
            self.shield_trace.append({
                "time": self.elapsed,
                "before_pose": [before_state.x, before_state.y, before_state.theta],
                "before_velocity": [before_state.v, before_state.w],
                "target_velocity": [target_v, target_w],
                "executed_velocity": [result.v, result.w],
                "mode": result.mode,
                "action_type": result.action_type,
                "risk_source": result.risk_source,
                "predicted_ttc": result.predicted_ttc,
                "cpa_distance": result.cpa_distance,
                "predicted_minimum_clearance": result.predicted_minimum_clearance,
                "false_positive": result.false_positive,
                "false_negative": result.false_negative,
                "tracking_speed_error": result.tracking_speed_error,
                "after_pose": [self.robot.x, self.robot.y, self.robot.theta],
                "collision": self.collision,
            })
        moved = math.dist(before, (self.robot.x, self.robot.y))
        self.path_length += moved
        self.trajectory.append((self.robot.x, self.robot.y))
        self.step_count += 1
        self.total_abs_rotation += abs(self.robot.w) * self.config["policy_dt"]
        self.idle_time += self.config["policy_dt"] if abs(self.robot.v) < 0.03 else 0.0
        sign = 1 if self.robot.w > 0.15 else -1 if self.robot.w < -0.15 else 0
        if sign and self.previous_w_sign and sign != self.previous_w_sign:
            self.angular_oscillation += 1
        if sign:
            self.previous_w_sign = sign
        current_goal_distance = math.dist((self.robot.x, self.robot.y), self.scenario.goal)
        self.success = current_goal_distance <= self.config["goal_tolerance"] and not self.collision
        self.scan_history.append(self._sensor_scan())
        path = path_state(self.global_path, self.robot, self.config["lookahead_distance"])
        current_path_remaining = remaining_path_distance(self.global_path, path)
        self.cross_track_samples.append(abs(path.cross_track_error))
        clearance = self._minimum_clearance()
        self.min_clearance_seen = min(self.min_clearance_seen, clearance)
        reward_components = self._reward_components(self.previous_path_remaining - current_path_remaining,
                                                    clearance, action_array, path.cross_track_error,
                                                    path.heading_error)
        reward = float(sum(reward_components.values()))
        self.cumulative_reward += reward
        terminated = self.collision or self.success or invalid or not self.global_path
        truncated = self.elapsed >= self.episode_timeout and not terminated
        self.previous_action = action_array.copy()
        self.previous_path_remaining = current_path_remaining
        info = self._info(reward_components=reward_components, invalid_action=invalid,
                          shield_mode=result.mode, safety_cost=float(self.collision) + max(0.0, self.config["safety_distance"] - clearance))
        observation = self._observation()
        if self._observation_queue is not None:
            self._observation_queue.append(observation)
            observation = self._observation_queue.popleft()
        return observation, reward, terminated, truncated, info

    def _scan(self) -> np.ndarray:
        return lidar_scan(self.robot, self.scenario.warehouse_map, self.dynamic_obstacles,
                          self.config["lidar_rays"], math.radians(self.config["lidar_fov_deg"]),
                          self.config["lidar_max_range"])

    def _sensor_scan(self, force: bool = False) -> np.ndarray:
        """Contract scan seen by the policy, held between sensor updates."""
        if force or self._held_scan is None or self._scan_due():
            self._held_scan = self._fresh_scan()
        return self._held_scan

    def _scan_due(self) -> bool:
        period = self.sensor_model.scan_period
        if period <= self.config["policy_dt"] + 1e-9 and self.sensor_model.scan_phase <= 1e-9:
            return True
        if self.elapsed + 1e-9 < self._next_scan_time:
            return False
        while self._next_scan_time <= self.elapsed + 1e-9:
            self._next_scan_time += period
        return True

    def _fresh_scan(self) -> np.ndarray:
        model = self.sensor_model
        self.last_scan_pose = (self.robot.x, self.robot.y, self.robot.theta)
        bias = math.radians(model.angular_bias_deg)
        if model.lidar_type == "raw":
            raw = simulate_raw_scan(self.robot, self.scenario.warehouse_map, self.dynamic_obstacles,
                                    self._raw_spec, cast_range=self._raw_cast_range, angular_bias=bias)
            raw = noisy_raw_ranges(raw, model, self._noise_rng)
            self.last_raw_scan = raw
            spec = self._raw_spec
            return resample_scan(raw, spec.angle_min, spec.angle_increment, spec.range_min,
                                 spec.range_max, mount_x=spec.mount_x, mount_y=spec.mount_y,
                                 mount_yaw=spec.mount_yaw, contract=self._contract_spec)
        if model.lidar_type != "contract":
            raise ValueError(f"Unknown lidar_model.type: {model.lidar_type!r}")
        if bias:
            sensed = RobotState(self.robot.x, self.robot.y, self.robot.theta + bias)
            scan = lidar_scan(sensed, self.scenario.warehouse_map, self.dynamic_obstacles,
                              self._contract_spec.rays, self._contract_spec.fov,
                              self._contract_spec.max_range)
        else:
            scan = self._scan()
        if model.scan_noise:
            scan = noisy_contract_scan(scan, model, self._noise_rng, self._contract_spec.max_range)
        return scan

    def _actuate(self, command_v: float, command_w: float, dt: float) -> tuple[float, float]:
        """Velocity the wheels actually achieve for a commanded velocity."""
        model = self.sensor_model
        if model.actuation_identity:
            return command_v, command_w
        target_v, target_w = command_v * model.v_scale, command_w * model.w_scale
        if model.actuator_time_constant <= 0.0:
            return target_v, target_w
        blend = 1.0 - math.exp(-dt / model.actuator_time_constant)
        return (self.robot.v + (target_v - self.robot.v) * blend,
                self.robot.w + (target_w - self.robot.w) * blend)

    def _perceived_robot(self) -> RobotState:
        """Pose used for path features: true pose plus localization error."""
        model = self.sensor_model
        if model.localization_identity:
            return self.robot
        noise_xy, noise_theta = model.localization_noise_xy, model.localization_noise_theta
        dx = float(self._noise_rng.normal(0.0, noise_xy)) if noise_xy > 0.0 else 0.0
        dy = float(self._noise_rng.normal(0.0, noise_xy)) if noise_xy > 0.0 else 0.0
        dtheta = float(self._noise_rng.normal(0.0, noise_theta)) if noise_theta > 0.0 else 0.0
        return RobotState(self.robot.x + model.bias_x + dx, self.robot.y + model.bias_y + dy,
                          self.robot.theta + model.bias_theta + dtheta, self.robot.v, self.robot.w)

    def _observation(self) -> np.ndarray:
        robot = self._perceived_robot()
        path = path_state(self.global_path, robot, self.config["lookahead_distance"])
        return build_observation(list(self.scan_history), path, robot,
                                 self.config["lidar_max_range"], self.config["goal_distance_scale"],
                                 self.config["v_max"], self.config["w_max"],
                                 self.config["max_cross_track_error"])

    def _collision_radius(self) -> float:
        radius = self.config["robot_radius"]
        if self.sensor_model.collision_radius_margin:
            radius += self.sensor_model.collision_radius_margin
        return radius

    def _collides(self, state: RobotState) -> bool:
        radius = self._collision_radius()
        if self.scenario.warehouse_map.collides(state.x, state.y, radius):
            return True
        return any(math.hypot(state.x - obstacle.x, state.y - obstacle.y) <= radius + obstacle.radius
                   for obstacle in self.dynamic_obstacles)

    def _minimum_clearance(self) -> float:
        static = self.scenario.warehouse_map.clearance(self.robot.x, self.robot.y, self.config["robot_radius"])
        dynamic = [math.hypot(self.robot.x - item.x, self.robot.y - item.y) - self.config["robot_radius"] - item.radius
                   for item in self.dynamic_obstacles]
        return min([static, *dynamic])

    def _reward_components(self, path_progress: float, clearance: float, action: np.ndarray,
                           cross_track: float, heading_error: float) -> dict[str, float]:
        return {
            "progress": self.config["reward_progress"] * path_progress,
            "goal": self.config["reward_goal"] if self.success else 0.0,
            "collision": self.config["reward_collision"] if self.collision else 0.0,
            "near": self.config["reward_near"] * max(0.0, self.config["safety_distance"] - clearance),
            "smooth": self.config["reward_smooth"] * float(np.sum((action - self.previous_action) ** 2)),
            "path": self.config["reward_path"] * abs(cross_track),
            "idle": self.config["reward_idle"] if abs(self.robot.v) < 0.03 else 0.0,
            "heading": self.config["reward_heading"] * abs(heading_error),
            "turn": self.config["reward_turn"] * float(action[1] ** 2),
            "time": self.config["reward_time"],
        }

    def _failure_type(self, invalid_action: bool = False) -> str:
        if not self.global_path:
            return "NO_PATH"
        if invalid_action:
            return "INVALID_ACTION"
        if self.collision:
            dynamic = any(math.hypot(self.robot.x - item.x, self.robot.y - item.y) <= self._collision_radius() + item.radius
                          for item in self.dynamic_obstacles)
            return "COLLISION_DYNAMIC" if dynamic else "COLLISION_STATIC"
        if self.shield_counts["STOP"] > max(20, self.step_count // 2):
            return "SHIELD_SATURATION"
        if self.angular_oscillation > 12:
            return "OSCILLATION"
        if self.total_abs_rotation > 4.0 * math.pi:
            return "CIRCLING"
        if self.idle_time > self.episode_timeout * 0.6:
            return "STUCK"
        return "TIMEOUT" if self.elapsed >= self.episode_timeout else ""

    def _info(self, seed: int | None = None, reward_components: dict | None = None,
              invalid_action: bool = False, **extra: Any) -> dict[str, Any]:
        shortest = sum(math.dist(a, b) for a, b in zip(self.global_path, self.global_path[1:]))
        cross = self.cross_track_samples or [0.0]
        info = {
            "seed": seed,
            "scenario_id": self.scenario.scenario_id,
            "scenario_instance_id": self.scenario.instance_id,
            "robot_pose": (self.robot.x, self.robot.y, self.robot.theta),
            "goal_distance": math.dist((self.robot.x, self.robot.y), self.scenario.goal),
            "success": self.success,
            "collision": self.collision,
            "failure_type": self._failure_type(invalid_action),
            "minimum_clearance": self._minimum_clearance(),
            "minimum_clearance_episode": self.min_clearance_seen,
            "elapsed_time": self.elapsed,
            "path_length": self.path_length,
            "shortest_path_length": shortest,
            "path_length_ratio": self.path_length / shortest if shortest else math.inf,
            "mean_cross_track_error": float(np.mean(cross)),
            "max_cross_track_error": max(cross),
            "idle_time": self.idle_time,
            "angular_oscillation": self.angular_oscillation,
            "total_abs_rotation": self.total_abs_rotation,
            "path_remaining_distance": getattr(self, "previous_path_remaining", shortest),
            "shield_counts": dict(self.shield_counts),
            "shield_action_counts": dict(self.shield_action_counts),
            "unsafe_without_shield": self.unsafe_without_shield,
            "shield_sustained_time": self.max_shield_streak * self.config["policy_dt"],
            "shield_false_positive": self.shield_false_positive,
            "shield_false_negative": self.shield_false_negative,
            "shield_actual_false_negative": self.shield_actual_false_negative,
            "shield_ineffective_intervention": self.shield_ineffective_intervention,
            "shield_risk_counts": dict(self.shield_risk_counts),
            "tracking_speed_error_mean": (
                float(np.mean(self.tracking_speed_errors))
                if self.tracking_speed_errors else 0.0
            ),
            "tracking_speed_error_max": (
                max(self.tracking_speed_errors) if self.tracking_speed_errors else 0.0
            ),
            "minimum_predicted_ttc": self.minimum_predicted_ttc,
            "reward_components": reward_components or {},
        }
        info.update(extra)
        return info

    def render(self) -> np.ndarray | None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle as PatchRectangle

        fig, ax = plt.subplots(figsize=(9, 6), dpi=100)
        ax.set(xlim=(0, self.scenario.warehouse_map.width), ylim=(0, self.scenario.warehouse_map.height), aspect="equal")
        grid = getattr(self.scenario.warehouse_map, "occupied_cells", None)
        if grid is not None:
            ax.imshow(grid, origin="lower", cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest",
                      extent=(0, self.scenario.warehouse_map.width, 0, self.scenario.warehouse_map.height))
        for rect in self.scenario.warehouse_map.obstacles:
            ax.add_patch(PatchRectangle((rect.x_min, rect.y_min), rect.x_max - rect.x_min,
                                        rect.y_max - rect.y_min, color="#39424e"))
        if self.global_path:
            xs, ys = zip(*self.global_path)
            ax.plot(xs, ys, "--", color="#4aa3ff", label="A* path")
        if self.trajectory:
            xs, ys = zip(*self.trajectory)
            ax.plot(xs, ys, color="#f5a623", linewidth=2, label="trajectory")
        scan = self.scan_history[-1]
        angles = np.linspace(-math.radians(self.config["lidar_fov_deg"]) / 2.0,
                             math.radians(self.config["lidar_fov_deg"]) / 2.0,
                             self.config["lidar_rays"])
        for relative, distance in zip(angles[::6], scan[::6]):
            end = (self.robot.x + float(distance) * math.cos(self.robot.theta + relative),
                   self.robot.y + float(distance) * math.sin(self.robot.theta + relative))
            ax.plot((self.robot.x, end[0]), (self.robot.y, end[1]), color="#9ad5ca", alpha=0.35, linewidth=0.6)
        local = path_state(self.global_path, self.robot, self.config["lookahead_distance"]).local_goal
        ax.scatter(*local, marker="x", s=70, color="#9b5de5", label="local goal")
        ax.add_patch(Circle((self.robot.x, self.robot.y), self.config["robot_radius"], color="#1fd1a5"))
        ax.arrow(self.robot.x, self.robot.y, 0.5 * math.cos(self.robot.theta), 0.5 * math.sin(self.robot.theta),
                 width=0.03, color="black")
        for obstacle in self.dynamic_obstacles:
            ax.add_patch(Circle((obstacle.x, obstacle.y), obstacle.radius, color="#ef476f"))
        ax.scatter(*self.scenario.goal, marker="*", s=180, color="#ffd166", edgecolor="black", label="goal")
        policy = getattr(self, "policy_name", "policy")
        result = "SUCCESS" if self.success else self._failure_type() or "RUNNING"
        ax.set_title(f"{policy} | {self.scenario.scenario_id} | {result}\n"
                     f"v={self.robot.v:.2f} w={self.robot.w:.2f} clearance={self._minimum_clearance():.2f} "
                     f"reward={self.cumulative_reward:.1f} shield={self.shield_counts}")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        if self.render_mode == "human":
            plt.show(block=False)
            plt.pause(0.001)
        plt.close(fig)
        return image if self.render_mode == "rgb_array" else None

    def save_render(self, path: str | Path) -> None:
        old_mode, self.render_mode = self.render_mode, "rgb_array"
        image = self.render()
        self.render_mode = old_mode
        import matplotlib.pyplot as plt
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(path, image)

    def close(self) -> None:
        return None
