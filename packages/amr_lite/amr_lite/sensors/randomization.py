"""Per-episode sensor, actuation and localization model with domain randomization.

Every value defaults to a no-op so the Phase 0/M1 environment is unchanged.
Randomized values are drawn from a stream derived from the episode seed only,
so they never consume the environment's own random generator.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

INTEGER_KEYS = frozenset({"observation_delay_steps", "action_delay_steps"})


@dataclass
class SensorDynamicsModel:
    lidar_type: str = "contract"
    scan_period: float = 0.05
    scan_phase: float = 0.0
    range_noise_std: float = 0.0
    dropout_probability: float = 0.0
    angular_bias_deg: float = 0.0
    observation_delay_steps: int = 0
    action_delay_steps: int = 0
    v_scale: float = 1.0
    w_scale: float = 1.0
    actuator_time_constant: float = 0.0
    collision_radius_margin: float = 0.0
    localization_bias_xy: float = 0.0
    localization_bias_theta: float = 0.0
    localization_noise_xy: float = 0.0
    localization_noise_theta: float = 0.0
    path_resample_spacing: float = 0.0
    path_lateral_noise: float = 0.0
    randomized: bool = False
    bias_x: float = 0.0
    bias_y: float = 0.0
    bias_theta: float = 0.0

    @property
    def scan_noise(self) -> bool:
        return self.range_noise_std > 0.0 or self.dropout_probability > 0.0

    @property
    def actuation_identity(self) -> bool:
        return self.v_scale == 1.0 and self.w_scale == 1.0 and self.actuator_time_constant <= 0.0

    @property
    def localization_identity(self) -> bool:
        return (self.bias_x == 0.0 and self.bias_y == 0.0 and self.bias_theta == 0.0
                and self.localization_noise_xy <= 0.0 and self.localization_noise_theta <= 0.0)

    @property
    def perturbs_path(self) -> bool:
        return self.path_resample_spacing > 0.0 or self.path_lateral_noise > 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_RANDOMIZABLE = tuple(item.name for item in fields(SensorDynamicsModel)
                      if item.name not in {"lidar_type", "randomized", "bias_x", "bias_y", "bias_theta"})


def episode_sensor_model(config: dict[str, Any], seed: int) -> tuple[SensorDynamicsModel, np.random.Generator]:
    """Build the episode model and the noise generator used during the episode."""
    base = dict(config.get("sensor_model") or {})
    model = SensorDynamicsModel(
        lidar_type=str((config.get("lidar_model") or {}).get("type", "contract")),
        scan_period=float(base.get("scan_period", config.get("policy_dt", 0.05))),
    )
    for name in _RANDOMIZABLE:
        if name in base and name != "scan_period":
            setattr(model, name, _cast(name, base[name]))
    sequence = np.random.SeedSequence([int(seed) % 2**32, zlib.crc32(b"sensor-dynamics-model")])
    parameter_rng, noise_rng = (np.random.default_rng(child) for child in sequence.spawn(2))
    randomization = config.get("domain_randomization") or {}
    if bool(randomization.get("enabled", False)):
        if parameter_rng.uniform() < float(randomization.get("probability", 1.0)):
            ranges = randomization.get("ranges") or {}
            for name in sorted(ranges):
                if name not in _RANDOMIZABLE:
                    raise KeyError(f"Unknown domain randomization key: {name}")
                low, high = ranges[name]
                if name in INTEGER_KEYS:
                    value: float | int = int(parameter_rng.integers(int(low), int(high) + 1))
                else:
                    value = float(parameter_rng.uniform(float(low), float(high)))
                setattr(model, name, value)
            model.randomized = True
        if bool(randomization.get("randomize_scan_phase", False)):
            model.scan_phase = float(parameter_rng.uniform(0.0, model.scan_period))
    if model.localization_bias_xy > 0.0:
        heading = float(parameter_rng.uniform(-math.pi, math.pi))
        model.bias_x = model.localization_bias_xy * math.cos(heading)
        model.bias_y = model.localization_bias_xy * math.sin(heading)
    if model.localization_bias_theta > 0.0:
        model.bias_theta = model.localization_bias_theta * float(parameter_rng.choice((-1.0, 1.0)))
    return model, noise_rng


def _cast(name: str, value: Any) -> float | int:
    return int(value) if name in INTEGER_KEYS else float(value)


def noisy_raw_ranges(ranges: np.ndarray, model: SensorDynamicsModel,
                     rng: np.random.Generator) -> np.ndarray:
    """Gaussian range noise on returns and dropouts reported as no-return."""
    output = np.array(ranges, dtype=np.float32, copy=True)
    if not model.scan_noise:
        return output
    hit = np.isfinite(output)
    if model.range_noise_std > 0.0:
        output[hit] += rng.normal(0.0, model.range_noise_std, int(hit.sum())).astype(np.float32)
        np.maximum(output, 0.0, out=output, where=hit)
    if model.dropout_probability > 0.0:
        output[rng.uniform(size=output.shape) < model.dropout_probability] = np.inf
    return output


def noisy_contract_scan(scan: np.ndarray, model: SensorDynamicsModel, rng: np.random.Generator,
                        max_range: float) -> np.ndarray:
    output = np.array(scan, dtype=np.float32, copy=True)
    if not model.scan_noise:
        return output
    hit = output < max_range
    if model.range_noise_std > 0.0:
        output[hit] += rng.normal(0.0, model.range_noise_std, int(hit.sum())).astype(np.float32)
    if model.dropout_probability > 0.0:
        output[rng.uniform(size=output.shape) < model.dropout_probability] = max_range
    return np.clip(output, 0.0, max_range).astype(np.float32)


def perturb_path(path: list[tuple[float, float]], spacing: float, lateral_noise: float,
                 rng: np.random.Generator) -> list[tuple[float, float]]:
    """Densify a polyline and add smooth lateral noise, keeping both endpoints.

    Models the A* output of the ROS stack (dense, smoothed, slightly uneven)
    rather than the sparse simplified polyline of the Lite planner.
    """
    if len(path) < 2:
        return list(path)
    points = [tuple(path[0])]
    for start, end in zip(path, path[1:]):
        length = math.dist(start, end)
        pieces = max(1, int(math.ceil(length / spacing))) if spacing > 0.0 else 1
        for index in range(1, pieces + 1):
            ratio = index / pieces
            points.append((start[0] + ratio * (end[0] - start[0]),
                           start[1] + ratio * (end[1] - start[1])))
    array = np.asarray(points, dtype=np.float64)
    if lateral_noise > 0.0 and len(array) > 2:
        window = 5
        raw = rng.normal(0.0, lateral_noise * math.sqrt(window), len(array))
        offsets = np.convolve(raw, np.ones(window) / window, mode="same")
        offsets[0] = offsets[-1] = 0.0
        tangent = np.gradient(array, axis=0)
        norm = np.maximum(np.hypot(tangent[:, 0], tangent[:, 1]), 1e-9)
        normal = np.stack((-tangent[:, 1] / norm, tangent[:, 0] / norm), axis=1)
        array = array + normal * offsets[:, None]
        array[0], array[-1] = points[0], points[-1]
    return [(float(x), float(y)) for x, y in array]
