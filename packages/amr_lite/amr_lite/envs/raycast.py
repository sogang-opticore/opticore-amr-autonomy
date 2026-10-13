from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .dynamics import RobotState
from .geometry import ray_circle_distance, ray_rectangle_distance
from .map import WarehouseMap
from .scenarios import DynamicObstacle


def ray_directions(angles: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Unit direction vectors for absolute ray angles.

    ``math.cos``/``math.sin`` are used per ray so the vectorized cast stays
    bit-identical to the scalar reference implementation.
    """
    count = len(angles)
    dx = np.fromiter((math.cos(angle) for angle in angles), dtype=np.float64, count=count)
    dy = np.fromiter((math.sin(angle) for angle in angles), dtype=np.float64, count=count)
    return dx, dy


def circle_ray_distances(ox: float, oy: float, dx: np.ndarray, dy: np.ndarray,
                         dynamic_obstacles: Sequence[DynamicObstacle],
                         max_range: float) -> np.ndarray:
    """First intersection of every ray with any circle, or ``max_range``."""
    if not dynamic_obstacles:
        return np.full(dx.shape, max_range, dtype=np.float64)
    cx = np.asarray([item.x for item in dynamic_obstacles], dtype=np.float64)
    cy = np.asarray([item.y for item in dynamic_obstacles], dtype=np.float64)
    radius = np.asarray([item.radius for item in dynamic_obstacles], dtype=np.float64)
    qx, qy = ox - cx, oy - cy
    b = 2.0 * (qx[None, :] * dx[:, None] + qy[None, :] * dy[:, None])
    c = qx * qx + qy * qy - radius * radius
    disc = b * b - 4.0 * c[None, :]
    root = np.sqrt(np.maximum(disc, 0.0))
    first = (-b - root) / 2.0
    second = (-b + root) / 2.0
    hit = np.where(first >= 0.0, first, np.where(second >= 0.0, second, max_range))
    hit = np.where(disc < 0.0, max_range, hit)
    return np.minimum(max_range, hit.min(axis=1))


def cast_rays(ox: float, oy: float, angles: Sequence[float], warehouse_map,
              dynamic_obstacles: Sequence[DynamicObstacle], max_range: float) -> np.ndarray:
    """Vectorized ray cast against static geometry and circular obstacles.

    Returns float64 distances clipped to ``[0, max_range]``. Static geometry is
    delegated to ``warehouse_map.ray_distances`` so rectangle and grid maps
    share this entry point.
    """
    dx, dy = ray_directions(angles)
    static = warehouse_map.ray_distances(ox, oy, dx, dy, max_range)
    dynamic = circle_ray_distances(ox, oy, dx, dy, dynamic_obstacles, max_range)
    distance = np.minimum(static, dynamic)
    # ``+ 0.0`` folds a possible -0.0 into +0.0, matching ``max(0.0, ...)``.
    return np.maximum(np.minimum(distance, max_range), 0.0) + 0.0


def lidar_scan(state: RobotState, warehouse_map: WarehouseMap,
               dynamic_obstacles: list[DynamicObstacle], ray_count: int = 72,
               fov: float = math.radians(270.0), max_range: float = 8.0) -> np.ndarray:
    relative = np.linspace(-fov / 2.0, fov / 2.0, ray_count, dtype=np.float64)
    angles = [state.theta + float(value) for value in relative]
    return cast_rays(state.x, state.y, angles, warehouse_map, dynamic_obstacles,
                     max_range).astype(np.float32)


def lidar_scan_reference(state: RobotState, warehouse_map: WarehouseMap,
                         dynamic_obstacles: list[DynamicObstacle], ray_count: int = 72,
                         fov: float = math.radians(270.0), max_range: float = 8.0) -> np.ndarray:
    """Original per-ray implementation, kept as the parity reference."""
    angles = np.linspace(-fov / 2.0, fov / 2.0, ray_count, dtype=np.float64)
    ranges = np.full(ray_count, max_range, dtype=np.float32)
    for index, relative in enumerate(angles):
        angle = state.theta + float(relative)
        dx, dy = math.cos(angle), math.sin(angle)
        distance = max_range
        for rect in warehouse_map.all_rectangles():
            distance = min(distance, ray_rectangle_distance(state.x, state.y, dx, dy, rect, max_range))
        for obstacle in dynamic_obstacles:
            distance = min(distance, ray_circle_distance(state.x, state.y, dx, dy,
                                                          obstacle.x, obstacle.y, obstacle.radius, max_range))
        ranges[index] = max(0.0, min(max_range, distance))
    return ranges
