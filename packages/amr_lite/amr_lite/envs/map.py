from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Rectangle, circle_rectangle_clearance


@dataclass
class WarehouseMap:
    width: float = 12.0
    height: float = 8.0
    obstacles: list[Rectangle] = field(default_factory=list)
    _walls: tuple[tuple[float, float], list[Rectangle]] | None = field(
        default=None, init=False, repr=False, compare=False)

    def _boundary(self) -> list[Rectangle]:
        key = (self.width, self.height)
        if self._walls is None or self._walls[0] != key:
            t = 0.05
            self._walls = (key, [Rectangle(-t, -t, 0.0, self.height + t),
                                 Rectangle(self.width, -t, self.width + t, self.height + t),
                                 Rectangle(0.0, -t, self.width, 0.0),
                                 Rectangle(0.0, self.height, self.width, self.height + t)])
        return self._walls[1]

    def all_rectangles(self) -> list[Rectangle]:
        return self.obstacles + self._boundary()

    def clearance(self, x: float, y: float, radius: float = 0.0) -> float:
        return min(circle_rectangle_clearance(x, y, radius, rect) for rect in self.all_rectangles())

    def collides(self, x: float, y: float, radius: float) -> bool:
        return self.clearance(x, y, radius) <= 0.0

    def occupied(self, x: float, y: float, inflation: float = 0.0) -> bool:
        return self.collides(x, y, inflation)

    def world_to_grid(self, x: float, y: float, resolution: float) -> tuple[int, int]:
        return int(x / resolution), int(y / resolution)

    def grid_to_world(self, col: int, row: int, resolution: float) -> tuple[float, float]:
        return (col + 0.5) * resolution, (row + 0.5) * resolution

    def ray_distances(self, ox: float, oy: float, dx: np.ndarray, dy: np.ndarray,
                      max_range: float) -> np.ndarray:
        """Slab-method ray cast of every ray against every rectangle.

        Mirrors ``geometry.ray_rectangle_distance`` operation by operation so
        the result is bit-identical to the scalar loop.
        """
        rects = np.asarray([(rect.x_min, rect.y_min, rect.x_max, rect.y_max)
                            for rect in self.all_rectangles()], dtype=np.float64)
        x_min, y_min, x_max, y_max = rects.T
        parallel_x = (np.abs(dx) < 1e-12)[:, None]
        parallel_y = (np.abs(dy) < 1e-12)[:, None]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ax = (x_min - ox)[None, :] / dx[:, None]
            bx = (x_max - ox)[None, :] / dx[:, None]
            ay = (y_min - oy)[None, :] / dy[:, None]
            by = (y_max - oy)[None, :] / dy[:, None]
        t_min = np.where(parallel_x, -np.inf, np.minimum(ax, bx))
        t_max = np.where(parallel_x, np.inf, np.maximum(ax, bx))
        t_min = np.where(parallel_y, t_min, np.maximum(t_min, np.minimum(ay, by)))
        t_max = np.where(parallel_y, t_max, np.minimum(t_max, np.maximum(ay, by)))
        miss = (parallel_x & ((ox < x_min) | (ox > x_max))[None, :]) | \
               (parallel_y & ((oy < y_min) | (oy > y_max))[None, :]) | (t_min > t_max)
        hit = np.where(t_min >= 0.0, t_min, t_max)
        valid = ~miss & (hit >= 0.0) & (hit <= max_range)
        return np.where(valid, hit, max_range).min(axis=1, initial=max_range)
