from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import Rectangle, bilinear_centres, circle_rectangle_clearance


@dataclass
class WarehouseMap:
    width: float = 12.0
    height: float = 8.0
    obstacles: list[Rectangle] = field(default_factory=list)
    _walls: tuple[tuple[float, float], list[Rectangle]] | None = field(
        default=None, init=False, repr=False, compare=False)
    _raster: tuple[tuple, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)

    PLANNING_RESOLUTION = 0.05

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

    def _clearance_raster(self) -> np.ndarray:
        """Obstacle clearance at 5 cm cell centres, for planner cost lookups.

        Cached per (size, obstacle list); replacing the map object or
        appending obstacles invalidates it.
        """
        key = (self.width, self.height, len(self.obstacles), id(self.obstacles))
        if self._raster is None or self._raster[0] != key:
            step = self.PLANNING_RESOLUTION
            xs = (np.arange(int(np.ceil(self.width / step))) + 0.5) * step
            ys = (np.arange(int(np.ceil(self.height / step))) + 0.5) * step
            rects = np.asarray([(r.x_min, r.y_min, r.x_max, r.y_max) for r in self.all_rectangles()])
            dx = np.maximum(np.maximum(rects[:, 0][:, None] - xs[None, :], xs[None, :] - rects[:, 2][:, None]), 0.0)
            dy = np.maximum(np.maximum(rects[:, 1][:, None] - ys[None, :], ys[None, :] - rects[:, 3][:, None]), 0.0)
            field = np.full((ys.size, xs.size), np.inf)
            for index in range(len(rects)):
                np.minimum(field, np.hypot(dx[index][None, :], dy[index][:, None]), out=field)
            self._raster = (key, field)
        return self._raster[1]

    def planning_clearance(self, x: float, y: float) -> float:
        """Obstacle clearance for planner costs: bilinear on a 5 cm raster (0 outside)."""
        if not (0.0 <= x <= self.width and 0.0 <= y <= self.height):
            return 0.0
        return bilinear_centres(self._clearance_raster(), self.PLANNING_RESOLUTION, x, y)

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
