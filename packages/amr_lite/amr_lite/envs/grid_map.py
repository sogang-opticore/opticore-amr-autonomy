"""Occupancy-grid maps for AMR-Lite (ROS ``map_server`` format).

``GridMap`` implements the same interface as ``WarehouseMap`` (clearance,
collision, A* occupancy, vectorized ray casting) on top of an exact Euclidean
feature transform, so the opticore-amr warehouse map can be used for training
and for Lite <-> Gazebo paired evaluation.

Lite coordinates put the lower-left map corner at the origin; ``map_origin``
converts to and from the ROS ``map`` frame.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from amr_lite.config import ROOT

from .geometry import bilinear_centres

_SQRT2 = math.sqrt(2.0)


def _nearest_in_columns(occupied: np.ndarray, sentinel: int) -> tuple[np.ndarray, np.ndarray]:
    """Per column: row distance to and row index of the nearest occupied cell."""
    rows, cols = occupied.shape
    distance = np.full((rows, cols), sentinel, dtype=np.int64)
    nearest = np.full((rows, cols), -1, dtype=np.int64)
    last = np.full(cols, -1, dtype=np.int64)
    for row in range(rows):
        last = np.where(occupied[row], row, last)
        seen = last >= 0
        distance[row] = np.where(seen, row - last, sentinel)
        nearest[row] = last
    following = np.full(cols, -1, dtype=np.int64)
    for row in range(rows - 1, -1, -1):
        following = np.where(occupied[row], row, following)
        candidate = np.where(following >= 0, following - row, sentinel)
        better = candidate < distance[row]
        distance[row] = np.where(better, candidate, distance[row])
        nearest[row] = np.where(better, following, nearest[row])
    return distance, nearest


def feature_transform(occupied: np.ndarray, max_cells: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Euclidean distance transform with nearest-cell indices.

    Distances (in cells, between cell centres) are exact below ``max_cells``
    and saturate at ``max_cells`` beyond it. The returned row/col give the
    nearest occupied cell, or -1 where the distance saturated.
    """
    rows, cols = occupied.shape
    capped = max_cells + 1
    column_distance, column_row = _nearest_in_columns(occupied, capped)
    column_distance = np.minimum(column_distance, capped)
    squared = column_distance * column_distance
    best = np.full((rows, cols), 2 * capped * capped, dtype=np.int64)
    best_row = np.full((rows, cols), -1, dtype=np.int64)
    best_col = np.full((rows, cols), -1, dtype=np.int64)
    columns = np.arange(cols)
    for shift in range(-max_cells, max_cells + 1):
        lo, hi = max(0, -shift), min(cols, cols - shift)
        if lo >= hi:
            continue
        candidate = squared[:, lo + shift:hi + shift] + shift * shift
        better = candidate < best[:, lo:hi]
        best[:, lo:hi] = np.where(better, candidate, best[:, lo:hi])
        best_row[:, lo:hi] = np.where(better, column_row[:, lo + shift:hi + shift], best_row[:, lo:hi])
        best_col[:, lo:hi] = np.where(better, columns[lo + shift:hi + shift][None, :], best_col[:, lo:hi])
    distance = np.sqrt(best.astype(np.float64))
    saturated = distance >= max_cells
    distance[saturated] = max_cells
    best_row[saturated] = -1
    best_col[saturated] = -1
    return distance, best_row, best_col


@dataclass
class GridMap:
    occupied_cells: np.ndarray            # bool (rows, cols); row 0 is the bottom edge
    resolution: float
    map_origin: tuple[float, float] = (0.0, 0.0)
    name: str = "grid"
    max_distance: float = 4.0
    max_ray_iterations: int = 64
    obstacles: list = field(default_factory=list)   # rectangle API compatibility; always empty
    _distance: np.ndarray = field(init=False, repr=False, compare=False)
    _nearest_row: np.ndarray = field(init=False, repr=False, compare=False)
    _nearest_col: np.ndarray = field(init=False, repr=False, compare=False)
    _ray_field: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.occupied_cells = np.ascontiguousarray(self.occupied_cells, dtype=bool)
        max_cells = max(1, int(math.ceil(self.max_distance / self.resolution)))
        cached = _FEATURE_CACHE.get(self.fingerprint(max_cells))
        if cached is None:
            cached = feature_transform(self.occupied_cells, max_cells)
            _FEATURE_CACHE[self.fingerprint(max_cells)] = cached
        self._distance, self._nearest_row, self._nearest_col = cached
        # Cell-centre distances in metres with a one-cell border of zeros so
        # rays leaving the map hit the border without masked indexing.
        padded = np.zeros((self.rows + 2, self.cols + 2), dtype=np.float64)
        padded[1:-1, 1:-1] = self._distance * self.resolution
        self._ray_field = padded.ravel()

    def fingerprint(self, max_cells: int | None = None) -> str:
        digest = hashlib.sha256(np.packbits(self.occupied_cells).tobytes())
        digest.update(repr((self.occupied_cells.shape, max_cells)).encode("utf-8"))
        return digest.hexdigest()[:16]

    @property
    def rows(self) -> int:
        return int(self.occupied_cells.shape[0])

    @property
    def cols(self) -> int:
        return int(self.occupied_cells.shape[1])

    @property
    def width(self) -> float:
        return self.cols * self.resolution

    @property
    def height(self) -> float:
        return self.rows * self.resolution

    def to_lite(self, x: float, y: float) -> tuple[float, float]:
        return x - self.map_origin[0], y - self.map_origin[1]

    def to_map(self, x: float, y: float) -> tuple[float, float]:
        return x + self.map_origin[0], y + self.map_origin[1]

    def all_rectangles(self) -> list:
        return []

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(y / self.resolution)), int(math.floor(x / self.resolution))

    def _inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def clearance(self, x: float, y: float, radius: float = 0.0) -> float:
        """Distance from ``(x, y)`` to the nearest occupied cell square, minus ``radius``.

        Exact up to the choice of nearest cells among the 3 x 3 neighbourhood.
        Points outside the map are treated as inside an obstacle.
        """
        row, col = self.cell(x, y)
        if not self._inside(row, col):
            return -radius - self.resolution
        if self.occupied_cells[row, col]:
            return -radius
        if self._nearest_row[row, col] < 0:
            return self.max_distance - radius
        r0, r1 = max(0, row - 1), min(self.rows, row + 2)
        c0, c1 = max(0, col - 1), min(self.cols, col + 2)
        rows = self._nearest_row[r0:r1, c0:c1].ravel()
        cols = self._nearest_col[r0:r1, c0:c1].ravel()
        valid = rows >= 0
        rows, cols = rows[valid], cols[valid]
        half = 0.5 * self.resolution
        dx = np.maximum(np.abs(x - (cols + 0.5) * self.resolution) - half, 0.0)
        dy = np.maximum(np.abs(y - (rows + 0.5) * self.resolution) - half, 0.0)
        return float(np.min(np.hypot(dx, dy))) - radius

    def collides(self, x: float, y: float, radius: float) -> bool:
        return self.clearance(x, y, radius) <= 0.0

    def occupied(self, x: float, y: float, inflation: float = 0.0) -> bool:
        """Fast planning check (cell-centre field, error below half a cell)."""
        row, col = self.cell(x, y)
        if not self._inside(row, col):
            return True
        return (self._distance[row, col] - 0.5) * self.resolution <= inflation

    def planning_clearance(self, x: float, y: float) -> float:
        """Obstacle clearance for planner costs: bilinear cell-centre field (0 outside)."""
        row, col = self.cell(x, y)
        if not self._inside(row, col):
            return 0.0
        distance = bilinear_centres(self._distance, self.resolution, x, y)
        return max(0.0, (distance - 0.5) * self.resolution)

    def world_to_grid(self, x: float, y: float, resolution: float) -> tuple[int, int]:
        return int(x / resolution), int(y / resolution)

    def grid_to_world(self, col: int, row: int, resolution: float) -> tuple[float, float]:
        return (col + 0.5) * resolution, (row + 0.5) * resolution

    def ray_distances(self, ox: float, oy: float, dx: np.ndarray, dy: np.ndarray,
                      max_range: float) -> np.ndarray:
        """Sphere tracing on the distance field, vectorized over rays.

        A ray stops when its sample point lies in an occupied cell. Steps use
        the conservative bound ``d(cell centre) - sqrt(2) * resolution`` with a
        quarter-cell floor (larger floors leak through one-cell diagonal
        walls); rays that do not converge report their current, shorter,
        distance.
        """
        count = dx.size
        result = np.full(count, max_range, dtype=np.float64)
        travelled = np.zeros(count, dtype=np.float64)
        active = np.arange(count)
        inverse, width = 1.0 / self.resolution, self.cols + 2
        margin, floor = _SQRT2 * self.resolution, 0.25 * self.resolution
        for _ in range(self.max_ray_iterations):
            if active.size == 0:
                break
            t = travelled[active]
            col = np.clip(np.floor((ox + t * dx[active]) * inverse), -1, self.cols).astype(np.int64)
            row = np.clip(np.floor((oy + t * dy[active]) * inverse), -1, self.rows).astype(np.int64)
            centre = self._ray_field[(row + 1) * width + col + 1]
            hit = centre <= 0.0
            result[active[hit]] = np.minimum(t[hit], max_range)
            advanced = t + np.maximum(centre - margin, floor)
            travelled[active] = advanced
            active = active[~hit & (advanced < max_range)]
        if active.size:
            result[active] = np.minimum(travelled[active], max_range)
        return result

    def free_cells(self, min_clearance: float) -> np.ndarray:
        """``(N, 2)`` cell-centre coordinates whose clearance exceeds ``min_clearance``."""
        mask = (self._distance - 0.5) * self.resolution > min_clearance
        rows, cols = np.nonzero(mask)
        return np.stack(((cols + 0.5) * self.resolution, (rows + 0.5) * self.resolution), axis=1)


_FEATURE_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
_MAP_CACHE: dict[tuple, GridMap] = {}


def read_pgm(path: str | Path) -> np.ndarray:
    """Read a binary (P5) or ASCII (P2) PGM as uint8, row 0 at the top."""
    data = Path(path).read_bytes()
    tokens: list[bytes] = []
    position = 0
    while len(tokens) < 4:
        while data[position:position + 1].isspace():
            position += 1
        if data[position:position + 1] == b"#":
            position = data.index(b"\n", position) + 1
            continue
        end = position
        while end < len(data) and not data[end:end + 1].isspace():
            end += 1
        tokens.append(data[position:end])
        position = end
    magic, width, height, maxval = tokens[0], int(tokens[1]), int(tokens[2]), int(tokens[3])
    if maxval > 255:
        raise ValueError("16-bit PGM maps are not supported")
    if magic == b"P5":
        pixels = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=position + 1)
    elif magic == b"P2":
        pixels = np.asarray(data[position:].split()[:width * height], dtype=np.int64).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported PGM type {magic!r}")
    return pixels.reshape(height, width)


def load_ros_map(yaml_path: str | Path, *, unknown_values: Sequence[int] = (205,),
                 unknown_as_occupied: bool = True, name: str | None = None,
                 max_distance: float = 4.0) -> GridMap:
    """Load a ROS ``map_server`` YAML + PGM.

    Pixels are classified with the map_server trinary rule. ``unknown_values``
    marks pixels that map_saver writes for unexplored space; with the common
    ``free_thresh: 0.25`` map_server reads them as free although they are
    shelf interiors or outside walls, so Lite treats them as occupied.
    """
    yaml_path = Path(yaml_path)
    key = (str(yaml_path.resolve()), tuple(unknown_values), unknown_as_occupied, max_distance)
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    meta: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    image = read_pgm(yaml_path.parent / meta["image"])
    shade = image.astype(np.float64) / 255.0
    occupancy = shade if int(meta.get("negate", 0)) else 1.0 - shade
    occupied = occupancy > float(meta.get("occupied_thresh", 0.65))
    free = occupancy < float(meta.get("free_thresh", 0.25))
    unknown = ~occupied & ~free
    if unknown_values:
        unknown |= np.isin(image, np.asarray(unknown_values, dtype=np.uint8))
    if unknown_as_occupied:
        occupied |= unknown
    origin = meta.get("origin", [0.0, 0.0, 0.0])
    grid = GridMap(np.flipud(occupied), float(meta["resolution"]),
                   (float(origin[0]), float(origin[1])), name or yaml_path.stem,
                   max_distance=max_distance)
    _MAP_CACHE[key] = grid
    return grid


def named_map(name: str) -> GridMap:
    """Maps shipped in ``assets/maps``: ``opticore_warehouse``."""
    directory = ROOT / "assets" / "maps" / name
    candidates = sorted(directory.glob("*.yaml"))
    if not candidates:
        raise FileNotFoundError(f"No map YAML in {directory}")
    return load_ros_map(candidates[0], name=name)
