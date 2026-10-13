from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class Rectangle:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def inflated(self, amount: float) -> "Rectangle":
        return Rectangle(self.x_min - amount, self.y_min - amount,
                         self.x_max + amount, self.y_max + amount)

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


def point_segment_projection(px: float, py: float, ax: float, ay: float,
                             bx: float, by: float) -> tuple[float, float, float]:
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    t = 0.0 if denom <= 1e-12 else clamp(((px - ax) * dx + (py - ay) * dy) / denom, 0.0, 1.0)
    return ax + t * dx, ay + t * dy, t


def circle_rectangle_clearance(x: float, y: float, radius: float, rect: Rectangle) -> float:
    nx = clamp(x, rect.x_min, rect.x_max)
    ny = clamp(y, rect.y_min, rect.y_max)
    return math.hypot(x - nx, y - ny) - radius


def ray_rectangle_distance(ox: float, oy: float, dx: float, dy: float,
                           rect: Rectangle, max_range: float) -> float:
    t_min, t_max = -math.inf, math.inf
    for origin, direction, low, high in ((ox, dx, rect.x_min, rect.x_max),
                                          (oy, dy, rect.y_min, rect.y_max)):
        if abs(direction) < 1e-12:
            if origin < low or origin > high:
                return max_range
            continue
        a, b = (low - origin) / direction, (high - origin) / direction
        if a > b:
            a, b = b, a
        t_min, t_max = max(t_min, a), min(t_max, b)
        if t_min > t_max:
            return max_range
    hit = t_min if t_min >= 0.0 else t_max
    return hit if 0.0 <= hit <= max_range else max_range


def ray_circle_distance(ox: float, oy: float, dx: float, dy: float,
                        cx: float, cy: float, radius: float, max_range: float) -> float:
    qx, qy = ox - cx, oy - cy
    b = 2.0 * (qx * dx + qy * dy)
    c = qx * qx + qy * qy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return max_range
    root = math.sqrt(disc)
    candidates = [t for t in ((-b - root) / 2.0, (-b + root) / 2.0) if t >= 0.0]
    return min(candidates, default=max_range)



def bilinear_centres(field: np.ndarray, step: float, x: float, y: float) -> float:
    """Bilinear interpolation of a field sampled at cell centres ``(i + 0.5) * step``.

    Queries outside the centre lattice are clamped to the border values.
    """
    rows, cols = field.shape
    fx = min(max(x / step - 0.5, 0.0), cols - 1.0)
    fy = min(max(y / step - 0.5, 0.0), rows - 1.0)
    c0, r0 = min(int(fx), cols - 2) if cols > 1 else 0, min(int(fy), rows - 2) if rows > 1 else 0
    tx, ty = fx - c0, fy - r0
    c1, r1 = min(c0 + 1, cols - 1), min(r0 + 1, rows - 1)
    top = field[r0, c0] * (1.0 - tx) + field[r0, c1] * tx
    bottom = field[r1, c0] * (1.0 - tx) + field[r1, c1] * tx
    return float(top * (1.0 - ty) + bottom * ty)
