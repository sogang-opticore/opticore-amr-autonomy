"""Procedural warehouse-like scenarios with split-specific generation rules.

``P-<FAMILY>`` builds a static layout, ``PD-<FAMILY>`` adds scripted agents
placed relative to the planned path. Every instance is reproducible from
``(scenario id, split, seed)`` and is only returned once A* finds a path of
acceptable length with the environment's own resolution and inflation.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import zlib
from typing import Any, Callable

import numpy as np

from amr_lite.config import load_config
from amr_lite.planning.astar import astar, clearance_key

from .dynamics import RobotState
from .geometry import Rectangle, circle_rectangle_clearance
from .map import WarehouseMap
from .scenarios import DynamicObstacle, Scenario

Point = tuple[float, float]


@functools.lru_cache(maxsize=8)
def procedural_spec(name: str = "procedural") -> dict[str, Any]:
    return load_config(name)


def _uniform(rng: np.random.Generator, bounds: list[float] | tuple[float, float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _integer(rng: np.random.Generator, bounds: list[int] | tuple[int, int]) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _clip_rect(rect: Rectangle, width: float, height: float) -> Rectangle | None:
    clipped = Rectangle(max(0.0, rect.x_min), max(0.0, rect.y_min),
                        min(width, rect.x_max), min(height, rect.y_max))
    if clipped.x_max - clipped.x_min < 0.05 or clipped.y_max - clipped.y_min < 0.05:
        return None
    return clipped


def _mirror(width: float, obstacles: list[Rectangle], start: Point, goal: Point
            ) -> tuple[list[Rectangle], Point, Point]:
    mirrored = [Rectangle(width - rect.x_max, rect.y_min, width - rect.x_min, rect.y_max)
                for rect in obstacles]
    return mirrored, (width - start[0], start[1]), (width - goal[0], goal[1])


def _aisle(rng: np.random.Generator, width: float, height: float, p: dict
           ) -> tuple[list[Rectangle], Point, Point]:
    aisle, depth = _uniform(rng, p["aisle_width"]), _uniform(rng, p["shelf_depth"])
    gap, gaps = _uniform(rng, p["gap_width"]), _integer(rng, p["gaps_per_row"])
    margin = aisle * _uniform(rng, (0.8, 1.1))
    rows, y = [], margin
    while y + depth <= height - margin + 1e-9:
        rows.append(y)
        y += depth + aisle
    obstacles: list[Rectangle] = []
    x0, x1 = margin, width - margin
    for row in rows:
        centres = sorted(_uniform(rng, (x0 + gap, x1 - gap)) for _ in range(gaps))
        cursor = x0
        for centre in centres:
            left, right = centre - gap / 2.0, centre + gap / 2.0
            if left - cursor >= 0.5:
                obstacles.append(Rectangle(cursor, row, left, row + depth))
            cursor = max(cursor, right)
        if x1 - cursor >= 0.5:
            obstacles.append(Rectangle(cursor, row, x1, row + depth))
    top = rows[-1] + depth if rows else 0.0
    lanes = [margin / 2.0] + [row + depth + aisle / 2.0 for row in rows[:-1]] + [(top + height) / 2.0]
    start_lane = lanes[int(rng.integers(len(lanes)))]
    goal_lane = (lanes[int(rng.integers(len(lanes)))] if rng.uniform() < 0.7 else start_lane)
    start = (_uniform(rng, (0.6, 0.3 * width)), start_lane)
    goal = (_uniform(rng, (0.7 * width, width - 0.6)), goal_lane)
    return obstacles, start, goal


def _hall(rng: np.random.Generator, width: float, height: float, p: dict
          ) -> tuple[list[Rectangle], Point, Point]:
    start = (_uniform(rng, (0.8, 2.0)), _uniform(rng, (1.0, height - 1.0)))
    goal = (_uniform(rng, (width - 2.0, width - 0.8)), _uniform(rng, (1.0, height - 1.0)))
    target, obstacles = _integer(rng, p["pallets"]), []
    for _ in range(target * 6):
        if len(obstacles) >= target:
            break
        w, h = _uniform(rng, p["pallet_size"]), _uniform(rng, p["pallet_size"])
        cx, cy = _uniform(rng, (2.0, width - 2.0)), _uniform(rng, (0.6, height - 0.6))
        rect = _clip_rect(Rectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), width, height)
        if rect is None:
            continue
        if min(circle_rectangle_clearance(*point, 0.0, rect) for point in (start, goal)) < 1.0:
            continue
        obstacles.append(rect)
    return obstacles, start, goal


def _pinch(rng: np.random.Generator, width: float, height: float, p: dict
           ) -> tuple[list[Rectangle], Point, Point]:
    corridor, pinch = _uniform(rng, p["corridor_width"]), _uniform(rng, p["pinch_width"])
    pinches, jog = _integer(rng, p["pinches"]), _uniform(rng, p["jog"])
    centre = height / 2.0 + _uniform(rng, (-1.0, 1.0))
    jog_x = _uniform(rng, (0.35 * width, 0.65 * width))
    shifted = centre + jog * float(rng.choice((-1.0, 1.0)))
    obstacles: list[Rectangle] = []
    for x_a, x_b, c in ((0.0, jog_x, centre), (jog_x, width, shifted)):
        obstacles.append(Rectangle(x_a, 0.0, x_b, c - corridor / 2.0))
        obstacles.append(Rectangle(x_a, c + corridor / 2.0, x_b, height))
    placed: list[float] = []
    for _ in range(pinches * 10):
        if len(placed) >= pinches:
            break
        x = _uniform(rng, (2.5, width - 2.5))
        if abs(x - jog_x) < 0.8 or any(abs(x - other) < 1.5 for other in placed):
            continue
        c = centre if x < jog_x else shifted
        depth, length = corridor - pinch, _uniform(rng, (0.4, 1.0))
        if rng.uniform() < 0.5:
            obstacles.append(Rectangle(x - length / 2, c + corridor / 2 - depth, x + length / 2, c + corridor / 2))
        else:
            obstacles.append(Rectangle(x - length / 2, c - corridor / 2, x + length / 2, c - corridor / 2 + depth))
        placed.append(x)
    clipped = [rect for rect in (_clip_rect(item, width, height) for item in obstacles) if rect]
    return clipped, (_uniform(rng, (0.6, 1.2)), centre), (_uniform(rng, (width - 1.2, width - 0.6)), shifted)


_ARMS = {"X": ("W", "E", "N", "S"), "T": ("W", "E", "N"), "L": ("W", "N")}
_ROTATE = {"W": "N", "N": "E", "E": "S", "S": "W"}


def _junction(rng: np.random.Generator, width: float, height: float, p: dict
              ) -> tuple[list[Rectangle], Point, Point]:
    corridor = _uniform(rng, p["corridor_width"])
    arms = list(_ARMS[str(rng.choice(p["shapes"]))])
    for _ in range(int(rng.integers(4))):
        arms = [_ROTATE[arm] for arm in arms]
    xv, yh = _uniform(rng, (0.35 * width, 0.65 * width)), _uniform(rng, (0.35 * height, 0.65 * height))
    h0, h1, v0, v1 = yh - corridor / 2, yh + corridor / 2, xv - corridor / 2, xv + corridor / 2
    obstacles = [Rectangle(0.0, 0.0, v0, h0), Rectangle(v1, 0.0, width, h0),
                 Rectangle(0.0, h1, v0, height), Rectangle(v1, h1, width, height)]
    closed = {"W": Rectangle(0.0, h0, v0, h1), "E": Rectangle(v1, h0, width, h1),
              "N": Rectangle(v0, h1, v1, height), "S": Rectangle(v0, 0.0, v1, h0)}
    obstacles += [rect for arm, rect in closed.items() if arm not in arms]
    ends = {"W": (0.8, yh), "E": (width - 0.8, yh), "N": (xv, height - 0.8), "S": (xv, 0.8)}
    start_arm, goal_arm = (str(item) for item in rng.choice(arms, size=2, replace=False))
    start, goal = ends[start_arm], ends[goal_arm]
    for _ in range(_integer(rng, p["clutter"])):
        arm = closed[str(rng.choice(arms))]
        size = _uniform(rng, (0.4, 0.7))
        cx = _uniform(rng, (arm.x_min + 0.3, max(arm.x_min + 0.31, arm.x_max - 0.3)))
        cy = _uniform(rng, (arm.y_min + 0.3, max(arm.y_min + 0.31, arm.y_max - 0.3)))
        rect = Rectangle(cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)
        if min(circle_rectangle_clearance(*point, 0.0, rect) for point in (start, goal)) > 1.0:
            obstacles.append(rect)
    return obstacles, start, goal


GENERATORS: dict[str, Callable[..., tuple[list[Rectangle], Point, Point]]] = {
    "AISLE": _aisle, "HALL": _hall, "PINCH": _pinch, "JUNCTION": _junction,
}


def path_length(path: list[Point] | tuple[Point, ...]) -> float:
    return sum(math.dist(a, b) for a, b in zip(path, path[1:]))


def point_on_path(path: list[Point] | tuple[Point, ...], distance: float) -> tuple[Point, Point]:
    """Point at arc length ``distance`` and the unit tangent there."""
    remaining = max(0.0, distance)
    for a, b in zip(path, path[1:]):
        length = math.dist(a, b)
        if length <= 1e-9:
            continue
        tangent = ((b[0] - a[0]) / length, (b[1] - a[1]) / length)
        if remaining <= length:
            return (a[0] + tangent[0] * remaining, a[1] + tangent[1] * remaining), tangent
        remaining -= length
    a, b = path[-2], path[-1]
    length = max(math.dist(a, b), 1e-9)
    return tuple(b), ((b[0] - a[0]) / length, (b[1] - a[1]) / length)


def place_agents(rng: np.random.Generator, world, path: list[Point], start: Point, p: dict,
                 nominal_speed: list[float], resolution: float, inflation: float,
                 clearance: dict | None = None) -> list[DynamicObstacle]:
    """Scripted agents placed relative to the planned path.

    Patterns: ``cross`` (passes the path when the robot arrives), ``head_on``,
    ``follow`` (slower agent ahead), ``stop`` (crosses and stops near the
    path), ``static`` (parked beside the path, still passable) and
    ``patrol`` (crosses back and forth).
    """
    total = path_length(path)
    agents: list[DynamicObstacle] = []
    for _ in range(_integer(rng, p["count"])):
        for _attempt in range(20):
            pattern = str(rng.choice(p["patterns"]))
            speed, radius = _uniform(rng, p["speed"]), _uniform(rng, p["radius"])
            robot_speed = _uniform(rng, nominal_speed)
            agent = _make_agent(rng, pattern, path, total, speed, radius, robot_speed)
            if world.clearance(agent.x, agent.y, radius) < 0.05:
                continue
            if math.dist((agent.x, agent.y), start) < 2.0:
                continue
            if pattern == "static" and not _passable(world, path, agent, resolution, inflation, clearance):
                continue
            agents.append(agent)
            break
    return agents


def _make_agent(rng: np.random.Generator, pattern: str, path: list[Point], total: float,
                speed: float, radius: float, robot_speed: float) -> DynamicObstacle:
    side = float(rng.choice((-1.0, 1.0)))
    if pattern in {"cross", "stop", "patrol"}:
        arc = _uniform(rng, (0.35, 0.8)) * total
        (px, py), (tx, ty) = point_on_path(path, arc)
        nx, ny = -ty, tx
        offset = _uniform(rng, (1.5, 3.0))
        delay = max(0.0, arc / robot_speed - offset / speed + _uniform(rng, (-1.0, 1.0)))
        agent = DynamicObstacle(px + side * offset * nx, py + side * offset * ny,
                                -side * speed * nx, -side * speed * ny, radius=radius,
                                motion_type=pattern, start_delay=delay)
        if pattern == "stop":
            agent.stop_time = delay + (offset + _uniform(rng, (-0.4, 0.3))) / speed
        if pattern == "patrol":
            agent.start_delay = _uniform(rng, (0.0, 2.0))
            agent.patrol_distance = 2.0 * offset
        return agent
    if pattern == "head_on":
        (px, py), (tx, ty) = point_on_path(path, _uniform(rng, (0.55, 0.95)) * total)
        lateral = _uniform(rng, (-0.2, 0.2))
        return DynamicObstacle(px - ty * lateral, py + tx * lateral, -speed * tx, -speed * ty,
                               radius=radius, motion_type="approach",
                               start_delay=_uniform(rng, (0.0, 1.5)))
    if pattern == "follow":
        (px, py), (tx, ty) = point_on_path(path, _uniform(rng, (0.2, 0.45)) * total)
        slow = _uniform(rng, (0.15, 0.45))
        return DynamicObstacle(px, py, slow * tx, slow * ty, radius=radius, motion_type="same_direction")
    if pattern == "static":
        (px, py), (tx, ty) = point_on_path(path, _uniform(rng, (0.35, 0.8)) * total)
        lateral = _uniform(rng, (-0.35, 0.35))
        return DynamicObstacle(px - ty * lateral, py + tx * lateral, 0.0, 0.0, radius=radius,
                               motion_type="stopped")
    raise KeyError(f"Unknown agent pattern: {pattern}")


class _WithBlocker:
    """Planning view of a map with one extra square obstacle."""

    def __init__(self, world, rect: Rectangle):
        self.world, self.rect = world, rect
        self.width, self.height = world.width, world.height

    def occupied(self, x: float, y: float, inflation: float = 0.0) -> bool:
        return (self.world.occupied(x, y, inflation)
                or circle_rectangle_clearance(x, y, inflation, self.rect) <= 0.0)

    def clearance(self, x: float, y: float, radius: float = 0.0) -> float:
        return min(self.world.clearance(x, y, radius), circle_rectangle_clearance(x, y, radius, self.rect))

    def planning_clearance(self, x: float, y: float) -> float:
        return min(self.world.planning_clearance(x, y), circle_rectangle_clearance(x, y, 0.0, self.rect))

    def world_to_grid(self, x: float, y: float, resolution: float) -> tuple[int, int]:
        return self.world.world_to_grid(x, y, resolution)

    def grid_to_world(self, col: int, row: int, resolution: float) -> tuple[float, float]:
        return self.world.grid_to_world(col, row, resolution)


def _passable(world, path: list[Point], agent: DynamicObstacle, resolution: float,
              inflation: float, clearance: dict | None = None) -> bool:
    r = agent.radius
    blocker = _WithBlocker(world, Rectangle(agent.x - r, agent.y - r, agent.x + r, agent.y + r))
    return bool(astar(blocker, path[0], path[-1], resolution, inflation, clearance))


def instance_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def agent_payload(agents: list[DynamicObstacle]) -> list[list[Any]]:
    return [[round(a.x, 6), round(a.y, 6), round(a.vx, 6), round(a.vy, 6), round(a.radius, 6),
             a.motion_type, round(a.start_delay, 6), a.stop_time, a.patrol_distance] for a in agents]


def make_procedural_scenario(scenario_id: str, seed: int, config: dict[str, Any]) -> Scenario:
    prefix, _, family = scenario_id.partition("-")
    settings = config.get("procedural") or {}
    spec = procedural_spec(str(settings.get("spec", "procedural")))
    split = str(settings.get("split", "train"))
    if split not in spec["splits"]:
        raise KeyError(f"Unknown procedural split {split!r}; available: {sorted(spec['splits'])}")
    ranges = spec["splits"][split]
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed) % 2**32, zlib.crc32(scenario_id.encode("utf-8")), zlib.crc32(split.encode("utf-8"))]))
    if family == "MIXED":
        family = str(rng.choice(spec["families"]))
    if family not in GENERATORS:
        raise KeyError(f"Unknown procedural family in {scenario_id!r}; use one of {sorted(GENERATORS)} or MIXED")
    width, height = float(spec["map_width"]), float(spec["map_height"])
    resolution = float(config.get("grid_resolution", 0.2))
    inflation = float(config.get("robot_radius", 0.36)) + float(config.get("obstacle_inflation", 0.04))
    clearance = config.get("astar_clearance")
    for _ in range(int(spec.get("max_attempts", 40))):
        obstacles, start, goal = GENERATORS[family](rng, width, height, ranges[family.lower()])
        if rng.uniform() < 0.5:
            obstacles, start, goal = _mirror(width, obstacles, start, goal)
        world = WarehouseMap(width, height, list(obstacles))
        if world.occupied(*start, inflation + 0.05) or world.occupied(*goal, inflation + 0.05):
            continue
        path = astar(world, start, goal, resolution, inflation, clearance)
        if not path or not spec["min_path_length"] <= path_length(path) <= spec["max_path_length"]:
            continue
        heading = math.atan2(path[1][1] - path[0][1], path[1][0] - path[0][0])
        robot = RobotState(start[0], start[1], heading + _uniform(rng, (-0.6, 0.6)))
        agents: list[DynamicObstacle] = []
        if prefix == "PD":
            agents = place_agents(rng, world, path, start, ranges["dynamic"],
                                  spec["nominal_robot_speed"], resolution, inflation, clearance)
        digest = instance_digest({
            "scenario_id": scenario_id, "family": family, "split": split, "seed": int(seed),
            "obstacles": [[round(v, 6) for v in (r.x_min, r.y_min, r.x_max, r.y_max)] for r in obstacles],
            "start": [round(robot.x, 6), round(robot.y, 6), round(robot.theta, 6)],
            "goal": [round(goal[0], 6), round(goal[1], 6)], "agents": agent_payload(agents),
        })
        return Scenario(scenario_id, world, robot, goal, agents, True,
                        f"{scenario_id}-{split}-{int(seed)}-{digest}", tuple(path),
                        (resolution, inflation, clearance_key(clearance)))
    raise RuntimeError(f"Could not generate {scenario_id} (split={split}, seed={seed})")
