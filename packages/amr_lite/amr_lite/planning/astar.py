from __future__ import annotations

import heapq
import math
from typing import Any, Callable

from amr_lite.envs.map import WarehouseMap


def clearance_preference_cost(clearance: float, *, inflation: float, preferred_clearance: float,
                              clearance_cost_weight: float, wall_avoid_clearance: float,
                              wall_avoid_cost_weight: float, wall_avoid_min_margin: float) -> float:
    """Same formula as ``clearance_preference_cost`` in opticore-amr ``astar_node.py``."""
    if not math.isfinite(clearance):
        return 0.0
    cost = 0.0
    if clearance_cost_weight > 0.0 and preferred_clearance > 0.0 and clearance < preferred_clearance:
        ratio = (preferred_clearance - clearance) / preferred_clearance
        cost += clearance_cost_weight * ratio * ratio
    if wall_avoid_cost_weight > 0.0 and wall_avoid_clearance > inflation and clearance < wall_avoid_clearance:
        min_margin = max(wall_avoid_min_margin, 1.0e-6)
        free_margin = max(clearance - inflation, min_margin)
        desired_margin = max(wall_avoid_clearance - inflation, min_margin)
        barrier = max(0.0, desired_margin / free_margin - 1.0)
        cost += wall_avoid_cost_weight * barrier * barrier
    return cost


def clearance_key(settings: dict[str, Any] | None) -> tuple | None:
    """Hashable identity of an ``astar_clearance`` config (None when disabled)."""
    if not settings or not settings.get("enabled", False):
        return None
    return tuple(sorted((key, float(value)) for key, value in settings.items() if key != "enabled"))


def _clearance_cost_fn(warehouse_map, inflation: float,
                       settings: dict[str, Any]) -> Callable[[float, float], float]:
    query = getattr(warehouse_map, "planning_clearance", None)
    if query is None:
        def query(x: float, y: float) -> float:
            return warehouse_map.clearance(x, y, 0.0)
    params = {
        "preferred_clearance": float(settings.get("preferred_clearance", 0.0)),
        "clearance_cost_weight": float(settings.get("clearance_cost_weight", 0.0)),
        "wall_avoid_clearance": float(settings.get("wall_avoid_clearance", 0.0)),
        "wall_avoid_cost_weight": float(settings.get("wall_avoid_cost_weight", 0.0)),
        "wall_avoid_min_margin": float(settings.get("wall_avoid_min_margin", 0.05)),
    }
    return lambda x, y: clearance_preference_cost(query(x, y), inflation=inflation, **params)


def astar(warehouse_map: WarehouseMap, start: tuple[float, float], goal: tuple[float, float],
          resolution: float = 0.2, inflation: float = 0.4,
          clearance: dict[str, Any] | None = None) -> list[tuple[float, float]]:
    """Grid A* with inflation; ``clearance`` adds the opticore-amr wall-distance cost.

    With ``clearance`` disabled (default) this is the Phase 0/M1 shortest-path
    planner. Enabled, every step costs ``step * (1 + clearance_cost)`` like the
    ROS planner, and line-of-sight simplification keeps at least
    ``simplify_min_clearance`` from obstacles so shortcuts do not undo it.
    """
    cols, rows = int(math.ceil(warehouse_map.width / resolution)), int(math.ceil(warehouse_map.height / resolution))
    source = warehouse_map.world_to_grid(*start, resolution)
    target = warehouse_map.world_to_grid(*goal, resolution)
    if warehouse_map.occupied(*start, inflation) or warehouse_map.occupied(*goal, inflation):
        return []
    extra = (_clearance_cost_fn(warehouse_map, inflation, clearance)
             if clearance and clearance.get("enabled", False) else None)
    frontier: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {source: None}
    cost = {source: 0.0}
    neighbors = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2.0)), (1, -1, math.sqrt(2.0)),
                 (-1, 1, math.sqrt(2.0)), (-1, -1, math.sqrt(2.0))]
    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target:
            break
        for dc, dr, step_cost in neighbors:
            nxt = current[0] + dc, current[1] + dr
            if not (0 <= nxt[0] < cols and 0 <= nxt[1] < rows):
                continue
            wx, wy = warehouse_map.grid_to_world(*nxt, resolution)
            if warehouse_map.occupied(wx, wy, inflation):
                continue
            if extra is None:
                tentative = cost[current] + step_cost
            else:
                tentative = cost[current] + step_cost * (1.0 + extra(wx, wy))
            if tentative >= cost.get(nxt, math.inf):
                continue
            cost[nxt] = tentative
            heuristic = math.hypot(nxt[0] - target[0], nxt[1] - target[1])
            heapq.heappush(frontier, (tentative + heuristic, nxt))
            came_from[nxt] = current
    if target not in came_from:
        return []
    cells, cursor = [], target
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    points = [warehouse_map.grid_to_world(*cell, resolution) for cell in reversed(cells)]
    points[0], points[-1] = start, goal
    visibility = inflation
    if extra is not None:
        visibility = max(inflation, float(clearance.get("simplify_min_clearance", inflation)))
    return simplify_path(points, warehouse_map, visibility)


def simplify_path(path: list[tuple[float, float]], warehouse_map: WarehouseMap,
                  inflation: float) -> list[tuple[float, float]]:
    if len(path) < 3:
        return path
    result, anchor = [path[0]], 0
    while anchor < len(path) - 1:
        candidate = len(path) - 1
        while candidate > anchor + 1 and not _visible(path[anchor], path[candidate], warehouse_map, inflation):
            candidate -= 1
        result.append(path[candidate])
        anchor = candidate
    return result


def _visible(a: tuple[float, float], b: tuple[float, float], warehouse_map: WarehouseMap,
             inflation: float) -> bool:
    distance = math.dist(a, b)
    count = max(2, int(distance / 0.08))
    return all(not warehouse_map.occupied(a[0] + (b[0] - a[0]) * i / count,
                                            a[1] + (b[1] - a[1]) * i / count, inflation)
               for i in range(count + 1))
