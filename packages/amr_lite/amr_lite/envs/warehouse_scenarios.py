"""Scenarios on the opticore-amr warehouse occupancy map.

- ``WH-RANDOM`` / ``WHD-RANDOM``: random start/goal pairs (``WHD`` adds agents
  from the procedural ``dynamic`` rules of the active split).
- ``WH-<set>-<NN>``: leg ``NN`` of a paired scenario set in
  ``configs/paired/<set>.yaml``, written in ROS ``map`` coordinates so the same
  goals can be sent to Gazebo. Leg ``k`` starts where leg ``k-1`` arrived.
"""
from __future__ import annotations

import functools
import math
import zlib
from typing import Any

import numpy as np

from amr_lite.config import load_config
from amr_lite.planning.astar import astar, clearance_key

from .dynamics import RobotState
from .grid_map import GridMap, named_map
from .procedural import agent_payload, instance_digest, path_length, place_agents, procedural_spec
from .scenarios import Scenario, WaypointObstacle


@functools.lru_cache(maxsize=16)
def paired_spec(name: str) -> dict[str, Any]:
    return load_config(f"paired/{name}")


@functools.lru_cache(maxsize=8)
def _free_cells(map_name: str, min_clearance: float) -> np.ndarray:
    return named_map(map_name).free_cells(min_clearance)


def _planner(config: dict[str, Any]) -> tuple[float, float, tuple | None]:
    return (float(config.get("grid_resolution", 0.2)),
            float(config.get("robot_radius", 0.36)) + float(config.get("obstacle_inflation", 0.04)),
            clearance_key(config.get("astar_clearance")))


def _clearance_settings(key: tuple | None) -> dict[str, Any] | None:
    return None if key is None else {"enabled": True, **dict(key)}


def _heading(path: list[tuple[float, float]] | tuple, end: bool = False) -> float:
    (ax, ay), (bx, by) = (path[-2], path[-1]) if end else (path[0], path[1])
    return math.atan2(by - ay, bx - ax)


def make_warehouse_scenario(scenario_id: str, seed: int, config: dict[str, Any]) -> Scenario:
    if scenario_id in {"WH-RANDOM", "WHD-RANDOM"}:
        return _random_pair(scenario_id, seed, config)
    body, _, leg = scenario_id[len("WH-"):].rpartition("-")
    if not scenario_id.startswith("WH-") or not body or not leg.isdigit():
        raise KeyError(f"Unknown warehouse scenario {scenario_id!r}; use WH-RANDOM, WHD-RANDOM or WH-<set>-<NN>")
    return paired_leg(body, int(leg), seed, config)


def _random_pair(scenario_id: str, seed: int, config: dict[str, Any]) -> Scenario:
    settings = config.get("warehouse") or {}
    map_name = str(settings.get("map", "opticore_warehouse"))
    grid = named_map(map_name)
    resolution, inflation, clearance = _planner(config)
    settings_clearance = _clearance_settings(clearance)
    rng = np.random.default_rng(np.random.SeedSequence(
        [int(seed) % 2**32, zlib.crc32(scenario_id.encode("utf-8")), zlib.crc32(map_name.encode("utf-8"))]))
    candidates = _free_cells(map_name, float(settings.get("min_clearance", 0.7)))
    low, high = (float(value) for value in settings.get("path_length", [6.0, 25.0]))
    for _ in range(int(settings.get("max_attempts", 40))):
        start = tuple(float(v) for v in candidates[int(rng.integers(len(candidates)))])
        goal = tuple(float(v) for v in candidates[int(rng.integers(len(candidates)))])
        if not low <= math.dist(start, goal) <= high:
            continue
        path = astar(grid, start, goal, resolution, inflation, settings_clearance)
        if not path or not low <= path_length(path) <= 1.5 * high:
            continue
        robot = RobotState(start[0], start[1], _heading(path) + float(rng.uniform(-0.6, 0.6)))
        agents = []
        if scenario_id.startswith("WHD"):
            procedural = config.get("procedural") or {}
            spec = procedural_spec(str(procedural.get("spec", "procedural")))
            rules = spec["splits"][str(procedural.get("split", "train"))]["dynamic"]
            agents = place_agents(rng, grid, path, start, rules, spec["nominal_robot_speed"],
                                  resolution, inflation, settings_clearance)
        digest = instance_digest({"scenario_id": scenario_id, "map": map_name, "seed": int(seed),
                                  "start": [round(robot.x, 6), round(robot.y, 6), round(robot.theta, 6)],
                                  "goal": [round(goal[0], 6), round(goal[1], 6)],
                                  "agents": agent_payload(agents)})
        return Scenario(scenario_id, grid, robot, goal, agents, True,
                        f"{scenario_id}-{map_name}-{int(seed)}-{digest}", tuple(path),
                        (resolution, inflation, clearance))
    raise RuntimeError(f"Could not sample {scenario_id} on {map_name} (seed={seed})")


@functools.lru_cache(maxsize=64)
def _leg_plan(set_name: str, index: int, resolution: float, inflation: float, clearance: tuple | None
              ) -> tuple[tuple[float, float, float], tuple[float, float], tuple[tuple[float, float], ...]]:
    """Start pose, goal and A* path of a paired leg, in Lite coordinates."""
    spec = paired_spec(set_name)
    grid = named_map(str(spec["map"]))
    legs = spec["legs"]
    if not 0 <= index < len(legs):
        raise KeyError(f"Paired set {set_name!r} has legs 0..{len(legs) - 1}")
    start = spec["start"]
    x, y = grid.to_lite(float(start["x"]), float(start["y"]))
    pose = (x, y, float(start.get("yaw", 0.0)))
    for number in range(index + 1):
        goal = grid.to_lite(float(legs[number]["goal"]["x"]), float(legs[number]["goal"]["y"]))
        path = astar(grid, pose[:2], goal, resolution, inflation, _clearance_settings(clearance))
        if not path:
            raise RuntimeError(f"No A* path for paired leg {set_name}-{number:02d}")
        if number == index:
            return pose, goal, tuple(path)
        pose = (goal[0], goal[1], _heading(path, end=True))
    raise AssertionError("unreachable")


def paired_agents(set_name: str, grid: GridMap, seed: int) -> list[WaypointObstacle]:
    agents = []
    for item in paired_spec(set_name).get("agents") or []:
        waypoints = tuple(grid.to_lite(float(x), float(y)) for x, y in item["waypoints"])
        agents.append(WaypointObstacle(
            waypoints[0][0], waypoints[0][1], 0.0, 0.0, radius=float(item.get("radius", 0.3)),
            motion_type="waypoint", start_delay=float(item.get("start_delay", 0.0)),
            waypoints=waypoints, speed=float(item["speed"]),
            pause_range=tuple(float(v) for v in item.get("pause", [0.0, 0.0])),
            pause_seed=zlib.crc32(f"{item['name']}:{int(seed)}".encode("utf-8"))))
    return agents


def paired_leg(set_name: str, index: int, seed: int, config: dict[str, Any]) -> Scenario:
    resolution, inflation, clearance = _planner(config)
    pose, goal, path = _leg_plan(set_name, index, resolution, inflation, clearance)
    spec = paired_spec(set_name)
    grid = named_map(str(spec["map"]))
    agents = paired_agents(set_name, grid, seed)
    scenario_id = f"WH-{set_name}-{index:02d}"
    digest = instance_digest({"scenario_id": scenario_id, "seed": int(seed),
                              "start": [round(v, 6) for v in pose], "goal": [round(v, 6) for v in goal],
                              "agents": agent_payload(agents)})
    return Scenario(scenario_id, grid, RobotState(*pose), goal, agents, True,
                    f"{scenario_id}-{int(seed)}-{digest}", path, (resolution, inflation, clearance))
