"""Post-M1 train-only warehouse scenes and path-progress reward for M4."""
from __future__ import annotations

from functools import lru_cache
import json
import math

from amr_lite.config import ROOT
from amr_lite.envs.grid_map import named_map
from amr_lite.envs.m3_warehouse import (
    MAP_ASSET, M3WarehouseEnv, _ScanGeometry, development_missions, environment_config,
)
from amr_lite.experiments.manifest import sha256_file
from amr_lite.planning.astar import astar
from amr_lite.planning.path_utils import nearest_segment


CURRICULUM_MANIFEST = ROOT.parents[1] / "experiments/m4/curriculum.json"


@lru_cache(maxsize=1)
def _curriculum():
    spec = json.loads(CURRICULUM_MANIFEST.read_text())
    if (spec["version"], spec["frame"], spec["map_sha256"]) != (1, "map", sha256_file(MAP_ASSET)):
        raise ValueError("curriculum coordinate frame, schema or warehouse map mismatch")
    scenes = set(spec["scenes"])
    allowed = scenes | set(development_missions())
    assigned = {name for stage in spec["stages"] for name in stage}
    if not all(name.startswith("m4-train-") for name in scenes) or not scenes <= assigned <= allowed:
        raise ValueError("curriculum may only use registered train missions")
    return spec


def curriculum_stages():
    """Return ordered mission IDs for each of the three training stages."""
    return tuple(tuple(stage) for stage in _curriculum()["stages"])


@lru_cache(maxsize=32)
def _planned_segment(start, goal):
    grid = named_map("opticore_warehouse")
    config = environment_config()
    path = astar(grid, grid.to_lite(*start), grid.to_lite(*goal), config["grid_resolution"],
                 config["robot_radius"] + config["obstacle_inflation"], config["astar_clearance"])
    if len(path) < 2:
        raise ValueError(f"no warehouse A* route between {start} and {goal}")
    return tuple(path)


class _SceneGeometry:
    """Compose all actor boxes on a single static LiDAR raycast and clearance query."""

    def __init__(self, grid, schedules):
        self.grid = grid
        self.actors = tuple(_ScanGeometry(grid, schedule) for schedule in schedules)

    def set_time(self, elapsed):
        for actor in self.actors:
            actor.set_time(elapsed)

    def clearance(self, x, y, radius):
        distance = self.grid.clearance(x, y, radius)
        for actor in self.actors:
            distance = min(distance, actor.actor_clearance(x, y, radius))
        return distance

    def ray_distances(self, ox, oy, dx, dy, max_range):
        distances = self.grid.ray_distances(ox, oy, dx, dy, max_range)
        for actor in self.actors:
            distances = actor.clip_actor_rays(distances, ox, oy, dx, dy, max_range)
        return distances


class M4WarehouseEnv(M3WarehouseEnv):
    """M3 actuator, physics and arrival criteria with a path-distance correction."""

    def __init__(self, mission_id, dynamics=None, *, history=True, shield=True):
        self.history = history
        super().__init__(mission_id, dynamics)
        self.config["shield_enabled"] = shield

    def _configure_mission(self, mission_id):
        self.scene = _curriculum()["scenes"].get(mission_id)
        self.route_update_at = None
        self.block_removed_at = None
        if self.scene is None:
            # Preserve the original train geometry, deadline and start heading.
            super()._configure_mission(mission_id)
            if self.geometry.actor:
                self.block_removed_at = self.geometry.actor.get("remove_at")
            return

        scene = self.scene
        grid = named_map("opticore_warehouse")
        start, goal = tuple(scene["start"]), tuple(scene["goal"])
        self.start, self.goal = grid.to_lite(*start), grid.to_lite(*goal)
        self.start_yaw, self.timeout = float(scene["yaw"]), float(scene["timeout"])
        self.geometry = _SceneGeometry(grid, scene["actors"])
        self.block_removed_at = next((a["remove_at"] for a in scene["actors"] if "remove_at" in a), None)
        if "initial_via" in scene:
            via = tuple(scene["initial_via"])
            self.initial_path = _planned_segment(start, via) + _planned_segment(via, goal)[1:]
            self.replacement_path = _planned_segment(tuple(scene["replacement_from"]), goal)
            self.route_update_at = scene["route_update_at"]
        else:
            self.initial_path = _planned_segment(start, goal)
        self.path = self.initial_path

    def _remaining_distance(self, state):
        # At/after the endpoint the projection term alone vanishes. This term
        # still measures approaching or overshooting the physical goal.
        return super()._remaining_distance(state) + math.dist((self.robot.x, self.robot.y), state.projection)

    def _policy_input(self, observation):
        if self.history:
            return observation
        from amr_lite.learning.observation import repeat_latest_scan
        return repeat_latest_scan(observation)

    def reset(self, *, seed=None, options=None):
        if self.scene is not None:
            self.path = self.initial_path
        self.replacement_active = False
        self.route_rejoin_s = None
        self.was_off_replacement = False
        observation, info = super().reset(seed=seed, options=options)
        return self._policy_input(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if self.route_update_at is not None and info["disposition"] != "invalid":
            if not self.replacement_active and self.elapsed + 1e-9 >= self.route_update_at:
                self.path = self.replacement_path
                self.replacement_active = True
                sample = self.io.observe(self.clock, self.clock, self.robot, self.path)
                if sample.tensor is None:
                    raise RuntimeError(f"unavailable replacement path observation: {sample.fault}")
                # The transition up to this tick used the old path. Rebase its
                # successor potential so replacing a path never creates reward.
                self.remaining = self._remaining_distance(sample.path)
                self.last_observation = sample.tensor
                observation = sample.tensor.copy()
            if self.replacement_active and self.route_rejoin_s is None:
                _, x, y, _ = nearest_segment(self.path, self.robot.x, self.robot.y)
                error = math.hypot(self.robot.x - x, self.robot.y - y)
                if error > self.scene["rejoin_tolerance"]:
                    self.was_off_replacement = True
                elif self.was_off_replacement:
                    self.route_rejoin_s = self.elapsed - self.route_update_at
            info.update(route_rejoined=self.route_rejoin_s is not None, route_rejoin_s=self.route_rejoin_s)
        return self._policy_input(observation), reward, terminated, truncated, info

    def _info(self, **extras):
        return super()._info(block_removed_at=self.block_removed_at,
                             route_rejoined=None if self.route_update_at is None else self.route_rejoin_s is not None,
                             route_rejoin_s=self.route_rejoin_s, **extras)
