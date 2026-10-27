"""Deterministic M2 obstacle geometry and simulation-time motion in the map frame."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import json
import math
import re
import subprocess

MAP_IN_WORLD = (3.0, 15.0)
MODEL = "m2_mission_obstacle"


def world_xy(map_xy):
    return tuple(a + b for a, b in zip(map_xy, MAP_IN_WORLD))


@dataclass(frozen=True)
class Actor:
    size: tuple[float, float, float]
    # (elapsed sim seconds since goal, map x, map y), linear between knots
    knots: tuple[tuple[float, float, float], ...]
    remove_at: float | None

    def __post_init__(self):
        if len(self.size) != 3 or not all(math.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("box size must contain three positive finite values")
        if not self.knots or any(len(k) != 3 or not all(map(math.isfinite, k)) for k in self.knots):
            raise ValueError("invalid time/position knots")
        times = [k[0] for k in self.knots]
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("times must start at zero and increase")
        if self.remove_at is not None and (not math.isfinite(self.remove_at) or self.remove_at <= times[-1]):
            raise ValueError("removal must follow the last knot")

    @classmethod
    def from_dict(cls, spec):
        return cls(tuple(spec["size"]), tuple(map(tuple, spec["knots"])), spec.get("remove_at"))

    def position(self, elapsed):
        if self.remove_at is not None and elapsed >= self.remove_at:
            return None
        i = max(0, bisect_right([k[0] for k in self.knots], elapsed) - 1)
        t, x, y = self.knots[i]
        if i == len(self.knots) - 1 or elapsed <= t:
            return x, y
        end, u, v = self.knots[i + 1]
        f = (elapsed - t) / (end - t)
        return x + f * (u - x), y + f * (v - y)

    def sdf(self):
        x, y = world_xy(self.position(0))
        size = " ".join(str(v) for v in self.size)
        geom = f"<geometry><box><size>{size}</size></box></geometry>"
        return (f'<sdf version="1.8"><model name="{MODEL}"><static>true</static>'
                f'<pose>{x} {y} {self.size[2] / 2} 0 0 0</pose><link name="body">'
                f'<collision name="solid">{geom}</collision><visual name="visible">{geom}</visual>'
                '</link></model></sdf>')


def ignition_service(name, kind, request):
    result = subprocess.run([
        "ign", "service", "-s", "/world/warehouse/" + name,
        "--reqtype", "ignition.msgs." + kind, "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000", "--req", request,
    ], capture_output=True, text=True, timeout=5, check=True)
    if not re.search(r'\bdata:\s*true\b', result.stdout):
        raise RuntimeError(f"Gazebo {name} rejected request: {result.stdout} {result.stderr}")


class GazeboActor:
    """All mutations target the experiment's uniquely partitioned world."""
    def __init__(self, actor):
        self.actor = actor
        self.last = actor.position(0)
        ignition_service("create", "EntityFactory", "sdf: " + json.dumps(actor.sdf()))

    def update(self, elapsed):
        pos = self.actor.position(elapsed)
        if pos == self.last:
            return None
        if pos is None:
            ignition_service("remove", "Entity", f'name: "{MODEL}" type: MODEL')
        else:
            x, y = world_xy(pos)
            z = self.actor.size[2] / 2
            ignition_service("set_pose", "Pose", f'name: "{MODEL}" position: {{x:{x} y:{y} z:{z}}} orientation: {{w:1}}')
        self.last = pos
        return {"elapsed": elapsed, "map_xy": pos, "action": "remove" if pos is None else "set_pose"}
