from __future__ import annotations

from dataclasses import dataclass

from .dynamics import RobotState
from .geometry import Rectangle
from .map import WarehouseMap


@dataclass
class DynamicObstacle:
    x: float
    y: float
    vx: float
    vy: float
    radius: float = 0.28
    motion_type: str = "cross"
    start_delay: float = 0.0
    stop_time: float | None = None

    def update(self, time: float, dt: float) -> None:
        if time < self.start_delay or (self.stop_time is not None and time >= self.stop_time):
            return
        self.x += self.vx * dt
        self.y += self.vy * dt


@dataclass
class Scenario:
    scenario_id: str
    warehouse_map: WarehouseMap
    start: RobotState
    goal: tuple[float, float]
    dynamic_obstacles: list[DynamicObstacle]
    solvable: bool = True


def make_scenario(scenario_id: str, seed: int = 0) -> Scenario:
    del seed  # Reserved for randomized scenario families.
    base = WarehouseMap()
    specs: dict[str, tuple[list[Rectangle], RobotState, tuple[float, float], bool]] = {
        "S0": ([], RobotState(1.0, 4.0, 0.0), (11.0, 4.0), True),
        "S1": ([Rectangle(5.0, 0.0, 5.5, 5.7)], RobotState(1.0, 2.0, 0.0), (9.5, 6.5), True),
        "S2": ([Rectangle(3.0, 0.0, 9.0, 3.25), Rectangle(3.0, 4.75, 9.0, 8.0)], RobotState(1.0, 4.0, 0.0), (11.0, 4.0), True),
        "S3": ([Rectangle(5.3, 3.2, 6.2, 4.8)], RobotState(1.0, 4.0, 0.0), (11.0, 4.0), True),
        "S4": ([Rectangle(5.0, 4.55, 7.0, 6.2)], RobotState(1.0, 4.0, 0.0), (11.0, 4.0), True),
        "S5": ([Rectangle(4.0, 5.0, 8.0, 5.6)], RobotState(1.0, 6.5, -0.4), (11.0, 4.0), True),
        "S6": ([Rectangle(4.8, 0.0, 5.5, 5.9), Rectangle(7.5, 2.1, 8.2, 8.0)], RobotState(1.0, 4.0, 0.0), (11.0, 4.0), True),
        "S7": ([Rectangle(4.0, 1.0, 4.6, 7.0), Rectangle(0.0, 1.0, 4.0, 1.6), Rectangle(0.0, 6.4, 4.0, 7.0)], RobotState(2.0, 4.0, 0.0), (10.0, 4.0), False),
    }
    static_id = scenario_id if scenario_id in specs else "S0"
    obstacles, start, goal, solvable = specs[static_id]
    dynamics: list[DynamicObstacle] = []
    if scenario_id == "D0":
        dynamics = [DynamicObstacle(6.0, 2.0, 0.0, 0.7, motion_type="cross")]
    elif scenario_id == "D1":
        dynamics = [DynamicObstacle(9.0, 4.0, -0.45, 0.0, motion_type="approach")]
    elif scenario_id == "D2":
        dynamics = [DynamicObstacle(4.0, 4.0, 0.25, 0.0, motion_type="same_direction")]
    elif scenario_id == "D3":
        dynamics = [DynamicObstacle(6.0, 4.0, 0.0, 0.0, motion_type="stopped")]
    elif scenario_id == "D4":
        dynamics = [DynamicObstacle(6.0, 2.0, 0.0, 0.7, motion_type="stop", stop_time=3.0)]
    return Scenario(scenario_id, WarehouseMap(obstacles=list(obstacles)), start, goal, dynamics, solvable)

