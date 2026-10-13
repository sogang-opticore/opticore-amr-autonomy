"""Paired scenario legs sent as consecutive goals (Gazebo side of ROADMAP M6).

The paired spec (``packages/amr_lite/configs/paired/<set>.yaml``) lists goals
in the ROS ``map`` frame; Lite runs the same legs as ``WH-<set>-<NN>``. A
failed leg aborts the rest of the run by default, because later legs would
no longer start where their Lite counterparts start.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Goal:
    x: float
    y: float
    yaw: float
    scenario_id: str
    run: int
    leg: int


class GoalSequence:
    def __init__(self, spec: dict[str, Any], runs: int = 1, settle_sec: float = 2.0,
                 on_failure: str = "abort_run"):
        if on_failure not in {"abort_run", "continue"}:
            raise ValueError("on_failure must be abort_run or continue")
        self.name, self.legs = str(spec["name"]), list(spec["legs"])
        self.runs, self.settle_sec, self.on_failure = int(runs), float(settle_sec), on_failure
        self.timeout_sec = float(spec.get("goal_timeout_sec", 120.0))
        self.run, self.leg = 0, 0
        self.waiting = False
        self.ready_at = 0.0
        self.skipped: list[dict[str, Any]] = []

    @property
    def done(self) -> bool:
        return self.run >= self.runs

    def scenario_id(self, leg: int) -> str:
        return f"WH-{self.name}-{leg:02d}"

    def next_goal(self, now: float) -> Goal | None:
        """The goal to publish now, or None while waiting/settling/finished."""
        if self.done or self.waiting or now < self.ready_at:
            return None
        goal = self.legs[self.leg]["goal"]
        self.waiting = True
        return Goal(float(goal["x"]), float(goal["y"]), float(goal.get("yaw", 0.0)),
                    self.scenario_id(self.leg), self.run, self.leg)

    def on_result(self, now: float, success: bool) -> None:
        if not self.waiting:
            return
        self.waiting, self.ready_at = False, now + self.settle_sec
        if not success and self.on_failure == "abort_run":
            for leg in range(self.leg + 1, len(self.legs)):
                self.skipped.append({"scenario_id": self.scenario_id(leg), "run": self.run,
                                     "failure_type": "NOT_RUN_AFTER_FAILURE"})
            self.leg = len(self.legs)
        else:
            self.leg += 1
        if self.leg >= len(self.legs):
            self.run, self.leg = self.run + 1, 0
