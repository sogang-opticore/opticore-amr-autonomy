"""Command arbiter: the single publisher of ``cmd_vel`` and the effective ``dwa/status``.

Modes (ARCHITECTURE §9):

- ``dwa``: forward the DWA (rollback / unchanged behaviour)
- ``shadow``: forward the DWA; the learned planner only runs and is recorded
- ``learned``: forward the learned planner; on a fault or stale output hold a
  stop for ``fallback_hold_sec``, then forward the DWA. Return to the learned
  planner only after it has been healthy for ``recover_after_sec`` and at
  least ``min_fallback_sec`` has passed, if ``allow_recovery`` is set.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

MODES = ("dwa", "shadow", "learned")


@dataclass
class ArbiterConfig:
    mode: str = "shadow"
    cmd_timeout: float = 0.25
    health_timeout: float = 0.5
    fallback_hold_sec: float = 0.3
    min_fallback_sec: float = 2.0
    recover_after_sec: float = 2.0
    allow_recovery: bool = True


@dataclass
class Source:
    cmd: tuple[float, float] = (0.0, 0.0)
    cmd_stamp: float = -math.inf
    status: str = ""
    status_stamp: float = -math.inf

    def fresh(self, now: float, timeout: float) -> bool:
        return now - self.cmd_stamp <= timeout


@dataclass
class ArbiterOutput:
    cmd: tuple[float, float] | None = None       # None: nothing to publish now
    statuses: list[str] = field(default_factory=list)
    transition: dict[str, Any] | None = None


class CommandArbiter:
    def __init__(self, config: ArbiterConfig):
        if config.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {config.mode!r}")
        self.config = config
        self.learned, self.dwa = Source(), Source()
        self.learned_healthy = False
        self.learned_faults: list[str] = []
        self.health_stamp = -math.inf
        self.healthy_since: float | None = None
        # learned mode starts on the DWA until the learned planner reports healthy.
        self.state = "DWA" if config.mode in {"dwa", "shadow"} else "FALLBACK"
        self.state_since = 0.0
        self.reason = "start" if config.mode != "learned" else "startup: waiting for a healthy learned planner"
        self.transitions = 0
        self.fallbacks = 0
        self.stopped_published = False

    # ------------------------------------------------------------- helpers
    @property
    def active(self) -> str:
        return {"DWA": "dwa", "LEARNED": "learned", "FALLBACK": "dwa", "FALLBACK_HOLD": "stop"}[self.state]

    def _source(self, name: str) -> Source:
        return self.learned if name == "learned" else self.dwa

    def _learned_ok(self, now: float) -> tuple[bool, str]:
        if now - self.health_stamp > self.config.health_timeout:
            return False, "learned health stale"
        if not self.learned_healthy:
            return False, "learned faults: " + ",".join(self.learned_faults or ["unknown"])
        if not self.learned.fresh(now, self.config.cmd_timeout):
            return False, "learned cmd stale"
        return True, ""

    def _enter(self, state: str, now: float, reason: str, output: ArbiterOutput) -> None:
        previous, self.state, self.state_since, self.reason = self.state, state, now, reason
        self.transitions += 1
        self.stopped_published = False
        if state == "FALLBACK_HOLD":
            self.fallbacks += 1
        output.transition = {"from": previous, "to": state, "reason": reason, "time": now}
        active = self.active
        if active in {"dwa", "learned"}:
            source = self._source(active)
            if source.status:
                output.statuses.append(source.status)
            if source.fresh(now, self.config.cmd_timeout):
                output.cmd = source.cmd  # hand over without waiting for the next message

    # -------------------------------------------------------------- inputs
    def on_cmd(self, source: str, now: float, v: float, w: float) -> ArbiterOutput:
        if not (math.isfinite(v) and math.isfinite(w)):
            return ArbiterOutput()  # treated as missing: the source goes stale
        target = self._source(source)
        target.cmd, target.cmd_stamp = (float(v), float(w)), now
        return ArbiterOutput(cmd=target.cmd) if self.active == source else ArbiterOutput()

    def on_status(self, source: str, now: float, status: str) -> ArbiterOutput:
        target = self._source(source)
        target.status, target.status_stamp = status, now
        return ArbiterOutput(statuses=[status] if self.active == source else [])

    def on_health(self, now: float, healthy: bool, faults: list[str] | None = None) -> None:
        self.learned_healthy, self.learned_faults, self.health_stamp = bool(healthy), list(faults or []), now

    # ---------------------------------------------------------------- tick
    def tick(self, now: float) -> ArbiterOutput:
        output = ArbiterOutput()
        config = self.config
        ok, why = self._learned_ok(now)
        if ok:
            self.healthy_since = now if self.healthy_since is None else self.healthy_since
        else:
            self.healthy_since = None
        if config.mode == "learned":
            if self.state == "LEARNED" and not ok:
                self._enter("FALLBACK_HOLD", now, why, output)
            elif self.state == "FALLBACK_HOLD" and now - self.state_since >= config.fallback_hold_sec:
                self._enter("FALLBACK", now, "hold complete", output)
            elif (self.state == "FALLBACK" and config.allow_recovery and ok
                  and now - self.state_since >= config.min_fallback_sec
                  and self.healthy_since is not None and now - self.healthy_since >= config.recover_after_sec):
                self._enter("LEARNED", now, "learned planner healthy again", output)
        active = self.active
        if active == "stop":
            output.cmd = (0.0, 0.0)
        elif output.transition is not None and output.cmd is not None:
            pass  # the handover already forwarded the new source's latest command
        elif not self._source(active).fresh(now, config.cmd_timeout):
            # The forwarded planner went silent: stop the robot and say so once.
            output.cmd = (0.0, 0.0)
            if not self.stopped_published:
                output.statuses.append("STOPPED")
                self.stopped_published = True
        else:
            self.stopped_published = False
        return output

    def snapshot(self, now: float) -> dict[str, Any]:
        return {"mode": self.config.mode, "state": self.state, "active": self.active, "reason": self.reason,
                "since": self.state_since, "transitions": self.transitions, "fallbacks": self.fallbacks,
                "learned_healthy": self.learned_healthy, "learned_faults": self.learned_faults,
                "learned_cmd_age": now - self.learned.cmd_stamp, "dwa_cmd_age": now - self.dwa.cmd_stamp}
