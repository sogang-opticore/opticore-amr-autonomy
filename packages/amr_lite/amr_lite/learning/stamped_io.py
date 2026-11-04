"""Simulation-time sensor history and direct actuation shared by Lite and ROS."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from amr_lite.envs.dynamics import RobotState, limit_velocity
from amr_lite.learning.action import scale_action
from amr_lite.learning.observation import build_observation
from amr_lite.planning.path_utils import PathState, path_state
from amr_lite.sensors.lidar import ContractScanSpec, RawLidarSpec, resample_scan

M3_IO_ID = "post-m1-policy-io-v1"
_CLOCK_RESOLUTION = 1e-9


@dataclass(frozen=True)
class ObservationSample:
    tensor: np.ndarray | None
    stamps: tuple[float, ...]
    fault: str | None = None
    path: PathState | None = None


@dataclass(frozen=True)
class CommandSample:
    target: tuple[float, float] | None
    command: tuple[float, float]
    fault: str | None = None


class StampedPolicyIO:
    """Consume sensor acquisitions once and assemble the versioned M2 policy input."""

    def __init__(self):
        self.contract_scan = ContractScanSpec()
        self.raw_scan = RawLidarSpec()
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=4)
        self.reset()

    def reset(self) -> None:
        """Start a new clock/mission epoch without carrying scan identity or data."""
        self.frames.clear()
        self.scan_identity: tuple[str, str] | None = None
        self.scan_watermark: float | None = None
        self.scan_invalid = False
        self.last_tick: float | None = None

    def _start_scan(self, namespace: str, frame: str, stamp: float) -> bool:
        if not frame or not math.isfinite(stamp) or stamp < 0:
            return self._invalidate_scan()
        identity = (namespace, frame)
        if identity != self.scan_identity:
            self.frames.clear()
            self.scan_watermark = None
            self.scan_identity = identity
        if self.scan_watermark is not None and stamp <= self.scan_watermark:
            return False
        self.scan_watermark = stamp
        return True

    def _invalidate_scan(self) -> bool:
        self.frames.clear()
        self.scan_invalid = True
        return False

    def _append_scan(self, stamp: float, scan: np.ndarray) -> bool:
        if self.frames and stamp - self.frames[-1][0] > 0.5 + _CLOCK_RESOLUTION:
            # Detect a sensor outage even when no control tick ran during it.
            self.frames.clear()
        self.frames.append((stamp, scan))
        self.scan_invalid = False
        return True

    def ingest_scan(self, namespace: str, frame: str, stamp: float, ranges: Sequence[float],
                    angle_min: float, angle_increment: float, range_min: float, range_max: float) -> bool:
        """Validate a raw acquisition before filling empty contract bins."""
        if not self._start_scan(namespace, frame, stamp):
            return False
        values = np.asarray(ranges, dtype=np.float64)
        metadata = (angle_min, angle_increment, range_min, range_max)
        if (values.ndim != 1 or not values.size or not all(map(math.isfinite, metadata))
                or angle_increment == 0 or abs(angle_increment) * (values.size - 1) > math.tau + 0.02
                or not 0 <= range_min < range_max):
            return self._invalidate_scan()
        hits = (values >= range_min) & (values <= range_max)
        if not np.any(hits | np.isposinf(values)):
            return self._invalidate_scan()
        mount = self.raw_scan
        scan = resample_scan(values, angle_min, angle_increment, range_min, range_max,
                             mount_x=mount.mount_x, mount_y=mount.mount_y, mount_yaw=mount.mount_yaw,
                             contract=self.contract_scan)
        return self._append_scan(stamp, scan)

    def ingest_bins(self, namespace: str, frame: str, stamp: float, scan: np.ndarray) -> bool:
        """Accept verified contract bins from the synthetic side of the sensor boundary."""
        if not self._start_scan(namespace, frame, stamp):
            return False
        scan = np.asarray(scan, dtype=np.float32)
        if (scan.shape != (self.contract_scan.rays,) or not np.isfinite(scan).all()
                or np.any(scan < 0) or np.any(scan > self.contract_scan.max_range)):
            return self._invalidate_scan()
        return self._append_scan(stamp, scan.copy())

    def _unavailable(self, fault: str, *, clear_history: bool = False) -> ObservationSample:
        if clear_history:
            self.frames.clear()
        return ObservationSample(None, tuple(stamp for stamp, _ in self.frames), fault)

    def observe(self, now: float, odom_stamp: float, robot: RobotState,
                path: Sequence[tuple[float, float]], *,
                segment_range: tuple[int, int] | None = None) -> ObservationSample:
        if not math.isfinite(now) or now < 0:
            self.reset()
            return self._unavailable("CLOCK_INVALID")
        if self.last_tick is not None and now < self.last_tick - _CLOCK_RESOLUTION:
            self.reset()
            self.last_tick = now
            return self._unavailable("CLOCK_RESET")
        self.last_tick = now
        if not math.isfinite(odom_stamp) or odom_stamp < 0 or not all(map(
                math.isfinite, (robot.x, robot.y, robot.theta, robot.v, robot.w))):
            return self._unavailable("ODOM_INVALID", clear_history=True)
        if odom_stamp > now + _CLOCK_RESOLUTION:
            return self._unavailable("ODOM_FUTURE", clear_history=True)
        if now - odom_stamp > 0.5 + _CLOCK_RESOLUTION:
            return self._unavailable("ODOM_STALE", clear_history=True)
        if self.scan_invalid:
            return self._unavailable("SCAN_INVALID")
        if self.frames:
            if self.frames[-1][0] > now + _CLOCK_RESOLUTION:
                self.scan_watermark = None
                return self._unavailable("SCAN_FUTURE", clear_history=True)
            if now - self.frames[-1][0] > 0.5 + _CLOCK_RESOLUTION:
                return self._unavailable("SCAN_STALE", clear_history=True)
        if len(self.frames) < self.frames.maxlen:
            return self._unavailable("SCAN_WARMUP")
        if len(path) < 2 or not all(math.isfinite(x) and math.isfinite(y) for x, y in path):
            return self._unavailable("PATH_UNAVAILABLE")
        state = path_state(path, robot, 0.65, segment_range=segment_range)
        stamps, frames = zip(*self.frames)
        vector = build_observation(frames, state, robot, self.contract_scan.max_range, 8.0, 1.5, 1.5, 2.0)
        return ObservationSample(vector, stamps, path=state)

    @staticmethod
    def actuate(action: Sequence[float] | np.ndarray, robot: RobotState, *,
                inference_ms: float = 0.0, deadline_ms: float = 25.0) -> CommandSample:
        """Translate normalized actions and apply identical physical command limits."""
        stop = (0.0, 0.0)
        if (not math.isfinite(inference_ms) or not math.isfinite(deadline_ms)
                or not 0 <= inference_ms <= deadline_ms or deadline_ms <= 0):
            return CommandSample(None, stop, "INFERENCE_DEADLINE")
        try:
            action = np.asarray(action, dtype=np.float32)
        except (TypeError, ValueError, OverflowError):
            return CommandSample(None, stop, "INVALID_ACTION")
        if action.shape != (2,) or not np.isfinite(action).all():
            return CommandSample(None, stop, "INVALID_ACTION")
        if not all(map(math.isfinite, (robot.v, robot.w))):
            return CommandSample(None, stop, "ODOM_INVALID")
        target = scale_action(action, 1.5, 1.5)
        command = limit_velocity(robot.v, robot.w, *target, 0.05, 1.5, 1.5, 1.0, 2.0, 1.5)
        return CommandSample(target, command)
