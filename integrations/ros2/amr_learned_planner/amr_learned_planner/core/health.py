"""Health faults of the learned planner (ARCHITECTURE §10).

Permanent faults (bundle/contract/profile problems) keep the planner out of
control until it is restarted with a valid bundle; transient faults clear
themselves when the condition goes away.
"""
from __future__ import annotations

from typing import Any

PERMANENT_FAULTS = frozenset({"BUNDLE_LOAD_FAILED", "CONTRACT_MISMATCH", "PROFILE_MISMATCH",
                              "GOLDEN_PARITY_FAILED"})


class HealthMonitor:
    def __init__(self, deadline_miss_limit: int = 3):
        self.deadline_miss_limit = max(1, int(deadline_miss_limit))
        self.permanent: dict[str, str] = {}
        self.transient: dict[str, str] = {}
        self.consecutive_misses = 0
        self.total_misses = 0
        self.inferences = 0
        self.last_inference_ms = 0.0
        self.max_inference_ms = 0.0

    def set_permanent(self, code: str, detail: str) -> None:
        self.permanent[code] = detail

    def set_transient(self, code: str, detail: str) -> None:
        self.transient[code] = detail

    def clear(self, code: str) -> None:
        self.transient.pop(code, None)

    def record_inference(self, milliseconds: float, missed: bool) -> None:
        self.inferences += 1
        self.last_inference_ms = float(milliseconds)
        self.max_inference_ms = max(self.max_inference_ms, float(milliseconds))
        if missed:
            self.consecutive_misses += 1
            self.total_misses += 1
        else:
            self.consecutive_misses = 0
        if self.consecutive_misses >= self.deadline_miss_limit:
            self.set_transient("DEADLINE_MISS", f"{self.consecutive_misses} consecutive deadline misses")
        else:
            self.clear("DEADLINE_MISS")

    @property
    def healthy(self) -> bool:
        return not self.permanent and not self.transient

    def report(self) -> dict[str, Any]:
        details = {**self.permanent, **self.transient}
        return {
            "healthy": self.healthy,
            "faults": sorted(details),
            "details": details,
            "inference_ms": self.last_inference_ms,
            "max_inference_ms": self.max_inference_ms,
            "inferences": self.inferences,
            "deadline_misses": self.total_misses,
            "deadline_miss_rate": self.total_misses / max(1, self.inferences),
        }
