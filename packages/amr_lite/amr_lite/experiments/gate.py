"""Pre-registered G4 gate: is a candidate policy family fit for ROS integration?"""
from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from amr_lite.config import load_config


def _rate(rows: list[dict], predicate) -> float:
    return sum(1 for row in rows if predicate(row)) / len(rows) if rows else float("nan")


def _flag(row: dict, key: str) -> bool:
    return str(row[key]).lower() in {"1", "true"}


def _check(name: str, value: float, threshold: float, passed: bool, note: str = "") -> dict[str, Any]:
    return {"name": name, "value": value, "threshold": threshold, "passed": bool(passed), "note": note}


def evaluate_gate(rows: list[dict], comparisons: list[dict], family: str, role: str = "best",
                  criteria: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the G4 criteria to test-split ``episodes.csv`` rows and paired comparisons."""
    criteria = criteria or load_config("gates/g4")
    reference = criteria["reference"]

    def select(policy: str, checkpoint_role: str | None, shield: bool) -> list[dict]:
        return [row for row in rows if str(row.get("policy_family", row["policy"])) == policy
                and (checkpoint_role is None or str(row.get("checkpoint_role", "fixed")) == checkpoint_role)
                and _flag(row, "shield_enabled") == shield]

    candidate, baseline = select(family, role, False), select(reference, None, False)
    if not candidate or not baseline:
        raise ValueError(f"Missing shield-off rows for {family!r}/{role!r} or reference {reference!r}")
    checks = []
    paired = next((item for item in comparisons if item["comparison"] == "policy_minus_rule"
                   and item["policy"] == family and item["checkpoint_role"] == role
                   and not _flag(item, "shield_enabled")), None)
    rules = criteria["non_inferiority"]
    if paired is None:
        checks.append(_check("non_inferiority", float("nan"), float("nan"), False, "no paired comparison rows"))
    else:
        low, high = float(paired["success_delta_ci_low"]), float(paired["collision_delta_ci_high"])
        checks.append(_check("success_delta_ci_low", low, rules["success_delta_ci_low_min"],
                             low >= float(rules["success_delta_ci_low_min"])))
        checks.append(_check("collision_delta_ci_high", high, rules["collision_delta_ci_high_max"],
                             high <= float(rules["collision_delta_ci_high_max"])))
    improvement = criteria["improvement"]
    dynamic = set(improvement["dynamic_scenarios"])
    candidate_dynamic = _rate([r for r in candidate if r["scenario_id"] in dynamic], lambda r: _flag(r, "success"))
    baseline_dynamic = _rate([r for r in baseline if r["scenario_id"] in dynamic], lambda r: _flag(r, "success"))
    dynamic_delta = candidate_dynamic - baseline_dynamic
    solved = {(r["scenario_id"], str(r["seed"])) for r in baseline if _flag(r, "success")}
    both = [r for r in candidate if _flag(r, "success") and (r["scenario_id"], str(r["seed"])) in solved]
    reference_times = {(r["scenario_id"], str(r["seed"])): float(r["episode_time"])
                       for r in baseline if _flag(r, "success")}
    ratio = (statistics.mean(float(r["episode_time"]) for r in both)
             / statistics.mean(reference_times[(r["scenario_id"], str(r["seed"]))] for r in both)
             if both else float("nan"))
    improved = (dynamic_delta >= float(improvement["dynamic_success_delta_min"])
                or ratio <= float(improvement["mean_time_ratio_max"]))
    checks.append(_check("improvement_dynamic_success_delta", dynamic_delta,
                         improvement["dynamic_success_delta_min"], improved,
                         f"time ratio on jointly solved episodes = {ratio:.3f} (max "
                         f"{improvement['mean_time_ratio_max']}); either criterion suffices"))
    robustness = criteria["robustness"]
    by_seed: dict[str, float] = {}
    for seed in sorted({str(r.get("training_seed", "")) for r in candidate}):
        by_seed[seed] = _rate([r for r in candidate if str(r.get("training_seed", "")) == seed],
                              lambda r: _flag(r, "success"))
    if len(by_seed) > 1:
        gap = statistics.median(by_seed.values()) - min(by_seed.values())
        checks.append(_check("worst_seed_gap", gap, robustness["worst_seed_gap_max"],
                             gap <= float(robustness["worst_seed_gap_max"]), f"per seed {by_seed}"))
    final = select(family, "final", False)
    if final and role != "final":
        gaps = []
        for seed, best_rate in by_seed.items():
            final_rows = [r for r in final if str(r.get("training_seed", "")) == seed]
            if final_rows:
                gaps.append(best_rate - _rate(final_rows, lambda r: _flag(r, "success")))
        if gaps:
            checks.append(_check("best_final_gap", max(gaps), robustness["best_final_gap_max"],
                                 max(gaps) <= float(robustness["best_final_gap_max"])))
    behaviour = criteria["behaviour"]
    for failure in behaviour["failure_types"]:
        rate = _rate(candidate, lambda r, name=failure: r["failure_type"] == name)
        checks.append(_check(f"behaviour_{failure.lower()}_rate", rate, behaviour["max_rate"],
                             rate <= float(behaviour["max_rate"])))
    shield_on = select(family, role, True)
    if shield_on:
        saturation = _rate(shield_on, lambda r: r["failure_type"] == "SHIELD_SATURATION")
        checks.append(_check("shield_on_saturation_rate", saturation, criteria["shield_on"]["saturation_rate_max"],
                             saturation <= float(criteria["shield_on"]["saturation_rate_max"])))
    return {"family": family, "role": role, "reference": reference,
            "passed": all(check["passed"] for check in checks), "checks": checks,
            "episodes": {"candidate_shield_off": len(candidate), "reference_shield_off": len(baseline),
                         "candidate_shield_on": len(shield_on)}}


def write_gate_report(path: str | Path, gate: dict[str, Any]) -> Path:
    lines = [f"# G4 gate — {gate['family']} ({gate['role']}) vs {gate['reference']}", "",
             f"**Result: {'PASS' if gate['passed'] else 'FAIL'}**", "",
             "| Check | Value | Threshold | Pass | Note |", "|---|---:|---:|:---:|---|"]
    for check in gate["checks"]:
        value = check["value"]
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {check['name']} | {shown} | {check['threshold']} | "
                     f"{'✅' if check['passed'] else '❌'} | {check['note']} |")
    lines += ["", f"Episodes: {gate['episodes']}"]
    target = Path(path)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
