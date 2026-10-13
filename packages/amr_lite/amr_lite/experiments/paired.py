"""Lite side of the Lite <-> Gazebo paired evaluation and the gap report (ROADMAP M6).

Rows use the columns written by the ROS ``goal_sequence_runner`` so CSVs from
both simulators (or from DWA and learned runs) can be compared leg by leg.
Lite uses the paired set's ``goal_timeout_sec`` instead of the path-scaled
timeout so both sides give up after the same time.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from amr_lite.config import deep_merge, load_config, resolve_env_config

PAIRED_COLUMNS = ("policy", "scenario_id", "seed", "success", "collision", "truncated", "failure_type",
                  "episode_time", "path_length", "minimum_clearance", "shield_intervention_rate")


def load_policy_spec(spec: str, env_config: dict[str, Any]) -> Any:
    """``rule``, ``bundle:<directory>`` or ``checkpoint:<file.pt>``."""
    if spec == "rule":
        from amr_lite.planning.rule_baseline import RuleBasedPlanner

        return RuleBasedPlanner(env_config, load_config("baseline"))
    kind, _, location = spec.partition(":")
    if kind == "bundle":
        from amr_lite.deploy.runtime import PolicyRuntime

        return PolicyRuntime.load(location)
    if kind == "checkpoint":
        from amr_lite.learning.networks import load_policy

        return load_policy(location)
    raise ValueError(f"Unknown policy spec {spec!r}; use rule, bundle:<dir> or checkpoint:<file>")


def run_lite_paired(policy: Any, set_name: str, seeds: Sequence[int], *, label: str = "lite",
                    robot_profile: str = "opticore_gazebo", shield_enabled: bool = True,
                    legs: Sequence[int] | None = None, workers: int = 1, backend: str = "sync",
                    overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    from amr_lite.envs.warehouse_scenarios import paired_spec
    from amr_lite.rl.evaluation import run_episodes

    spec = paired_spec(set_name)
    config = resolve_env_config(robot_profile, deep_merge({
        "shield_strategy": "scan", "shield_enabled": bool(shield_enabled),
        "episode_timeout": float(spec.get("goal_timeout_sec", 120.0)), "episode_timeout_per_meter": 0.0,
    }, overrides))
    selected = list(range(len(spec["legs"]))) if legs is None else [int(leg) for leg in legs]
    episodes = [(f"WH-{set_name}-{leg:02d}", int(seed)) for seed in seeds for leg in selected]
    rows = []
    for (scenario, seed), info in zip(episodes, run_episodes(policy, episodes, config, workers=workers,
                                                             backend=backend)):
        counts = info["shield_counts"]
        rows.append({
            "policy": label, "scenario_id": scenario, "seed": seed, "success": int(info["success"]),
            "collision": int(info["collision"]), "truncated": int(info["truncated"]),
            "failure_type": info.get("failure_type", ""), "episode_time": float(info["elapsed_time"]),
            "path_length": float(info["path_length"]), "minimum_clearance": float(info["minimum_clearance_episode"]),
            "shield_intervention_rate": (counts["CLAMP"] + counts["STOP"]) / max(1, int(info["episode_length"])),
        })
    return rows


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PAIRED_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return target


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _flag(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "0")).strip().lower() in {"1", "true"}


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(_flag(row, key) for row in rows) / len(rows) if rows else math.nan


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else math.nan


def gap_report(reference: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-leg and overall differences; ``candidate - reference`` for rates."""
    legs = sorted({row["scenario_id"] for row in reference} | {row["scenario_id"] for row in candidate})
    per_leg, time_pairs = [], []
    for leg in legs:
        left = [row for row in reference if row["scenario_id"] == leg]
        right = [row for row in candidate if row["scenario_id"] == leg]
        left_time = _mean([float(row["episode_time"]) for row in left if _flag(row, "success")])
        right_time = _mean([float(row["episode_time"]) for row in right if _flag(row, "success")])
        if math.isfinite(left_time) and math.isfinite(right_time):
            time_pairs.append((left_time, right_time))
        failures: dict[str, int] = {}
        for row in right:
            if not _flag(row, "success"):
                failures[row.get("failure_type") or "UNKNOWN"] = failures.get(row.get("failure_type") or "UNKNOWN", 0) + 1
        per_leg.append({
            "scenario_id": leg, "reference_episodes": len(left), "candidate_episodes": len(right),
            "reference_success": _rate(left, "success"), "candidate_success": _rate(right, "success"),
            "success_delta": _rate(right, "success") - _rate(left, "success"),
            "reference_collision": _rate(left, "collision"), "candidate_collision": _rate(right, "collision"),
            "reference_time": left_time, "candidate_time": right_time,
            "reference_min_clearance": _mean([float(row["minimum_clearance"]) for row in left
                                              if row.get("minimum_clearance") not in ("", "nan")]),
            "candidate_min_clearance": _mean([float(row["minimum_clearance"]) for row in right
                                              if row.get("minimum_clearance") not in ("", "nan")]),
            "candidate_failures": failures,
        })
    overall = {
        "reference_success": _rate(reference, "success"), "candidate_success": _rate(candidate, "success"),
        "reference_collision": _rate(reference, "collision"), "candidate_collision": _rate(candidate, "collision"),
        "time_ratio": (statistics.mean(right for _, right in time_pairs) / statistics.mean(left for left, _ in time_pairs)
                       if time_pairs else math.nan),
        "legs_compared_for_time": len(time_pairs),
    }
    overall["success_delta"] = overall["candidate_success"] - overall["reference_success"]
    return {"overall": overall, "legs": per_leg}


def write_gap_report(path: str | Path, report: dict[str, Any], reference_label: str, candidate_label: str) -> Path:
    overall = report["overall"]
    percent = lambda value: "n/a" if not math.isfinite(value) else f"{100.0 * value:.1f}%"
    seconds = lambda value: "n/a" if not math.isfinite(value) else f"{value:.1f}"
    lines = [f"# Paired gap report — {candidate_label} vs {reference_label}", "",
             f"- Success: {percent(overall['candidate_success'])} vs {percent(overall['reference_success'])} "
             f"(delta {percent(overall['success_delta'])})",
             f"- Collision (proxy in Gazebo): {percent(overall['candidate_collision'])} vs "
             f"{percent(overall['reference_collision'])}",
             f"- Time ratio on legs both solved: {overall['time_ratio']:.3f} "
             f"({overall['legs_compared_for_time']} legs)", "",
             "| Leg | n ref | n cand | Success ref | Success cand | Δ | Collision cand | Time ref (s) | "
             "Time cand (s) | Failures (cand) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for leg in report["legs"]:
        failures = ", ".join(f"{key} {value}" for key, value in sorted(leg["candidate_failures"].items())) or "-"
        lines.append(f"| {leg['scenario_id']} | {leg['reference_episodes']} | {leg['candidate_episodes']} | "
                     f"{percent(leg['reference_success'])} | {percent(leg['candidate_success'])} | "
                     f"{percent(leg['success_delta'])} | {percent(leg['candidate_collision'])} | "
                     f"{seconds(leg['reference_time'])} | {seconds(leg['candidate_time'])} | {failures} |")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
