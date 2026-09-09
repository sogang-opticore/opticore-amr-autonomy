from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path: str | Path, rows: list[dict]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write empty result table")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(str(row["policy"]), bool(row["shield_enabled"]))].append(row)
    output = []
    for (policy, shield), values in sorted(groups.items()):
        count = len(values)
        total_steps = max(1, sum(int(item["steps"]) for item in values))
        output.append({
            "policy": policy,
            "shield_enabled": shield,
            "episodes": count,
            "success_rate": sum(int(item["success"]) for item in values) / count,
            "collision_rate": sum(int(item["collision"]) for item in values) / count,
            "timeout_rate": sum(int(item["truncated"]) for item in values) / count,
            "mean_episode_time": statistics.mean(float(item["episode_time"]) for item in values),
            "median_episode_time": statistics.median(float(item["episode_time"]) for item in values),
            "mean_path_length_ratio": statistics.mean(float(item["path_length_ratio"]) for item in values),
            "minimum_clearance": min(float(item["minimum_clearance"]) for item in values),
            "mean_cross_track_error": statistics.mean(float(item["mean_cross_track_error"]) for item in values),
            "mean_safety_cost": statistics.mean(float(item["safety_cost"]) for item in values),
            "shield_interventions_per_episode": statistics.mean(int(item["shield_interventions"]) for item in values),
            "shield_pass_ratio": sum(int(item["shield_pass"]) for item in values) / total_steps,
            "shield_clamp_ratio": sum(int(item["shield_clamp"]) for item in values) / total_steps,
            "shield_stop_ratio": sum(int(item["shield_stop"]) for item in values) / total_steps,
            "unsafe_without_shield_per_episode": statistics.mean(int(item["unsafe_without_shield"]) for item in values),
            "max_shield_sustained_time": max(float(item["shield_sustained_time"]) for item in values),
            "mean_inference_ms": statistics.mean(float(item["inference_ms"]) for item in values),
            "mean_simulation_steps_per_second": statistics.mean(float(item["simulation_steps_per_second"]) for item in values),
        })
    return output
