"""Summarize the frozen M3 post-training trace without rerunning the simulator.

Usage: python3 scripts/m4/diagnose_m3.py /path/to/train-65536-r2
The default input is the original M3 run location in this checkout.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROVENANCE = ROOT / "experiments/m3/training-provenance.json"
MANIFEST = ROOT / "experiments/m2/missions.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(rows, case, manifest):
    start = manifest["stations"][case["from"]]
    goal = manifest["stations"][case["to"]]
    bearing = math.atan2(goal[1] - start[1], goal[0] - start[0])
    heading_error = math.remainder(bearing - manifest["start_yaw_rad"], 2 * math.pi)
    closest = min(rows, key=lambda row: row["goal_error"])
    final = rows[-1]
    counts = Counter(row["shield_mode"] for row in rows)
    interventions = len(rows) - counts["PASS"]
    if final["shield_interventions"] != interventions:
        raise ValueError(f"shield count disagrees with the recorder for {case['id']}")
    result = {
        "mission": case["id"], "evaluation_seed": rows[0]["seed"], "ticks": len(rows),
        "start_goal_distance_m": round(math.dist(start, goal), 6),
        "start_goal_bearing_minus_yaw_deg": round(math.degrees(heading_error), 6),
        "closest": {
            "time_s": round(closest["elapsed"], 6),
            "goal_error_m": round(closest["goal_error"], 6),
            "linear_velocity_m_s": round(closest["velocity"][0], 6),
            "shield_mode": closest["shield_mode"],
        },
        "final": {
            "time_s": round(final["elapsed"], 6), "disposition": final["disposition"],
            "goal_error_m": round(final["goal_error"], 6),
            "linear_velocity_m_s": round(final["velocity"][0], 6),
            "progress_reward": round(final["reward_totals"]["progress"], 6),
        },
        "shield_mode_ticks": dict(sorted(counts.items())),
        "shield_intervention_ticks": interventions,
    }
    removed = (case["actor"] or {}).get("remove_at")
    if removed is not None:
        later = [row for row in rows if row["elapsed"] >= removed - 1e-8]
        result["after_actor_removal"] = {
            "removal_time_s": removed, "ticks": len(later),
            "shield_intervention_ticks": sum(row["shield_mode"] != "PASS" for row in later),
            "closest_goal_error_m": round(min(row["goal_error"] for row in later), 6),
        }
    return result


def diagnose(run, reference):
    files = reference["artifacts_sha256"]
    hashes = {name: digest(run / name) for name in ("after-trace.jsonl", "resolved.json", "final.zip")}
    if any(sha != files[name] for name, sha in hashes.items()):
        raise ValueError("input folder does not match the recorded M3 run")
    if digest(MANIFEST) != reference["mission_manifest_sha256"]:
        raise ValueError("mission manifest differs from the M3 training manifest")
    manifest = json.loads(MANIFEST.read_text())
    train = {case["id"]: case for case in manifest["missions"] if case["split"] == "train"}
    settings = json.loads((run / "resolved.json").read_text())
    by_mission = defaultdict(list)
    with (run / "after-trace.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            by_mission[row["mission_id"]].append(row)
    if set(by_mission) != set(train):
        raise ValueError("expected all and only M3 train missions")
    return {
        "m3_training_commit": reference["autonomy_commit"], "training_seed": reference["seed"],
        "input_sha256": {**hashes, "experiments/m2/missions.json": digest(MANIFEST)},
        "goal_tolerance_m": settings["environment"]["goal_tolerance"],
        "maximum_linear_velocity_m_s": settings["environment"]["v_max"],
        "missions": [describe(by_mission[key], case, manifest) for key, case in train.items()],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path)
    args = parser.parse_args()
    reference = json.loads(PROVENANCE.read_text())["training"]
    result = diagnose(args.run or ROOT / reference["local_directory"], reference)
    print(json.dumps(result, indent=2, allow_nan=False))
