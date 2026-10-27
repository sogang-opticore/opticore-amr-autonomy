"""Inventory M2 single-goal runs. Pass the runner's output directories, not CSV files."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "m2"))
from mission_spec import MAP, digest, load_cases, resolve
from run_mission import check_evidence

PLANNERS = ("b0", "b1")
SCORED = {"success", "collision", "timeout"}
CONTACT_SOURCES = {"gazebo_contacts", "ground_truth_geometry"}
ACTIVE_STATES = {"NORMAL", "ALIGN", "REJOIN", "SPIN", "FORWARD_ONLY", "EMERGENCY", "AVOID"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def check_events(path, row, case):
    """Check the raw recorder boundaries against the row before accepting its disposition."""
    goal = end = None
    contact_sources, coverage_epochs = set(), set()
    matched_at, armed_at, reached_at, pose = None, None, None, None
    contacts = []
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            require(event["run_id"] == row["run_id"] and event["version"] == 2, "mixed raw run IDs/versions")
            if event["episode_id"] is None:
                continue
            require(event["episode_id"] == row["episode_id"] and event["goal_id"] == row["goal_id"],
                    "mixed raw episode/goal IDs")
            require(end is None, "episode evidence after terminal record")
            kind = event["kind"]
            if kind == "goal":
                require(goal is None and [event["x"], event["y"]] == case["goal"] and event["frame"] == "map"
                        and event["scenario_id"] == row["scenario_id"] and event["seed"] == int(row["seed"]),
                        "duplicate or unrelated goal")
                goal = event
            else:
                require(goal is not None, "event before goal")
            if kind in {"initial_contact", "contact"}:
                contact_sources.add(event["source"])
                coverage_epochs.add(event.get("coverage_epoch"))
                if event["source"] == "gazebo_contacts":
                    require(all(type(event.get(k)) is int and event[k] >= 0
                                for k in ("coverage_epoch", "collision_events")), "unversioned contact telemetry")
                stamp = event["stamp"] if kind == "initial_contact" else event["time"]
                require(math.isfinite(stamp) and type(event["collision"]) is bool, "invalid raw contact")
                contacts.append((stamp, event["collision"], event.get("collision_events")))
            elif kind == "path":
                require(type(event["matching"]) is bool and event["time"] >= goal["time"], "invalid path match")
                if event["matching"]:
                    matched_at = event["time"]
                else:
                    matched_at = armed_at = None
            elif kind == "pose":
                if event["frame"] == row["frame"]:
                    pose = (event["time"], (event["x"], event["y"]))
            elif kind == "status" and matched_at is not None and event["time"] >= matched_at:
                if event["value"] in ACTIVE_STATES:
                    armed_at = event["time"]
                elif (event["value"] in {"REACHED", "GOAL_REACHED"} and pose and armed_at is not None
                      and event["time"] >= armed_at and 0 <= event["time"] - pose[0] <= goal["max_pose_age"]
                      and math.dist(pose[1], case["goal"]) <= goal["goal_tolerance"]):
                    reached_at = event["time"]
            elif kind == "end":
                end = event
    require(goal and end and goal["time"] == float(row["started"]) and end["time"] == float(row["ended"]),
            "missing or inconsistent goal/end events")
    csv_collision = None if row["collision"] == "" else int(row["collision"])
    require(end["reason"] == row["termination_reason"] and end["success"] == int(row["success"])
            and end["collision"] == csv_collision, "recorder end contradicts CSV")
    if end["reason"] == "ARRIVED":
        require(reached_at == end["time"], "missing path, pose or goal-qualified arrival status")
    if row["collision_known"] == "1":
        source = row["contact_source"]
        require(source in contact_sources and source in CONTACT_SOURCES and contacts, "missing contact witness")
        start, stop = float(row["started"]), float(row["ended"])
        impact = any(start <= t <= stop and hit for t, hit, _ in contacts)
        for (prev, _, a), (now, _, b) in zip(contacts, contacts[1:]):
            if start <= prev < now <= stop and a is not None and b is not None and b > a:
                impact = True
        require(impact == bool(csv_collision), "contact witness contradicts terminal flag")
        if not impact:
            gap = float(goal["contact_max_gap"])
            require(math.isfinite(gap) and gap > 0 and start - gap <= contacts[0][0] <= start
                    and contacts[-1][0] >= stop and len(contact_sources) == 1 and len(coverage_epochs) == 1,
                    "incomplete contact time span or coverage epoch change")
            require(all(0 < now - prev <= gap and (a is None or b is None or a == b)
                        for (prev, _, a), (now, _, b) in zip(contacts, contacts[1:])),
                    "gap or cumulative count change in contact trace")



def recorded_result(out, run, case):
    require(run["map_sha256"] == case["map_sha256"] == run["installed_hashes"]["maps/warehouse_map.pgm"]
            and run["manifest_sha256"] == case["manifest_sha256"], "source digest mismatch")
    import yaml  # Same dependency used by the ROS runner to validate finalized rosbag2 metadata.
    try:
        check_evidence(out)
    except (RuntimeError, yaml.YAMLError) as error:
        raise ValueError(f"raw evidence: {error}") from error
    for field in ("start", "goal", "actor", "timeout_sec"):
        require(run[field] == case[field], f"condition differs from manifest: {field}")
    require(run["spawn_world"] == list(case["spawn_world"]), "world spawn differs from manifest")
    actor = out / "actor.jsonl"
    require(actor.is_file() and (case["actor"] is None or nonempty(actor)), "missing obstacle log")
    with (out / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == 1, "runner must produce exactly one goal row")
    row = rows[0]
    require(row["version"] == "2" and row["run_id"] == run["run_id"] and row["scenario_id"] == case["id"]
            and row["policy"] == run["planner"] and int(row["seed"]) == run["seed"] and row["frame"] == "map"
            and row["episode_id"] and row["goal_id"], "recorder identity mismatch")
    require([float(row["goal_x"]), float(row["goal_y"])] == case["goal"] and math.isclose(
        float(row["started"]), run["goal_stamp"], abs_tol=1e-6, rel_tol=0), "goal dispatch mismatch")
    duration = float(row["ended"]) - float(row["started"])
    require(math.isfinite(duration) and duration >= 0 and math.isclose(duration, float(row["episode_time"])),
            "inconsistent episode clock")
    check_events(out / "episodes.events.jsonl", row, case)
    reason = row["termination_reason"]
    known = row["collision_known"] == "1"
    require(row["collision_known"] in {"0", "1"} and row["collision"] in ({"0", "1"} if known else {""}),
            "invalid collision field")
    require(row["arrived"] == str(int(reason == "ARRIVED")) and row["truncated"] == str(int(reason == "TIMEOUT"))
            and row["success"] == str(int(reason == "ARRIVED" and known and row["collision"] == "0")),
            "inconsistent episode flags")
    expected_failure = "" if row["success"] == "1" else "CONTACT_UNKNOWN" if reason == "ARRIVED" else reason
    require(row["failure_type"] == expected_failure, "failure_type contradicts terminal disposition")
    if reason in {"CLOCK_RESET", "PREEMPTED", "INTERRUPTED"}:
        return "rejected", reason
    if reason == "COLLISION":
        require(row["collision"] == "1" and duration <= case["timeout_sec"], "unsubstantiated collision")
        return "collision", reason
    if reason == "TIMEOUT":
        require(math.isclose(duration, case["timeout_sec"]) and row["collision"] != "1", "bad timeout boundary")
        return ("timeout" if known else "timeout_contact_unknown"), reason
    require(reason == "ARRIVED" and row["collision"] != "1" and duration < case["timeout_sec"], "invalid arrival")
    return ("success" if known else "contact_unknown"), reason


def summarize(directories, splits, seeds):
    spec = load_cases()
    require(digest(MAP) == spec["map_sha256"], "local warehouse map does not match the mission manifest")
    cases = {c["id"]: resolve(c["id"]) for c in spec["missions"] if c["split"] in splits}
    expected = {(p, c, s) for p in PLANNERS for c in cases for s in seeds}
    attempts, keys, run_ids = [], defaultdict(list), defaultdict(list)
    for out in directories:
        entry = {"directory": str(out), "disposition": "rejected"}
        attempts.append(entry)
        try:
            run = json.loads((out / "run.json").read_text())
            case = cases.get(run["id"])
            if run["split"] not in splits:
                entry.update(disposition="excluded", detail="outside requested development manifest/split")
                continue  # Never open outcomes for an unselected or held-out mission.
            require(case is not None, "unregistered mission for requested development split")
            require(run["split"] == case["split"] and run["scenario_id"] == case["id"]
                    and run["family"] == case["family"], "wrong mission metadata")
            key = run["planner"], case["id"], run["seed"]
            require(type(run["seed"]) is int and key in expected, "unexpected planner or repetition")
            keys[key].append(entry)
            require(isinstance(run["run_id"], str) and run["run_id"], "missing run ID")
            run_ids[run["run_id"]].append(entry)
            entry.update(planner=key[0], mission=key[1], seed=key[2], run_id=run["run_id"])
            if run["status"] == "invalid":
                # A failed dispatch/partial log still consumes the scheduled attempt.
                before_goal = ("goal_stamp" not in run and not nonempty(out / "episodes.csv")
                               and not nonempty(out / "episodes.events.jsonl"))
                entry.update(disposition="runner_error", detail=str(run.get("error", "invalid")),
                             pre_dispatch=before_goal)
                continue
            if run["status"] == "running":
                entry.update(disposition="incomplete", detail="runner has no final status")
                continue
            require(run["status"] == "recorded", "unknown runner status")
            entry["disposition"], entry["detail"] = recorded_result(out, run, case)
        except (OSError, ValueError, KeyError, TypeError, csv.Error) as err:
            entry["detail"] = str(err)
    for group in run_ids.values():
        if len(group) > 1:
            for entry in group:
                entry.update(disposition="rejected", detail="duplicate run ID")
    planners = {}
    for planner in PLANNERS:
        trials = []
        for _, mission, seed in sorted(k for k in expected if k[0] == planner):
            entries = keys[(planner, mission, seed)]
            scored_attempts = [e for e in entries if not (e["disposition"] == "runner_error" and e["pre_dispatch"])]
            if len(scored_attempts) > 1:
                for entry in scored_attempts:
                    entry.update(disposition="rejected", detail="multiple attempts for planner/mission/repetition")
            if scored_attempts:
                disposition = scored_attempts[0]["disposition"]
            else:
                disposition = "runner_error" if entries else "missing"
            trials.append({"mission": mission, "seed": seed, "disposition": disposition})
        counts = Counter(t["disposition"] for t in trials)
        complete = bool(trials) and all(t["disposition"] in SCORED for t in trials)
        planners[planner] = {"expected": len(trials), "scored": sum(counts[s] for s in SCORED),
                             "counts": dict(counts), "trials": trials,
                             "completion_rate": counts["success"] / len(trials) if complete else None}
    return {"splits": sorted(splits), "seeds": sorted(seeds), "planners": planners, "input_directories": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path, help="explicit run directories; copies count as duplicates")
    parser.add_argument("--split", choices=("train", "selection"), action="append", help="default: train")
    parser.add_argument("--seed", type=int, action="append", help="scheduled repetition IDs; default: 0")
    args = parser.parse_args()
    seeds = set(args.seed or [0])
    if min(seeds) < 0:
        parser.error("seed must be nonnegative")
    print(json.dumps(summarize(args.directories, set(args.split or ["train"]), seeds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
