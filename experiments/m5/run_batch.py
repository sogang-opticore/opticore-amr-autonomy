"""Run a serial M5 development batch and keep measured failures distinct from missing data."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys

SEEDS = (7, 19, 31)
SINGLE_CASES = ("baseline", "blockage", "sequence", "faults", "transport")
GOAL_TOLERANCE_M = 0.25
BLOCKAGE_EVENTS = {"moving_before_obstacle", "stopped_before_release", "actor_removed", "advanced_after_release"}
FAULT_EVENTS = {"drop_scans", "stale_stop", "restored_scans_motion", "fallback_motion",
                "learned_recovery", "cancel_sent", "cancelled_stop"}


def cases(suite):
    names = SINGLE_CASES if suite == "all" else ("baseline", "blockage") if suite == "paired" else (suite,)
    jobs = [(case, seed) for case in names if case != "pair" for seed in SEEDS]
    if suite in ("all", "pair"):
        jobs.append(("pair", SEEDS[0]))
    return jobs


def assess_single(report, case):
    legs = report["legs"]
    expected = len(report["case"]["goals"])
    # A failed leg can stop the driver before the remaining waypoints are issued.
    all_issued = len(legs) == expected
    statuses = [leg["row"]["termination_reason"] for leg in legs]
    successes = [tuple(leg["row"][k] for k in ("termination_reason", "success", "collision_known", "collision"))
                 == ("ARRIVED", "1", "1", "0") for leg in legs]
    for i, leg in enumerate(legs):
        if statuses[i] == "ARRIVED":
            error = leg["physical_goal_error_m"]
            successes[i] = (successes[i] and math.isfinite(error) and 0 <= error <= GOAL_TOLERANCE_M
                            and leg["states"].get("LEARNED", 0) > 0)
    checks = {"all_legs_recorded": all_issued}
    if case == "faults":
        checks["cancel_and_replacement"] = (all_issued and len(legs) == 2 and statuses[0] == "CANCELLED"
                                             and legs[0]["row"]["collision_known"] == "1"
                                             and legs[0]["row"]["collision"] == "0" and successes[1])
        checks["fault_sequence_observed"] = bool(legs) and FAULT_EVENTS <= legs[0]["events"].keys()
    else:
        checks["collision_free_arrivals"] = bool(legs) and all(successes)
    if case == "blockage":
        checks["stop_and_restart_observed"] = bool(legs) and BLOCKAGE_EVENTS <= legs[0]["events"].keys()
    return {"checks": checks, "passed": all(checks.values()), "leg_outcomes": statuses}


def run_case(folder, candidates, case, seed, index):
    here = Path(__file__).resolve().parent
    command = [sys.executable, str(here / ("run_pair.py" if case == "pair" else "run_single.py")),
               "--bundle", str(candidates / str(seed) / "bundle"), "--out", str(folder),
               "--domain", str(70 + index)]
    if case != "pair":
        command.extend(("--case", case))
    with folder.with_suffix(".log").open("x") as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    path = folder / "run.json"
    result = {"case": case, "seed": seed, "directory": folder.name, "exit_code": rc}
    if not path.exists():
        return {**result, "status": "invalid", "error": "missing run.json"}
    try:
        report = json.loads(path.read_text())
        if case != "pair" and rc == 0 and report["status"] == "recorded":
            return {**result, "status": "recorded", **assess_single(report, case)}
        if case == "pair" and rc == 0 and report["status"] == "verified":
            return {**result, "status": "recorded", "passed": True}
        if case == "pair" and rc != 0 and report["status"] == "failed":
            return {**result, "status": "recorded", "passed": False, "phase": report.get("phase"),
                    "error": report.get("error")}
    except (OSError, TypeError, KeyError, ValueError, AttributeError) as exc:
        return {**result, "status": "invalid", "error": f"invalid run.json: {exc}"}
    return {**result, "status": "invalid", "driver_status": report.get("status"), "error": report.get("error")}


def verdict(results):
    if any(row["status"] != "recorded" for row in results):
        return "invalid", 1
    if any(not row["passed"] for row in results):
        return "failed", 3
    return "passed", 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--suite", choices=("all", "paired", "faults", "transport", "sequence", "pair"), default="all")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve()
    jobs = cases(args.suite)
    report = {"suite": args.suite, "expected_trials": len(jobs), "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "results": []}
    for index, (case, seed) in enumerate(jobs):
        name = f"{index+1:02d}-{case}-{seed}"
        print(f"Starting {name}", flush=True)
        result = run_case(output / name, args.candidates.resolve(), case, seed, index)
        report["results"].append(result)
        report["finished_trials"] = len(report["results"])
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat() if index + 1 == len(jobs) else None
        report["suite_verdict"] = verdict(report["results"])[0] if report["finished_at_utc"] else "running"
        pending = output / "summary.tmp"
        pending.write_text(json.dumps(report, indent=2) + "\n")
        pending.replace(output / "summary.json")
        print(f"{name}: {result['status']}, passed={result.get('passed')}", flush=True)
    return verdict(report["results"])[1]


if __name__ == "__main__":
    sys.exit(main())
