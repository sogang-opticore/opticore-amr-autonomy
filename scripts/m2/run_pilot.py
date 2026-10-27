#!/usr/bin/env python3
"""Acquire the M2 paired pilot after the contact calibration passes in this environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys

from summarize_baseline import SCORED, summarize
from mission_spec import MANIFEST, digest, load_cases

MISSION_RUNNER = MANIFEST.with_name("run_mission.py")
PILOT_MANIFEST = "905bd935b6a384c885f0a771bc447e526303ab6e22fb26a84a3f1dca26a1e2c5"
SEEDS = (0, 1)


def schedule():
    if digest(MANIFEST) != PILOT_MANIFEST:
        raise ValueError("mission manifest differs from the approved M2 pilot")
    missions = [c["id"] for c in load_cases()["missions"] if c["split"] == "train"]
    return [(mission, planner, seed) for seed, order in enumerate([("b0", "b1"), ("b1", "b0")])
            for mission in missions for planner in order]


def acquire(command, log):
    # Deliver interrupts to the runner, which owns shutdown of its ROS launch and Gazebo.
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
        raise


def run_pilot(out, domain):
    planned = schedule()
    out.mkdir(parents=True, exist_ok=False)
    file = out / "summary.json"
    attempts = []
    error = None
    exit_code = 0
    try:
        for mission, planner, seed in planned:
            name = f"{mission}-{planner}-{seed}"
            trial = out / name
            attempts.append(trial)
            command = [sys.executable, str(MISSION_RUNNER), mission, "--planner", planner,
                       "--seed", str(seed), "--domain", str(domain), "--out", str(trial)]
            print(f"[{len(attempts)}/{len(planned)}] {name}", flush=True)
            with (out / f"{name}.log").open("x") as log:
                code = acquire(command, log)
            result = summarize([trial], {"train"}, SEEDS)["input_directories"][0]
            print(f"  {result['disposition']}: {result['detail']}", flush=True)
            if code or result["disposition"] not in SCORED:
                error = f"{name}: runner exit {code}, {result['disposition']}: {result['detail']}"
                exit_code = 1
                break
    except KeyboardInterrupt:
        error, exit_code = "pilot interrupted", 130
    except (OSError, ValueError) as exc:
        error, exit_code = str(exc), 1
    finally:
        report = summarize(attempts, {"train"}, SEEDS)
        report.update(manifest_sha256=PILOT_MANIFEST, planned_order=planned, error=error)
        if not all(p["scored"] == p["expected"] for p in report["planners"].values()):
            exit_code = exit_code or 1
        report["complete"] = exit_code == 0
        temp = out / "summary.json.tmp"
        temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temp.replace(file)
        for planner, arm in report["planners"].items():
            print(f"{planner}: {arm['scored']}/{arm['expected']} scored, {arm['counts']}")
        print(file)
    if error:
        print(error, file=sys.stderr)
    return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new pilot directory; retains all raw runs")
    parser.add_argument("--domain", type=int, choices=range(102), default=81, metavar="0..101")
    args = parser.parse_args()
    try:
        return run_pilot(args.out.resolve(), args.domain)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
