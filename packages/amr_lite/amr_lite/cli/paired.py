from __future__ import annotations

import argparse
import json

from amr_lite.config import resolve_env_config
from amr_lite.experiments.paired import (
    gap_report,
    load_policy_spec,
    read_rows,
    run_lite_paired,
    write_gap_report,
    write_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lite side of paired Lite <-> Gazebo evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    lite = commands.add_parser("lite", help="run a paired set in Lite and write a paired CSV")
    lite.add_argument("--policy", default="rule", help="rule | bundle:<dir> | checkpoint:<file.pt>")
    lite.add_argument("--set", dest="set_name", default="warehouse_v1")
    lite.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    lite.add_argument("--legs", type=int, nargs="+")
    lite.add_argument("--label", default="lite")
    lite.add_argument("--robot-profile", default="opticore_gazebo")
    lite.add_argument("--shield", choices=["on", "off"], default="on")
    lite.add_argument("--workers", type=int, default=1)
    lite.add_argument("--output", required=True)
    gap = commands.add_parser("gap", help="compare two paired CSVs (e.g. Lite vs Gazebo)")
    gap.add_argument("--reference", required=True)
    gap.add_argument("--candidate", required=True)
    gap.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "lite":
        policy = load_policy_spec(args.policy, resolve_env_config(args.robot_profile))
        rows = run_lite_paired(policy, args.set_name, args.seeds, label=args.label,
                               robot_profile=args.robot_profile, shield_enabled=args.shield == "on",
                               legs=args.legs, workers=args.workers,
                               backend="subprocess" if args.workers > 1 else "sync")
        path = write_rows(args.output, rows)
        print(json.dumps({"output": str(path), "episodes": len(rows),
                          "success_rate": sum(row["success"] for row in rows) / max(1, len(rows))}, indent=2))
    else:
        reference, candidate = read_rows(args.reference), read_rows(args.candidate)
        report = gap_report(reference, candidate)
        label = lambda rows: rows[0]["policy"] if rows else "?"
        path = write_gap_report(args.output, report, label(reference), label(candidate))
        print(json.dumps({"output": str(path), **report["overall"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
