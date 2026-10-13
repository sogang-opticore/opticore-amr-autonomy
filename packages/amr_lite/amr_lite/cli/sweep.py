from __future__ import annotations

import argparse
import json

from amr_lite.experiments.sweep import Sweep, apply_smoke_budget, load_sweep_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an experiment sweep (configs/sweeps/<name>.yaml)")
    parser.add_argument("spec", help="sweep name under configs/sweeps/ or a YAML path")
    parser.add_argument("--stage", choices=["plan", "train", "select", "test", "all"], default="all",
                        help="'all' runs plan, train and select; the one-time test stage is explicit")
    parser.add_argument("--output")
    parser.add_argument("--parallel", type=int, help="concurrent training trials")
    parser.add_argument("--smoke", action="store_true", help="two seeds, tiny budgets, two scenarios")
    parser.add_argument("--force", action="store_true", help="re-run the test stage (repair only)")
    args = parser.parse_args()
    spec = load_sweep_spec(args.spec)
    if args.smoke:
        spec = apply_smoke_budget(spec)
    sweep = Sweep(spec, args.output, smoke=args.smoke)
    if args.stage == "plan":
        result = {"trials": [trial["id"] for trial in sweep.plan()]}
    elif args.stage == "train":
        result = sweep.train(args.parallel)
    elif args.stage == "select":
        result = sweep.select()
    elif args.stage == "test":
        result = sweep.test(force=args.force)
    else:
        sweep.train(args.parallel)
        result = sweep.select()
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
