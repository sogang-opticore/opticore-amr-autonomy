from __future__ import annotations

import argparse
import json

from amr_lite.cli.train_rl import parse_overrides
from amr_lite.config import ROOT, deep_merge, load_config
from amr_lite.learning.dagger_loop import run_dagger


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BC -> DAgger rounds with selection evaluation")
    parser.add_argument("--config", default="dagger")
    parser.add_argument("--set", dest="overrides", action="append", metavar="KEY=VALUE")
    parser.add_argument("--output", default=str(ROOT / "artifacts/dagger/run"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = deep_merge(load_config(args.config), parse_overrides(args.overrides))
    print(json.dumps(run_dagger(args.output, config, smoke=args.smoke), indent=2, default=str))


if __name__ == "__main__":
    main()
