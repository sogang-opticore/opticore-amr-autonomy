"""Verify and score one mixed-profile three-seed follow-up against frozen M4 controls."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from collect_training import read_json, verify_run
from evaluate import TRAIN_SEEDS
from summarize import paired, summarize
from amr_lite.experiments.manifest import REPOSITORY_ROOT


def score_followup(runs, source, jobs):
    common = {}
    verified = [verify_run(runs / str(seed), "full", seed, common, source, mixed=True)
                for seed in TRAIN_SEEDS]
    archived = REPOSITORY_ROOT / "experiments/m4/development"
    cases = list(product(("primary", "secondary"), TRAIN_SEEDS, (True, False)))

    with TemporaryDirectory() as tmpdir:
        def score(case):
            suite, seed, shield = case
            suffix = "" if shield else "-shield-off"
            name = f"{suite}-full{suffix}-{seed}.json"
            path = Path(tmpdir) / name
            command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                       "--bundle", str(runs / str(seed) / "bundle"), "--variant", "full",
                       "--suite", suite, "--out", str(path)]
            if not shield:
                command.append("--shield-off")
            # Each evaluation fingerprints its imported environment modules.
            # Start clean processes as in the original M4 evaluation.
            env = dict(os.environ, PYTHONPATH=str(REPOSITORY_ROOT / "packages/amr_lite"),
                       OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
            subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, check=True)
            return name, read_json(path)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            revised = dict(pool.map(score, cases))

    controls = {path.name: read_json(path) for path in sorted(archived.glob("*.json"))}
    if not revised.keys() <= controls.keys():
        raise ValueError("mixed episodes do not correspond to the archived full configuration")
    comparison = summarize((controls | revised).values())
    delta = {seed: paired(revised[name]["rows"], controls[name]["rows"])
             for seed in TRAIN_SEEDS for name in [f"primary-full-{seed}.json"]}
    return {
        "training_revision": common["git"]["commit"], "training_source_sha256": common["source"]["sha256"],
        "training": verified, "new_episodes": sum(len(r["rows"]) for r in revised.values()),
        "reused_episodes": sum(len(r["rows"]) for k, r in controls.items() if k not in revised),
        "mixed": comparison["configurations"]["full"],
        "mixed_shield_off": comparison["configurations"]["full_shield_off"],
        "paired_vs_previous_full": delta, "reports": revised,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="directory with the three mixed training runs")
    parser.add_argument("source", type=Path, help="clean checkout of the training revision")
    parser.add_argument("out", type=Path)
    parser.add_argument("--jobs", type=int, default=3, choices=range(1, 13))
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output file")
    report = score_followup(args.runs.resolve(), args.source.resolve(), args.jobs)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
