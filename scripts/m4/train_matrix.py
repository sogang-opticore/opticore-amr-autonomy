"""Run the nine independent M4 PPO training jobs with fixed per-job compute budgets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

VARIANTS = ("full", "no_history", "no_randomization")
SEEDS = (7, 19, 31)
ROOT = Path(__file__).resolve().parents[2]


def run_one(job: tuple[str, int], output: Path) -> None:
    variant, seed = job
    folder = output / variant
    folder.mkdir(exist_ok=True, parents=True)
    destination = folder / str(seed)
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
           "PYTHONPATH": str(ROOT / "packages" / "amr_lite")}
    with (folder / f"{seed}.log").open("x") as log:
        print(f"{datetime.now(timezone.utc).isoformat()} start {variant}/{seed}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "amr_lite.cli.train_m4", "--out", str(destination),
             "--variant", variant, "--seed", str(seed), "--stage-steps", "65536"],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, env=env,
        )
        print(f"{datetime.now(timezone.utc).isoformat()} done  {variant}/{seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="root for new run folders")
    parser.add_argument("--jobs", type=int, choices=range(1, 10), default=1)
    parser.add_argument("--variants", choices=VARIANTS, nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", choices=SEEDS, type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    output = args.out.resolve()
    jobs = [(v, s) for v in args.variants for s in args.seeds]
    if len(set(jobs)) != len(jobs):
        parser.error("Each variant and seed may be requested only once.")
    for variant, seed in jobs:
        folder = output / variant
        if (folder / str(seed)).exists() or (folder / f"{seed}.log").exists():
            parser.error(f"{variant}/{seed} already exists. Choose a new output root.")
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, job, output) for job in jobs]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
