"""Check the three M4 candidate actors on verified serial Gazebo inputs."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
from stable_baselines3 import PPO
import torch

from amr_lite.deploy.runtime import GOLDEN_TOLERANCE, PolicyRuntime
from amr_lite.envs.dynamics import RobotState
from amr_lite.experiments.manifest import REPOSITORY_ROOT, sha256_file
from amr_lite.learning.stamped_io import M3_IO_ID, StampedPolicyIO
from amr_lite.learning.stamped_replay import replay


def verified_inputs(root, reference):
    directory = root / Path(reference["directory"]).name
    evidence = reference["evidence_sha256"]
    for name in ("run.json", "bag/bag_0.db3", "replay.jsonl"):
        if sha256_file(directory / name) != evidence[name]:
            raise ValueError(f"{directory.name}: {name} differs from the M3 run")
    trace = directory / "replay.jsonl"
    with trace.open() as stream:
        rebuilt = replay(stream)
    samples = []
    with trace.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "tick" and row["action"] is not None:
                samples.append((np.asarray(row["observation"]["tensor"], dtype=np.float32),
                                RobotState(**row["robot"]), row["deadline_ms"]))
    archived = reference["host_replay"]["counts"]
    if (len(samples) != archived["policy_prediction"] or not samples
            or rebuilt["counts"]["valid_observation"] != archived["valid_observation"]):
        raise ValueError(f"{directory.name}: incomplete replay")
    return samples, {"run_id": reference["run_id"], "replay_sha256": evidence["replay.jsonl"],
                     "bag_sha256": evidence["bag/bag_0.db3"], "ticks": rebuilt["counts"]["tick"],
                     "valid_observations": rebuilt["counts"]["valid_observation"],
                     "max_observation_error": rebuilt["max_observation_error"],
                     "policy_inference_ticks": len(samples)}


def run_actor(runtime, model, samples):
    actions, elapsed, faults = [], [], Counter()
    for observation, state, deadline in samples:
        start = time.perf_counter_ns()
        action = runtime.predict(observation)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        elapsed.append(elapsed_ms)
        np.testing.assert_array_equal(action, runtime.predict(observation))
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("non-finite or out-of-contract action")
        result = StampedPolicyIO.actuate(action, state, inference_ms=elapsed_ms, deadline_ms=deadline)
        if result.fault:
            if result.fault != "INFERENCE_DEADLINE" or result.command != (0.0, 0.0):
                raise ValueError(f"unexpected physical command: {result}")
            faults[result.fault] += 1
        actions.append(action)
    actions = np.stack(actions)
    sb3, _ = model.predict(np.stack([obs for obs, _, _ in samples]), deterministic=True)
    error = float(np.max(np.abs(actions - sb3)))
    if not error <= GOLDEN_TOLERANCE:
        raise ValueError(f"SB3 and exported actor disagree: {error}")
    return {"predictions": len(actions), "sb3_max_abs_error": error,
            "repeat_max_abs_error": 0.0,
            "actions_float32_le_sha256": hashlib.sha256(actions.astype("<f4").tobytes()).hexdigest(),
            "host_predict_ms": {"median": float(np.median(elapsed)), "p95": float(np.percentile(elapsed, 95)),
                                "maximum": max(elapsed)},
            "deadline_misses": faults["INFERENCE_DEADLINE"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("m3_runs", type=Path, help="parent of shadow-train-serial and learned-train-serial")
    parser.add_argument("m4_runs", type=Path, help="parent of the three mixed candidate seed directories")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    experiments = REPOSITORY_ROOT / "experiments"
    m3 = json.loads((experiments / "m3/warehouse-verification.json").read_text())
    m4 = json.loads((experiments / "m4/followup/mixed.json").read_text())
    if not m4["mixed"]["gate"]["eligible"]:
        raise ValueError("mixed policy was not selected")
    runs = {r["mode"]: verified_inputs(args.m3_runs, r) for r in m3["results"]
            if r["runtime"] == "serial-single-thread-blas"}
    if set(runs) != {"shadow", "learned"} or sorted(r["seed"] for r in m4["training"]) != [7, 19, 31]:
        raise ValueError("incomplete traces or candidate set")
    torch.set_num_threads(1)
    actors = []
    for training in m4["training"]:
        seed = training["seed"]
        root = args.m4_runs / str(seed)
        bundle = root / "bundle"
        hashes = {"bundle.json": training["bundle_sha256"], **training["bundle_files_sha256"]}
        for name, expected in hashes.items():
            if sha256_file(bundle / name) != expected:
                raise ValueError(f"candidate {seed}/{name} digest mismatch")
        checkpoint = training["stages"][-1]["checkpoint_sha256"]
        if sha256_file(root / "final.zip") != checkpoint:
            raise ValueError(f"candidate {seed} checkpoint digest mismatch")
        runtime = PolicyRuntime.load(bundle)
        if runtime.manifest["observation_contract"].get("acquisition_contract") != M3_IO_ID:
            raise ValueError("candidate acquisition contract mismatch")
        model = PPO.load(root / "final.zip", device="cpu")
        actors.append({"seed": seed, "bundle_sha256": training["bundle_sha256"], "checkpoint_sha256": checkpoint,
                       "traces": {mode: run_actor(runtime, model, samples) for mode, (samples, _) in runs.items()}})
    report = {"scope": "Offline policy inference on fixed historical M3 train trajectories",
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
              "blas_environment": {k: os.environ.get(k) for k in
                                   ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
              "prediction_deadlines_ms": sorted({t for samples, _ in runs.values() for _, _, t in samples}),
              "traces": {mode: summary for mode, (_, summary) in runs.items()}, "actors": actors}
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
