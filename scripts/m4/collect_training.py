"""Validate the nine frozen M4 training jobs and summarize their rollout outcomes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
from zipfile import ZipFile

import numpy as np

from amr_lite.deploy.runtime import GOLDEN_TOLERANCE, PolicyRuntime
from amr_lite.experiments.manifest import sha256_file
from train_matrix import SEEDS, VARIANTS

STAGE_STEPS = 65536
WORKERS = 4
OUTCOMES = ("success", "collision", "timeout", "invalid")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def verify_source(source, settings):
    revision = subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip()
    require(settings["git"]["commit"] == revision and not settings["git"]["dirty"],
            "training must identify the specified clean revision")
    subprocess.run(["git", "-C", source, "diff", "--quiet", "HEAD"], check=True)
    combined = hashlib.sha256()
    package = source / "packages/amr_lite"
    for entry in settings["source"]["files"]:
        name = entry["path"]
        file = package / name
        require(file.resolve().is_relative_to(package.resolve()), "source path outside package")
        require(file.stat().st_size == entry["bytes"] and sha256_file(file) == entry["sha256"],
                f"source file mismatch: {name}")
        combined.update(name.encode())
        combined.update(entry["sha256"].encode("ascii"))
    require(combined.hexdigest() == settings["source"]["sha256"], "combined source hash mismatch")
    assets = {
        "curriculum_manifest_sha256": "experiments/m4/curriculum.json",
        "mission_manifest_sha256": "experiments/m2/missions.json",
        "map_sha256": "packages/amr_lite/assets/maps/opticore_warehouse/warehouse_map.pgm",
        "policy_io_sha256": "packages/amr_lite/configs/m2/policy_io.yaml",
        "m2_ppo_sha256": "packages/amr_lite/configs/m2/ppo_start.yaml",
    }
    for key, name in assets.items():
        require(sha256_file(source / name) == settings[key], f"{name} hash mismatch")
    stages = read_json(source / assets["curriculum_manifest_sha256"])["stages"]
    require(len(stages) == 3 and stages == settings["curriculum_stages"], "curriculum does not match source")


def utc_time(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.utcoffset() == timedelta(0), f"expected a UTC runtime timestamp: {value}")
    return parsed


def verify_stage(root, stats, settings, index):
    number = index + 1
    seed = settings["seed"]
    checkpoint = "final.zip" if number == 3 else f"stage-{number}.zip"
    require(stats["stage"] == number and stats["checkpoint"] == checkpoint, "incorrect checkpoint schedule")
    count = STAGE_STEPS * number
    expected_updates = count // (WORKERS * settings["ppo"]["ppo"]["n_steps"]) * settings["ppo"]["ppo"]["n_epochs"]
    checkpoint_hash = sha256_file(root / checkpoint)
    require(checkpoint_hash == stats["checkpoint_sha256"], f"{root / checkpoint}: checkpoint hash mismatch")
    with ZipFile(root / checkpoint) as archive:
        data = json.loads(archive.read("data"))
        require("policy.optimizer.pth" in archive.namelist()
                and data["num_timesteps"] == stats["total_timesteps"] == count
                and data["_n_updates"] == stats["optimizer_updates"] == expected_updates
                and data["seed"] == seed and data["n_envs"] == WORKERS, "SB3 budget/optimizer mismatch")
    log = root / f"stage-{number}-episodes.jsonl"
    require(sha256_file(log) == stats["episodes_sha256"], f"{log}: episode log hash mismatch")
    outcomes = Counter()
    episode_index = Counter()
    missions = settings["curriculum_stages"][index]
    for line in log.read_text().splitlines():
        row = json.loads(line)
        rank = row["worker"]
        require(0 <= rank < WORKERS and row["episode_index"] == episode_index[rank], "worker episode mismatch")
        expected_mission = missions[(rank + episode_index[rank]) % len(missions)]
        require(row["worker_seed"] == settings["worker_seeds_by_stage"][index][rank]
                and row["mission_id"] == expected_mission
                and count - STAGE_STEPS < row["step"] <= count, "episode outside declared training assignment")
        sensor, profile = np.random.SeedSequence([row["worker_seed"], episode_index[rank]]).generate_state(2)
        require((row["sensor_seed"], row["profile_seed"]) == (sensor, profile), "episode seed streams changed")
        delay = row["dynamics"]["drive_delay_ticks"]
        noise = row["dynamics"]["scan_noise_std"]
        if settings["variant"] == "no_randomization":
            require(delay == 0 and noise == 0.03, "non-nominal profile in no_randomization")
        else:
            require(delay in (1, 2) and 0.0297 <= noise <= 0.0302, "profile outside measured envelope")
        require(row["disposition"] in OUTCOMES, "unknown training outcome")
        outcomes[row["disposition"]] += 1
        episode_index[rank] += 1
    require(outcomes.total() == stats["episodes"] and outcomes["invalid"] == stats["invalid_episodes"] == 0,
            "invalid or missing rollout episodes")
    return {
        "stage": number, "cumulative_transitions": count, "optimizer_updates": expected_updates,
        "outcomes": {name: outcomes[name] for name in OUTCOMES},
        "training_seconds": stats["training_seconds"], "checkpoint_sha256": checkpoint_hash,
        "episode_log_sha256": stats["episodes_sha256"],
    }


def verify_run(root, variant, seed, common, source):
    settings = read_json(root / "resolved.json")
    summary = read_json(root / "summary.json")
    bundle = read_json(root / "bundle/bundle.json")
    provenance = bundle["provenance"]
    for key, expected in (("variant", variant), ("seed", seed)):
        require(settings[key] == summary[key] == provenance[key] == expected, "variant/seed mismatch")
    started, finished = settings["started_at_utc"], summary["finished_at_utc"]
    require(utc_time(started) <= utc_time(bundle["created_at_utc"]) <= utc_time(finished), "invalid UTC run window")
    require(settings["stage_steps"] == STAGE_STEPS and settings["cpu_threads"] == 1 and settings["device"] == "cpu"
            and settings["ppo"]["n_envs"] == WORKERS, "unexpected training budget")
    require(settings["worker_seeds_by_stage"] == [[seed + 10000 * i + rank for rank in range(WORKERS)]
                                                  for i in range(3)], "worker seeds mismatch")
    history = "newest" if variant == "no_history" else "stack"
    require(settings["environment"]["history_mode"] == bundle["observation_contract"]["history_mode"] == history
            and settings["environment"]["shield_enabled"] is True
            and settings["randomization"] == ("nominal" if variant == "no_randomization" else "measured_variation"),
            "training variant contract mismatch")
    require(bundle["policy"].get("input_transform") == ("repeat_latest_scan" if variant == "no_history" else None),
            "exported preprocessing mismatch")
    shared = {field: settings[field] for field in (
        "git", "source", "dependencies", "curriculum_stages", "curriculum_manifest_sha256",
        "mission_manifest_sha256", "map_sha256", "policy_io_sha256", "m2_ppo_sha256")}
    shared["ppo"] = settings["ppo"]["ppo"]
    if not common:
        verify_source(source, settings)
        common.update(shared)
    require(common == shared, "frozen training implementation, inputs or PPO parameters changed between runs")
    require(settings["source"]["sha256"] == provenance["source_sha256"]
            and settings["git"]["commit"] == provenance["git_commit"]
            and settings["curriculum_manifest_sha256"] == provenance["curriculum_sha256"], "bundle source mismatch")
    settings_hash = sha256_file(root / "resolved.json")
    require(settings_hash == provenance["settings_sha256"], "bundle settings mismatch")
    require(sha256_file(root / "bundle/bundle.json") == summary["bundle_sha256"], "bundle manifest hash mismatch")
    check = PolicyRuntime.load(root / "bundle", verify=False)
    golden_error = check.verify_golden()
    with np.load(root / "bundle/golden.npz") as rows:
        samples = len(rows["observations"])
    require({"weights.npz", "golden.npz"} == set(bundle["files"]) and samples == summary["numpy_golden_samples"] == 32
            and 0 <= summary["numpy_max_abs_error"] <= GOLDEN_TOLERANCE
            and 0 <= golden_error <= GOLDEN_TOLERANCE, "failed golden actor verification")
    stages = read_json(root / "stages.json")
    require(len(stages) == 3 and stages == summary["stages"], "missing or inconsistent training stages")
    verified = [verify_stage(root, stats, settings, index) for index, stats in enumerate(stages)]
    require(summary["total_timesteps"] == provenance["total_timesteps"] == 3 * STAGE_STEPS
            and verified[-1]["checkpoint_sha256"] == summary["checkpoint_sha256"] == provenance["checkpoint_sha256"],
            "exported policy does not identify the final checkpoint")
    require(math.isfinite(summary["training_seconds"]) and summary["training_seconds"] > 0, "invalid wall time")
    return {
        "variant": variant, "seed": seed, "started_at_utc": started, "finished_at_utc": finished,
        "total_transitions": summary["total_timesteps"], "training_seconds": summary["training_seconds"],
        "settings_sha256": settings_hash, "summary_sha256": sha256_file(root / "summary.json"),
        "stages_sha256": sha256_file(root / "stages.json"), "bundle_sha256": summary["bundle_sha256"],
        "bundle_files_sha256": bundle["files"], "golden_samples": samples,
        "recorded_golden_max_abs_error": summary["numpy_max_abs_error"], "verified_golden_max_abs_error": golden_error,
        "stages": verified,
    }


def collect(runs, source):
    common = {}
    results = [verify_run(runs / variant / str(seed), variant, seed, common, source)
               for variant, seed in product(VARIANTS, SEEDS)]
    stage_totals = []
    for index in range(3):
        counts = Counter()
        for result in results:
            counts.update(result["stages"][index]["outcomes"])
        stage_totals.append({"stage": index + 1, "completed_training_episodes": counts.total(), "outcomes": counts})
    return {
        "format": "opticore-m4-training-matrix-v1", "run_layout": "<raw-root>/<variant>/<seed>/",
        "train_git_commit": common["git"]["commit"], "source_sha256": common["source"]["sha256"],
        **{key: value for key, value in common.items() if key.endswith("_sha256")},
        "dependencies": common["dependencies"], "ppo": common["ppo"], "seeds": SEEDS, "variants": VARIANTS,
        "workers_per_run": WORKERS, "transitions_per_stage": STAGE_STEPS,
        "total_collected_transitions": sum(row["total_transitions"] for row in results),
        "started_at_utc": min(row["started_at_utc"] for row in results),
        "finished_at_utc": max(row["finished_at_utc"] for row in results),
        "stage_totals": stage_totals, "runs": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="clean frozen training checkout")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = collect(args.runs, args.source)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Verified {len(report['runs'])} runs and {report['total_collected_transitions']:,} transitions.")


if __name__ == "__main__":
    main()
