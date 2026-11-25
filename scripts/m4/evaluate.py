#!/usr/bin/env python3
"""Compare one verified policy on the fixed M4 development conditions."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import sys
from pathlib import Path

import numpy as np

from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.envs.m3_variation import measured_variation
from amr_lite.envs.m3_warehouse import (
    MANIFEST, MAP_ASSET, DynamicsProfile, M3WarehouseEnv, development_missions,
)
from amr_lite.experiments.manifest import REPOSITORY_ROOT, sha256_file
from amr_lite.learning.stamped_io import M3_IO_ID


SENSOR_SEEDS = (100, 200)
PROFILE_SEED = 20261123
PRIMARY = ("train-static", "train-crossing", "train-headon", "train-blockage")
SECONDARY = ("m4-train-multi", "m4-train-rejoin")
VARIANTS = ("m3", "full", "no_history", "no_randomization")
TRAIN_SEEDS = (7, 19, 31)


def conditions(suite):
    missions = PRIMARY if suite == "primary" else SECONDARY
    for mission in missions:
        for sensor_seed in SENSOR_SEEDS:
            for name in ("nominal", "measured"):
                yield mission, sensor_seed, name


def run_episode(policy, mission, sensor_seed, dynamics, shield, release_s):
    if mission in PRIMARY:
        env = M3WarehouseEnv(mission, dynamics)
        env.config["shield_enabled"] = shield
    elif mission in SECONDARY:
        from amr_lite.envs.m4_curriculum import M4WarehouseEnv

        env = M4WarehouseEnv(mission, dynamics, shield=shield)
    else:
        raise ValueError(f"unexpected development mission: {mission}")
    try:
        obs, info = env.reset(seed=sensor_seed)
        previous_xy = info["pose_lite"][:2]
        previous_t = info["elapsed"]
        nearest_goal = info["goal_error"]
        traveled = 0.0
        idle_run = 0.0
        longest_idle = 0.0
        stalled_seconds = 0.0
        cleared_xy = None
        stopped_at_release = None
        resumed_s = None
        ticks = 0
        while True:
            obs, _, terminated, truncated, info = env.step(policy.predict(obs))
            now = info["elapsed"]
            xy = info["pose_lite"][:2]
            traveled += math.dist(xy, previous_xy)
            nearest_goal = min(nearest_goal, info["goal_error"])
            if abs(info["velocity"][0]) < 0.05:
                idle_run += now - previous_t
                longest_idle = max(longest_idle, idle_run)
            else:
                if idle_run >= 2.0 - 1e-8:
                    stalled_seconds += idle_run
                idle_run = 0.0
            if release_s is not None and now >= release_s - 1e-8:
                if cleared_xy is None:
                    cleared_xy = xy
                    stopped_at_release = idle_run >= 2.0 - 1e-8
                elif stopped_at_release and resumed_s is None and math.dist(xy, cleared_xy) >= 0.25:
                    resumed_s = now - release_s
            previous_t, previous_xy = now, xy
            ticks += 1
            if terminated or truncated:
                break
        if idle_run >= 2.0 - 1e-8:
            stalled_seconds += idle_run
        return {
            "mission": mission, "sensor_seed": sensor_seed, "outcome": info["disposition"],
            "fault": info.get("fault"), "elapsed_s": now, "ticks": ticks,
            "goal_error_m": info["goal_error"], "nearest_goal_m": nearest_goal,
            "min_clearance_m": info["min_clearance"], "distance_m": traveled,
            "interventions": info["shield_interventions"], "stalled_s": stalled_seconds,
            "longest_idle_s": longest_idle, "stopped_at_release": stopped_at_release,
            "release_to_restart_s": resumed_s, "route_rejoined": info.get("route_rejoined"),
            "route_rejoin_s": info.get("route_rejoin_s"),
        }
    finally:
        env.close()


def evaluate(bundle, variant, shield, suite):
    if set(PRIMARY) != set(development_missions()):
        raise ValueError("expected the frozen M2 train mission set")
    policy = PolicyRuntime.load(bundle)
    manifest = policy.manifest
    if manifest["observation_contract"].get("acquisition_contract") != M3_IO_ID:
        raise ValueError("expected M3 stamped observation contract")
    provenance = manifest["provenance"]
    seed = provenance["seed"]
    if variant == "m3":
        reference = json.loads((REPOSITORY_ROOT / "experiments/m3/training-provenance.json").read_text())
        if sha256_file(bundle / "bundle.json") != reference["training"]["artifacts_sha256"]["bundle/bundle.json"]:
            raise ValueError("reference must be the M3 diagnostic actor")
    else:
        if provenance.get("total_timesteps") != 196608:
            raise ValueError("primary M4 results require the predeclared final 196,608-transition checkpoint")
        if seed not in TRAIN_SEEDS or provenance.get("variant") != variant:
            raise ValueError(f"unexpected training identity: seed={seed}, variant={provenance.get('variant')}")
    if not shield and variant != "full":
        raise ValueError("shield-off diagnostic requires the full-training checkpoint")
    profiles = {"nominal": DynamicsProfile(), "measured": measured_variation(np.random.default_rng(PROFILE_SEED))}
    release_times = {m["id"]: (m["actor"] or {}).get("remove_at")
                     for m in json.loads(MANIFEST.read_text())["missions"] if m["id"] in PRIMARY}
    started = datetime.now(timezone.utc).isoformat()
    rows = []
    for mission, sensor_seed, profile in conditions(suite):
        episode = run_episode(policy, mission, sensor_seed, profiles[profile], shield, release_times.get(mission))
        rows.append({"profile": profile, **episode})
    code_paths = {Path(module.__file__).resolve() for name, module in sys.modules.items()
                  if name.startswith("amr_lite.") and getattr(module, "__file__", None)}
    config_root = REPOSITORY_ROOT / "packages/amr_lite/configs"
    code_paths.update(config_root / name for name in ("env.yaml", "m2/ppo_start.yaml", "m2/policy_io.yaml",
                                                         "robot_profiles/opticore_gazebo.yaml"))
    code_paths.update(MAP_ASSET.parent.glob("*.yaml"))
    code_paths.add(REPOSITORY_ROOT / "experiments/m4/curriculum.json")
    fingerprints = {str(p.relative_to(REPOSITORY_ROOT)): sha256_file(p) for p in sorted(code_paths)}
    return {
        "schema": "opticore-m4-development-v1", "suite": suite, "variant": variant,
        "training_seed": seed, "shield": shield, "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": sha256_file(Path(__file__)), "dependencies_sha256": fingerprints,
        "bundle_sha256": sha256_file(bundle / "bundle.json"),
        "checkpoint_sha256": provenance["checkpoint_sha256"], "manifest_sha256": sha256_file(MANIFEST),
        "map_sha256": sha256_file(MAP_ASSET), "profile_seed": PROFILE_SEED,
        "profiles": {k: asdict(v) for k, v in profiles.items()}, "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--suite", choices=("primary", "secondary"), default="primary")
    parser.add_argument("--shield-off", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"output already exists: {args.out}")
    report = evaluate(args.bundle, args.variant, not args.shield_off, args.suite)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
