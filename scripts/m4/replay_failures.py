"""Replay the original M4 static and head-on failures on the paired train conditions."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from evaluate import PROFILE_SEED, SENSOR_SEEDS, TRAIN_SEEDS
from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.envs.m3_variation import measured_variation
from amr_lite.envs.m3_warehouse import DynamicsProfile, M3WarehouseEnv
from amr_lite.experiments.manifest import REPOSITORY_ROOT, sha256_file


def replay(policy, mission, sensor_seed, profile):
    env = M3WarehouseEnv(mission, profile)
    with_actor = mission == "train-headon"
    events = {}
    if with_actor:
        # Attribute a raw ray to the actor only when it occludes nearer than the
        # static map. This diagnostic probe leaves the sensor and policy untouched.
        clip = env.geometry.clip_actor_rays

        def mark_first_return(static, ox, oy, dx, dy, max_range):
            result = clip(static, ox, oy, dx, dy, max_range)
            if "first_actor_ray_s" not in events and np.any(
                (result < static) & (result >= env.raw_spec.range_min) & (result <= 8.0)
            ):
                events["first_actor_ray_s"] = max(0.0, getattr(env, "elapsed", 0.0))
            return result

        env.geometry.clip_actor_rays = mark_first_return

    try:
        obs, info = env.reset(seed=sensor_seed)
        closest = None
        previous_xy = info["pose_lite"][:2]
        traveled = 0.0
        while True:
            start = info["elapsed"]
            v_before = env.robot.v
            obs, _, terminated, truncated, info = env.step(policy.predict(obs))
            if info["disposition"] == "invalid":
                raise ValueError(f"invalid replay: {info}")
            v = info["velocity"][0]
            xy = info["pose_lite"][:2]
            traveled += math.dist(previous_xy, xy)
            previous_xy = xy
            if closest is None or info["goal_error"] < closest["distance_m"]:
                closest = {"distance_m": info["goal_error"], "time_s": info["elapsed"],
                           "target_v": info["target"][0], "command_v": info["command"][0],
                           "speed_mps": v, "shield": info["shield_mode"]}
            seen = events.get("first_actor_ray_s")
            if seen is not None and start >= seen - 1e-8:
                # Control timestamps refer to decisions before the physics step.
                for key, active in (
                    ("first_lower_policy_target_s", info["target"][0] < v_before - 0.01),
                    ("first_lower_command_s", info["command"][0] < v_before - 0.01),
                    ("first_speed_drop_end_s", v < v_before - 0.01),
                ):
                    if active and key not in events:
                        events[key] = info["elapsed"] if key.endswith("end_s") else start
            if terminated or truncated:
                break
        result = {"outcome": info["disposition"], "elapsed_s": info["elapsed"],
                  "nearest": closest, "final_error_m": info["goal_error"], "distance_m": traveled,
                  "final_target_v": info["target"][0], "final_command_v": info["command"][0],
                  "final_speed_mps": v, "final_shield": info["shield_mode"], **events}
        if result["outcome"] == "collision":
            result["actor_contact"] = env.geometry.actor_clearance(*xy, env.radius) <= 0
            result["map_contact"] = env.geometry.grid.clearance(*xy, env.radius) <= 0
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--mixed-score", type=Path, help="score report for the three mixed-profile actors")
    args = parser.parse_args()
    root = REPOSITORY_ROOT / "experiments/m4/development"
    profiles = {"nominal": DynamicsProfile(), "measured": measured_variation(np.random.default_rng(PROFILE_SEED))}
    rows = []
    mixed_score = json.loads(args.mixed_score.read_text()) if args.mixed_score else None
    headon_variant = "full" if mixed_score else "no_randomization"
    for variant, mission in (("full", "train-static"), (headon_variant, "train-headon")):
        for seed in TRAIN_SEEDS:
            name = f"primary-{variant}-{seed}.json"
            if mixed_score:
                bundle = args.runs / str(seed) / "bundle"
                reference = mixed_score["reports"][name]
            else:
                bundle = args.runs / variant / str(seed) / "bundle"
                reference = json.loads((root / name).read_text())
            if reference["bundle_sha256"] != sha256_file(bundle / "bundle.json"):
                raise ValueError("replay bundle does not match recorded actor")
            policy = PolicyRuntime.load(bundle)
            for original in reference["rows"]:
                if original["mission"] != mission:
                    continue
                profile, sensor_seed = original["profile"], original["sensor_seed"]
                result = replay(policy, mission, sensor_seed, profiles[profile])
                if (result["outcome"] != original["outcome"] or
                    not math.isclose(result["elapsed_s"], original["elapsed_s"], abs_tol=1e-6) or
                    not math.isclose(result["nearest"]["distance_m"], original["nearest_goal_m"], abs_tol=1e-6)):
                    raise ValueError(f"replay differs for {variant}/{seed}/{profile}/{sensor_seed}")
                rows.append({"variant": variant, "mission": mission, "seed": seed,
                             "profile": profile, "sensor_seed": sensor_seed, **result})
    if len(rows) != 2 * len(TRAIN_SEEDS) * len(SENSOR_SEEDS) * len(profiles):
        raise ValueError("incomplete replay set")
    with args.out.open("x") as stream:
        json.dump({"finished_at_utc": datetime.now(timezone.utc).isoformat(), "rows": rows},
                  stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
