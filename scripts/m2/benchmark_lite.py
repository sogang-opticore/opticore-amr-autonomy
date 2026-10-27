"""Measure single-worker Lite development rollouts, including resets and Rule inference.

From the repository root run
PYTHONPATH=packages/amr_lite python scripts/m2/benchmark_lite.py > /tmp/lite-throughput.json
"""

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml
from amr_lite.config import ROOT, resolve_env_config
from amr_lite.envs.warehouse_env import WarehouseEnv
from amr_lite.planning.rule_baseline import RuleBasedPlanner


STEPS_PER_TRIAL = 4096
TRAIN_SEEDS = (1_000_000, 1_010_000, 1_020_000)


def run_trial(env, policy, seed):
    # The seed ranges are spaced wider than the longest possible number of
    # resets (one per step), so deterministic trials cannot reuse an episode.
    episode_seed = seed
    finished = 0
    start = perf_counter()
    obs, _ = env.reset(seed=episode_seed)
    initial_id = env.scenario.instance_id
    for step in range(STEPS_PER_TRIAL):
        obs, _, terminated, truncated, _ = env.step(policy.predict(obs))
        if terminated or truncated:
            finished += 1
            if step + 1 < STEPS_PER_TRIAL:
                episode_seed += 1
                obs, _ = env.reset(seed=episode_seed)
    wall = perf_counter() - start
    return {
        "first_seed": seed,
        "last_seed": episode_seed,
        "initial_instance": initial_id,
        "completed_episodes": finished,
        "steps": STEPS_PER_TRIAL,
        "wall_seconds": wall,
        "steps_per_wall_second": STEPS_PER_TRIAL / wall,
    }


def main():
    started_at = datetime.now(timezone.utc).isoformat()
    scenario_results = []
    for scenario in ("WH-RANDOM", "WHD-RANDOM"):
        config = resolve_env_config("opticore_gazebo", {
            "seed": TRAIN_SEEDS[0],
            "scenario": scenario,
            "procedural": {"split": "train"},
            "shield_enabled": False,
        })
        start = perf_counter()
        env = WarehouseEnv(config, scenario_id=scenario)
        policy = RuleBasedPlanner(config)
        init_wall = perf_counter() - start
        try:
            samples = [run_trial(env, policy, seed) for seed in TRAIN_SEEDS]
        finally:
            env.close()
        wall = sum(trial["wall_seconds"] for trial in samples)
        steps = sum(trial["steps"] for trial in samples)
        scenario_results.append({
            "scenario": scenario,
            "effective_env_config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "initialization_wall_seconds": init_wall,
            "trials": samples,
            "pooled_steps_per_wall_second": steps / wall,
            "projected_minutes_per_100k_rule_steps": 100_000 * wall / steps / 60,
        })

    code_tree = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD:packages/amr_lite/amr_lite"],
        text=True,
    ).strip()
    map_file = ROOT / "assets/maps/opticore_warehouse/warehouse_map.pgm"
    print(json.dumps({
        "started_at_utc": started_at,
        "protocol": "m2-lite-throughput-v1",
        "reproduce": "PYTHONPATH=packages/amr_lite python scripts/m2/benchmark_lite.py",
        "system": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyyaml": yaml.__version__,
        "amr_lite_source_tree": code_tree,
        "map_sha256": hashlib.sha256(map_file.read_bytes()).hexdigest(),
        "profile": "opticore_gazebo",
        "legacy_shield_enabled": False,
        "scenario_split": "train",
        "workers": 1,
        "measurement": "One initial reset and all terminal resets inside each timed window, "
                       "serial Rule prediction plus WarehouseEnv.step. Constructor time "
                       "is reported separately. Map/path caches persist in-process.",
        "budget_scope": "Linear projection for this Rule + Lite rollout workload. PPO "
                        "inference, gradient updates, evaluation, Gazebo and job startup "
                        "require their own measurement.",
        "results": scenario_results,
    }, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
