"""Train one three-stage SB3 development policy and export its final actor."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from amr_lite.cli.train_m3 import CaptureRollouts
from amr_lite.config import ROOT, load_config
from amr_lite.deploy.sb3 import export_sb3
from amr_lite.envs.m3_variation import measured_variation
from amr_lite.envs.m3_warehouse import MANIFEST, MAP_ASSET, DynamicsProfile, environment_config
from amr_lite.envs.m4_curriculum import CURRICULUM_MANIFEST, M4WarehouseEnv, curriculum_stages
from amr_lite.experiments.manifest import dependency_snapshot, git_snapshot, sha256_file, source_snapshot
from amr_lite.learning.observation import repeat_latest_scan

VARIANTS = ("full", "no_history", "no_randomization")


class CurriculumWorker(gym.Wrapper):
    """Cycle through a stage's train missions with paired, per-episode RNG streams."""

    def __init__(self, missions, rank: int, variant: str, mix_profiles: bool = False):
        if mix_profiles and variant != "full":
            raise ValueError("mixed profiles require the full-history development configuration")
        self.missions = missions
        self.rank = rank
        self.history = variant != "no_history"
        self.randomize = variant != "no_randomization"
        self.mix_profiles = mix_profiles
        self.worker_seed = None
        self.episode = 0
        super().__init__(M4WarehouseEnv(missions[rank % len(missions)]))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.worker_seed = int(seed)
            self.episode = 0
        if self.worker_seed is None:
            raise ValueError("the vector runner must seed each new curriculum worker")
        sensor_seed, profile_seed, mixture_draw = map(int, np.random.SeedSequence(
            [self.worker_seed, self.episode]).generate_state(3))
        measured = self.randomize and (not self.mix_profiles or mixture_draw % 2 == 1)
        dynamics = measured_variation(np.random.default_rng(profile_seed)) if measured else DynamicsProfile()
        mission = self.missions[(self.rank + self.episode) % len(self.missions)]
        self.env.close()
        self.env = M4WarehouseEnv(mission, dynamics, history=self.history, shield=True)
        self.episode_spec = {"worker_seed": self.worker_seed, "episode_index": self.episode,
                             "sensor_seed": sensor_seed, "profile_seed": profile_seed,
                             "dynamics": asdict(dynamics)}
        self.episode += 1
        return self.env.reset(seed=sensor_seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            info.update(self.episode_spec)
        return obs, reward, terminated, truncated, info


def golden_observations(model, variant):
    """Collect raw, stamped observations for a deploy-path check, including for no_history."""
    observations = []
    env = M4WarehouseEnv(curriculum_stages()[0][0], history=True)
    try:
        obs, _ = env.reset(seed=730)
        for _ in range(32):
            observations.append(obs)
            sample = repeat_latest_scan(obs) if variant == "no_history" else obs
            action, _ = model.predict(sample, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if info["disposition"] == "invalid":
                raise RuntimeError(f"invalid golden probe: {info['fault']}")
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return np.stack(observations)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def train(output: Path, variant: str, seed: int, stage_steps: int, mix_profiles: bool = False):
    if variant not in VARIANTS or seed < 0:
        raise ValueError("choose a valid training variant and nonnegative seed")
    if mix_profiles and variant != "full":
        raise ValueError("mixed profiles require the full-history development configuration")
    ppo = load_config("m2/ppo_start")
    dependencies = dependency_snapshot()
    for name, version in ppo["versions"].items():
        if dependencies[name] != version:
            raise RuntimeError(f"{name} {version} required, found {dependencies[name]}")
    workers = ppo["n_envs"]
    rollout = workers * ppo["ppo"]["n_steps"]
    if stage_steps <= 0 or stage_steps % rollout:
        raise ValueError(f"stage steps must be a positive multiple of {rollout}")
    stages = curriculum_stages()
    config = environment_config()
    config["history_mode"] = "newest" if variant == "no_history" else "stack"
    ppo["seed"] = seed
    ppo["environment_overrides"]["history_mode"] = config["history_mode"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    settings = {
        "format": "opticore-m4-train-v1", "variant": variant, "seed": seed,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", "amr_lite.cli.train_m4", *sys.argv[1:]],
        "git": git_snapshot(), "source": source_snapshot(), "dependencies": dependencies,
        "platform": platform.platform(), "cpu_threads": torch.get_num_threads(), "device": "cpu",
        "policy_io_sha256": sha256_file(ROOT / "configs/m2/policy_io.yaml"),
        "m2_ppo_sha256": sha256_file(ROOT / "configs/m2/ppo_start.yaml"),
        "mission_manifest_sha256": sha256_file(MANIFEST), "map_sha256": sha256_file(MAP_ASSET),
        "curriculum_manifest_sha256": sha256_file(CURRICULUM_MANIFEST),
        "ppo": ppo, "environment": config, "curriculum_stages": stages, "stage_steps": stage_steps,
        "randomization": ("nominal_measured_50_50" if mix_profiles else
                          "measured_variation" if variant != "no_randomization" else "nominal"),
        "worker_seeds_by_stage": [[seed + 10000 * index + rank for rank in range(workers)]
                                  for index in range(len(stages))],
    }
    write_json(output / "resolved.json", settings)
    stages_completed = []
    model = None
    start = time.monotonic()
    for index, missions in enumerate(stages):
        vec = DummyVecEnv([lambda rank=rank: Monitor(CurriculumWorker(missions, rank, variant, mix_profiles))
                           for rank in range(workers)])
        stage = index + 1
        stage_start = time.monotonic()
        try:
            if model is None:
                model = PPO(env=vec, seed=seed, device="cpu", verbose=0, **ppo["ppo"])
            else:
                model.set_env(vec)
            vec.seed(settings["worker_seeds_by_stage"][index][0])
            with (output / f"stage-{stage}-episodes.jsonl").open("x") as stream:
                # Stage boundaries select checkpoints; keep the M3 invalid-transition guard.
                capture = CaptureRollouts(stream, output, checkpoint_every=stage_steps * len(stages) + 1)
                model.learn(total_timesteps=stage_steps, callback=capture, reset_num_timesteps=index == 0)
            if model.num_timesteps != stage * stage_steps:
                raise RuntimeError("unexpected number of collected transitions")
            checkpoint = output / ("final.zip" if stage == len(stages) else f"stage-{stage}.zip")
            model.save(checkpoint)
            stats = {"stage": stage, "total_timesteps": model.num_timesteps,
                     "optimizer_updates": model._n_updates, "episodes": capture.episodes,
                     "invalid_episodes": capture.invalid, "training_seconds": time.monotonic() - stage_start,
                     "checkpoint": checkpoint.name, "checkpoint_sha256": sha256_file(checkpoint),
                     "episodes_sha256": sha256_file(output / f"stage-{stage}-episodes.jsonl")}
            stages_completed.append(stats)
            write_json(output / "stages.json", stages_completed)
            print(json.dumps(stats), flush=True)
        finally:
            vec.close()

    elapsed = time.monotonic() - start
    checkpoint = output / "final.zip"
    restored = PPO.load(checkpoint, device="cpu")
    raw = golden_observations(restored, variant)
    samples = repeat_latest_scan(raw) if variant == "no_history" else raw
    np.testing.assert_array_equal(model.predict(samples, deterministic=True)[0],
                                  restored.predict(samples, deterministic=True)[0])
    error = export_sb3(restored, checkpoint, output / "bundle", config, raw, {
        "seed": seed, "variant": variant, "total_timesteps": model.num_timesteps,
        "randomization": settings["randomization"],
        "settings_sha256": sha256_file(output / "resolved.json"), "git_commit": settings["git"]["commit"],
        "source_sha256": settings["source"]["sha256"], "curriculum_sha256": settings["curriculum_manifest_sha256"],
    })
    result = {"variant": variant, "seed": seed, "stages": stages_completed,
              "finished_at_utc": datetime.now(timezone.utc).isoformat(), "total_timesteps": model.num_timesteps,
              "training_seconds": elapsed, "transitions_per_second": model.num_timesteps / elapsed,
              "checkpoint_sha256": sha256_file(checkpoint),
              "bundle_sha256": sha256_file(output / "bundle/bundle.json"),
              "numpy_golden_samples": len(raw), "numpy_max_abs_error": error}
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new directory for this run")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-steps", type=int, default=65536, help="transitions per stage, across all workers")
    parser.add_argument("--mix-profiles", action="store_true",
                        help="sample nominal or measured dynamics with equal probability per full-policy episode")
    args = parser.parse_args()
    train(args.out, args.variant, args.seed, args.stage_steps, args.mix_profiles)


if __name__ == "__main__":
    main()
