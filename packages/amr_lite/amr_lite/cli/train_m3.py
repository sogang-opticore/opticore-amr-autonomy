"""Exercise the pinned SB3 PPO path on the M3 development warehouse."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from amr_lite.config import ROOT, load_config
from amr_lite.deploy.sb3 import export_sb3
from amr_lite.envs.m3_warehouse import (
    MANIFEST, MAP_ASSET, DynamicsProfile, M3WarehouseEnv, development_missions, environment_config,
)
from amr_lite.experiments.manifest import dependency_snapshot, git_snapshot, sha256_file, source_snapshot
from amr_lite.learning.stamped_io import M3_IO_ID


class CaptureRollouts(BaseCallback):
    """Keep episode accounting and abort on any corrupt sampled transition."""

    def __init__(self, stream, output, checkpoint_every):
        super().__init__()
        self.stream = stream
        self.output = output
        self.checkpoint_every = checkpoint_every
        self.episodes = 0
        self.invalid = 0

    def _on_step(self) -> bool:
        for index, info in enumerate(self.locals["infos"]):
            if self.locals["dones"][index]:
                self.episodes += 1
                self.invalid += int(info["disposition"] == "invalid")
                record = {"step": self.num_timesteps, "worker": index, **info}
                self.stream.write(json.dumps(record, allow_nan=False) + "\n")
                self.stream.flush()
                if info["disposition"] == "invalid":
                    raise RuntimeError(f"invalid training transition at step {self.num_timesteps}: {info['fault']}")
        return True

    def _on_rollout_start(self):
        # A new rollout starts after the previous PPO parameter update.
        if self.num_timesteps and self.num_timesteps % self.checkpoint_every == 0:
            self.model.save(self.output / f"step-{self.num_timesteps:06}.zip")


def diagnose(model, output: Path, label: str, seed: int):
    observations = []
    rows = []
    with (output / f"{label}-trace.jsonl").open("x") as stream:
        for index, mission in enumerate(development_missions()):
            env = M3WarehouseEnv(mission)
            mission_seed = seed + index
            obs, info = env.reset(seed=mission_seed)
            done = False
            reward_sum = 0.0
            tick = 0
            nearest_goal = info["goal_error"]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                if tick < 16:
                    observations.append(obs)
                post, reward, terminated, truncated, info = env.step(action)
                reward_sum += reward
                done = terminated or truncated
                nearest_goal = min(nearest_goal, info["goal_error"])
                stream.write(json.dumps({"tick": tick, "seed": mission_seed, "action": action.tolist(),
                                         "reward": reward, **info}, allow_nan=False) + "\n")
                obs = post
                tick += 1
            row = {"seed": mission_seed, "ticks": tick, "reward": reward_sum,
                   "nearest_goal_m": nearest_goal, **info}
            rows.append(row)
            env.close()
            print(label, mission, info["disposition"], f"t={info['elapsed']:.3f}",
                  f"reward={reward_sum:.3f}", flush=True)
    return rows, np.stack(observations)


def train(output: Path, steps: int, resume: Path | None = None):
    ppo = load_config("m2/ppo_start")
    for distribution, version in ppo["versions"].items():
        actual = importlib.metadata.version(distribution)
        if actual != version:
            raise RuntimeError(f"{distribution} {version} is required, found {actual}")
    if (ppo["policy_io"] != M3_IO_ID or ppo["normalize_observations"] or ppo["normalize_rewards"]
            or ppo["n_envs"] != 4):
        raise ValueError("expected M3 stamped, fixed-scale observations and four PPO workers")
    rollout_size = ppo["ppo"]["n_steps"] * ppo["n_envs"]
    if steps < rollout_size or steps % rollout_size:
        raise ValueError(f"--steps must be a positive multiple of the {rollout_size}-transition rollout")
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dependencies = dependency_snapshot()
    config = environment_config()
    profiles = {
        "format": "opticore-m3-train-v1", "started_at_utc": started.isoformat(),
        "command": [sys.executable, "-m", "amr_lite.cli.train_m3", *sys.argv[1:]],
        "seed": ppo["seed"], "cpu_threads": torch.get_num_threads(), "device": "cpu",
        "dependencies": dependencies, "python": platform.python_version(),
        "machine": platform.machine(), "platform": platform.system(),
        "source": source_snapshot(), "git": git_snapshot(),
        "policy_io_sha256": sha256_file(ROOT / "configs/m2/policy_io.yaml"),
        "ppo_start_sha256": sha256_file(ROOT / "configs/m2/ppo_start.yaml"),
        "mission_manifest_sha256": sha256_file(MANIFEST), "map_sha256": sha256_file(MAP_ASSET),
        "training_mission": "train-static", "dynamics": asdict(DynamicsProfile()),
        "worker_seeds": [ppo["seed"] + rank for rank in range(ppo["n_envs"])],
        "ppo": ppo, "environment": config, "requested_transitions": steps,
        "normalization": {"observations": "fixed_contract_scales", "rewards": "none"},
    }
    if resume:
        prior = json.loads((resume.parent / "resolved.json").read_text())
        for field in ("format", "policy_io_sha256", "ppo_start_sha256", "mission_manifest_sha256",
                      "map_sha256", "environment", "training_mission", "dynamics", "normalization"):
            if prior[field] != profiles[field]:
                raise ValueError(f"cannot resume checkpoint with different {field}")
        profiles["resume"] = {"checkpoint": str(resume.resolve()), "sha256": sha256_file(resume),
                              "settings_sha256": sha256_file(resume.parent / "resolved.json"),
                              "environment_restart": True}
    settings = output / "resolved.json"
    settings.write_text(json.dumps(profiles, indent=2, allow_nan=False) + "\n")

    check_env(M3WarehouseEnv(), warn=True, skip_render_check=True)
    vec = DummyVecEnv([lambda: Monitor(M3WarehouseEnv()) for _ in range(ppo["n_envs"])])
    try:
        if resume:
            model = PPO.load(resume, env=vec, device="cpu")
        else:
            model = PPO(env=vec, seed=ppo["seed"], device="cpu", verbose=1, **ppo["ppo"])
        initial_step = model.num_timesteps
        if initial_step % rollout_size:
            raise ValueError("resume checkpoint is outside the configured rollout boundary")
        before, _ = diagnose(model, output, "before", seed=100)
        clock = time.monotonic()
        with (output / "episodes.jsonl").open("x") as stream:
            capture = CaptureRollouts(stream, output, max(rollout_size, 8192))
            model.learn(total_timesteps=steps, callback=capture, log_interval=1,
                        reset_num_timesteps=resume is None)
        elapsed = time.monotonic() - clock
        checkpoint = output / "final.zip"
        model.save(checkpoint)
    finally:
        vec.close()

    restored = PPO.load(checkpoint, device="cpu")
    if restored.num_timesteps != initial_step + steps:
        raise RuntimeError("PPO collected a different number of transitions than requested")
    after, samples = diagnose(restored, output, "after", seed=100)
    expected, _ = model.predict(samples, deterministic=True)
    reloaded, _ = restored.predict(samples, deterministic=True)
    if not np.array_equal(expected, reloaded):
        raise RuntimeError("SB3 deterministic action changed after save/load")
    difference = export_sb3(restored, checkpoint, output / "bundle", config, samples, {
        "seed": ppo["seed"], "total_timesteps": restored.num_timesteps,
        "settings_sha256": sha256_file(settings), "git_commit": profiles["git"]["commit"],
        "source_sha256": profiles["source"]["sha256"],
    })
    result = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(), "gymnasium_sb3_check_env": "passed",
        "total_timesteps": restored.num_timesteps, "initial_timesteps": initial_step,
        "training_seconds": elapsed, "transitions_per_second": steps / elapsed,
        "completed_training_episodes": capture.episodes, "invalid_training_episodes": capture.invalid,
        "fixed_diagnostic_seed_base": 100, "before": before, "after": after,
        "checkpoints": {p.name: sha256_file(p) for p in sorted(output.glob("*.zip"))},
        "checkpoint_sha256": sha256_file(checkpoint), "bundle_sha256": sha256_file(output / "bundle/bundle.json"),
        "numpy_golden_samples": len(samples), "numpy_max_abs_error": difference,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="new directory for M3 training evidence")
    parser.add_argument("--steps", type=int, default=16384, help="total transitions across all four workers")
    parser.add_argument("--resume", type=Path, help="prior M3 checkpoint next to its resolved.json")
    args = parser.parse_args()
    train(args.out, args.steps, args.resume)


if __name__ == "__main__":
    main()
