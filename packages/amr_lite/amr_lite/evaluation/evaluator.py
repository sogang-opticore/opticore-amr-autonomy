from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from amr_lite.config import ROOT, deep_merge, load_config, resolve_env_config
from amr_lite.envs import WarehouseEnv

from .metrics import (
    paired_comparisons,
    summarize,
    summarize_by_scenario,
    summarize_by_training_seed,
    write_csv,
)
from .plots import make_plots


def _sequential_rows(policies: dict[str, Any], cfg: dict, base_env_config: dict,
                     episode_seeds: list[int]) -> list[dict]:
    rows = []
    total_runs = (len(cfg["scenarios"]) * len(episode_seeds)
                  * len(cfg["shield_modes"]) * len(policies))
    progress_every = int(cfg.get("progress_every", 0))
    completed_runs = 0
    for scenario in cfg["scenarios"]:
        for episode_seed in episode_seeds:
            for shield_enabled in cfg["shield_modes"]:
                for policy_name, policy in policies.items():
                    local_config = deep_merge(base_env_config, getattr(policy, "observation_overrides", None))
                    local_config["shield_enabled"] = bool(shield_enabled)
                    env = WarehouseEnv(local_config, scenario)
                    observation, _ = env.reset(seed=episode_seed, options={"scenario_id": scenario})
                    actions, inference_seconds, safety_cost = [], 0.0, 0.0
                    wall_start = time.perf_counter()
                    terminated = truncated = False
                    info: dict[str, Any] = {}
                    for _ in range(int(cfg["max_steps"])):
                        start = time.perf_counter()
                        action = np.asarray(policy.predict(observation, deterministic=True), dtype=np.float32)
                        inference_seconds += time.perf_counter() - start
                        actions.append(action)
                        observation, _, terminated, truncated, info = env.step(action)
                        safety_cost += float(info.get("safety_cost", 0.0))
                        if terminated or truncated:
                            break
                    wall = max(1e-9, time.perf_counter() - wall_start)
                    diffs = np.diff(np.asarray(actions), axis=0) if len(actions) > 1 else np.zeros((1, 2))
                    rows.append(episode_row(
                        policy_name, policy, cfg, local_config, scenario, episode_seed, bool(shield_enabled),
                        info, terminated=terminated, truncated=truncated, steps=len(actions),
                        action_smoothness=float(np.mean(np.sum(diffs * diffs, axis=1))),
                        safety_cost=safety_cost,
                        inference_ms=1000.0 * inference_seconds / max(1, len(actions)),
                        simulation_steps_per_second=len(actions) / wall))
                    completed_runs += 1
                    if progress_every > 0 and (
                            completed_runs % progress_every == 0 or completed_runs == total_runs):
                        print(f"evaluation progress {completed_runs}/{total_runs}", flush=True)
                    env.close()
    return rows


def episode_row(policy_name: str, policy: Any, cfg: dict, local_config: dict, scenario: str,
                episode_seed: int, shield_enabled: bool, info: dict[str, Any], *, terminated: bool,
                truncated: bool, steps: int, action_smoothness: float, safety_cost: float,
                inference_ms: float, simulation_steps_per_second: float) -> dict[str, Any]:
    """One episodes.csv row (shared by the sequential and parallel evaluators)."""
    interventions = info["shield_counts"]["CLAMP"] + info["shield_counts"]["STOP"]
    return {
        "policy": policy_name, "scenario_id": scenario, "seed": episode_seed,
        "policy_family": getattr(policy, "policy_family", policy_name),
        "training_seed": getattr(policy, "training_seed", ""),
        "checkpoint_role": getattr(policy, "checkpoint_role", "fixed"),
        "checkpoint_sha256": getattr(policy, "checkpoint_sha256", ""),
        "evaluation_split": cfg.get("split", "unspecified"),
        "scenario_instance_id": info.get("scenario_instance_id", ""),
        "shield_strategy": local_config.get("shield_strategy", "legacy"),
        "shield_enabled": bool(shield_enabled), "success": int(info.get("success", False)),
        "collision": int(info.get("collision", False)), "terminated": int(terminated),
        "truncated": int(truncated), "failure_type": info.get("failure_type", "ENVIRONMENT_ERROR"),
        "episode_time": float(info.get("elapsed_time", 0.0)), "steps": steps,
        "path_length": float(info.get("path_length", 0.0)),
        "path_length_ratio": float(info.get("path_length_ratio", 0.0)),
        "minimum_clearance": float(info.get("minimum_clearance_episode", 0.0)),
        "mean_cross_track_error": float(info.get("mean_cross_track_error", 0.0)),
        "max_cross_track_error": float(info.get("max_cross_track_error", 0.0)),
        "idle_time": float(info.get("idle_time", 0.0)),
        "angular_oscillation": int(info.get("angular_oscillation", 0)),
        "total_abs_rotation": float(info.get("total_abs_rotation", 0.0)),
        "action_smoothness": action_smoothness,
        "shield_pass": info["shield_counts"]["PASS"], "shield_clamp": info["shield_counts"]["CLAMP"],
        "shield_stop": info["shield_counts"]["STOP"], "shield_interventions": interventions,
        "shield_slow": info.get("shield_action_counts", {}).get("SLOW", 0),
        "shield_evade": info.get("shield_action_counts", {}).get("EVADE", 0),
        "shield_reverse": info.get("shield_action_counts", {}).get("REVERSE", 0),
        "unsafe_without_shield": int(info.get("unsafe_without_shield", 0)),
        "shield_sustained_time": float(info.get("shield_sustained_time", 0.0)),
        "shield_false_positive": int(info.get("shield_false_positive", 0)),
        "shield_false_negative": int(info.get("shield_false_negative", 0)),
        "shield_actual_false_negative": int(info.get("shield_actual_false_negative", 0)),
        "shield_ineffective_intervention": int(info.get("shield_ineffective_intervention", 0)),
        "tracking_speed_error_mean": float(info.get("tracking_speed_error_mean", 0.0)),
        "tracking_speed_error_max": float(info.get("tracking_speed_error_max", 0.0)),
        "minimum_predicted_ttc": float(info.get("minimum_predicted_ttc", float("inf"))),
        "shield_risk_counts": json.dumps(info.get("shield_risk_counts", {}), sort_keys=True),
        "safety_cost": safety_cost,
        "inference_ms": inference_ms,
        "simulation_steps_per_second": simulation_steps_per_second,
    }


def _evaluation_seeds(cfg: dict) -> list[int]:
    return ([int(seed) for seed in cfg["seeds"]] if cfg.get("seeds") is not None
            else [int(cfg["seed"]) + episode for episode in range(int(cfg["episodes_per_scenario"]))])


def _parallel_rows(policies: dict[str, Any], cfg: dict, base_env_config: dict,
                   episode_seeds: list[int]) -> list[dict]:
    """Same rows as the sequential loop, computed with lockstep worker envs."""
    from amr_lite.rl.evaluation import run_episodes

    episodes = [(scenario, seed) for scenario in cfg["scenarios"] for seed in episode_seeds]
    results: dict[tuple, dict] = {}
    workers, backend = int(cfg.get("workers", 1)), str(cfg.get("backend", "subprocess"))
    for shield_enabled in cfg["shield_modes"]:
        for policy_name, policy in policies.items():
            infos = run_episodes(policy, episodes, base_env_config, workers=workers, backend=backend,
                                 shield_enabled=bool(shield_enabled))
            for (scenario, seed), info in zip(episodes, infos):
                results[(scenario, seed, bool(shield_enabled), policy_name)] = info
            if int(cfg.get("progress_every", 0)) > 0:
                print(f"evaluation progress {policy_name} shield={bool(shield_enabled)} "
                      f"{len(episodes)} episodes", flush=True)
    rows = []
    for scenario in cfg["scenarios"]:
        for seed in episode_seeds:
            for shield_enabled in cfg["shield_modes"]:
                for policy_name, policy in policies.items():
                    local_config = deep_merge(base_env_config, getattr(policy, "observation_overrides", None))
                    local_config["shield_enabled"] = bool(shield_enabled)
                    info = results[(scenario, seed, bool(shield_enabled), policy_name)]
                    rows.append(episode_row(
                        policy_name, policy, cfg, local_config, scenario, seed, bool(shield_enabled), info,
                        terminated=bool(info["terminated"]), truncated=bool(info["truncated"]),
                        steps=int(info["episode_length"]), action_smoothness=float(info["action_smoothness"]),
                        safety_cost=float(info["episode_cost"]), inference_ms=float(info["inference_ms"]),
                        simulation_steps_per_second=float(info["simulation_steps_per_second"])))
    return rows


def evaluate(policies: dict[str, Any], output_dir: str | Path | None = None,
             config: dict | None = None, env_config: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Paired evaluation of policies on every scenario, seed and shield mode.

    ``robot_profile`` in the evaluation config selects the robot (protocol
    v2); without it the base is ``env.yaml`` as in protocol v1. ``workers``
    > 1 runs episodes in parallel worker processes with identical results
    except for timing columns.
    """
    cfg = config or load_config("evaluation")
    if cfg.get("robot_profile"):
        base_env_config = resolve_env_config(cfg["robot_profile"], cfg.get("environment_overrides"),
                                             base=env_config)
    else:
        base_env_config = deep_merge(env_config or load_config("env"), cfg.get("environment_overrides"))
    output = Path(output_dir or ROOT / "artifacts/results")
    episode_seeds = _evaluation_seeds(cfg)
    if int(cfg.get("workers", 1)) > 1:
        rows = _parallel_rows(policies, cfg, base_env_config, episode_seeds)
    else:
        rows = _sequential_rows(policies, cfg, base_env_config, episode_seeds)
    bootstrap_resamples = int(cfg.get("bootstrap_resamples", 2000))
    bootstrap_seed = int(cfg.get("bootstrap_seed", 20260914))
    summary = summarize(rows, bootstrap_resamples, bootstrap_seed)
    seed_summary = summarize_by_training_seed(rows)
    scenario_summary = summarize_by_scenario(
        rows, bootstrap_resamples, bootstrap_seed
    )
    comparisons = paired_comparisons(
        rows, bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed
    )
    write_csv(output / "episodes.csv", rows)
    write_csv(output / "summary.csv", summary)
    write_csv(output / "summary_by_training_seed.csv", seed_summary)
    write_csv(output / "summary_by_scenario.csv", scenario_summary)
    if comparisons:
        write_csv(output / "paired_comparisons.csv", comparisons)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({"evaluation": cfg, "environment": base_env_config}, indent=2), encoding="utf-8")
    make_plots(output / "summary.csv", output.parent / "plots")
    return rows, summary
