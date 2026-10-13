"""Deterministic, optionally parallel policy evaluation on an evaluation protocol.

Used for checkpoint selection during PPO v2 training, DAgger rounds and sweep
aggregation. Workers only host environments; the policy stays in this
process and acts on the batch of observations, so any object with
``predict`` (torch, NumPy runtime, rule baseline) can be evaluated.
"""
from __future__ import annotations

import math
import statistics
import time
from typing import Any, Sequence

import numpy as np

from amr_lite.config import deep_merge, load_config, resolve_env_config

from .vector_env import SlotSpec, make_vector_env


def load_protocol(name_or_config: str | dict[str, Any]) -> dict[str, Any]:
    return load_config(name_or_config) if isinstance(name_or_config, str) else dict(name_or_config)


def protocol_seeds(protocol: dict[str, Any], episodes_per_scenario: int | None = None) -> list[int]:
    if protocol.get("seeds") is not None:
        seeds = [int(seed) for seed in protocol["seeds"]]
        return seeds[:episodes_per_scenario] if episodes_per_scenario else seeds
    count = int(episodes_per_scenario or protocol["episodes_per_scenario"])
    return [int(protocol["seed"]) + offset for offset in range(count)]


def protocol_episodes(protocol: dict[str, Any], episodes_per_scenario: int | None = None
                      ) -> list[tuple[str, int]]:
    seeds = protocol_seeds(protocol, episodes_per_scenario)
    return [(str(scenario), seed) for scenario in protocol["scenarios"] for seed in seeds]


def protocol_env_config(protocol: dict[str, Any], robot_profile: str | None = None,
                        base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Environment for a protocol: its robot profile (or the caller's) + overrides."""
    profile = protocol.get("robot_profile") or robot_profile
    return resolve_env_config(profile, protocol.get("environment_overrides"), base=base)


def _predict(policy: Any, observations: np.ndarray) -> np.ndarray:
    batch = getattr(policy, "predict_batch", None)
    if batch is not None:
        return np.asarray(batch(observations, deterministic=True), dtype=np.float32)
    return np.stack([np.asarray(policy.predict(item, deterministic=True), dtype=np.float32)
                     for item in observations])


def run_episodes(policy: Any, episodes: Sequence[tuple[str, int]], env_config: dict[str, Any], *,
                 workers: int = 1, backend: str = "sync", shield_enabled: bool | None = None
                 ) -> list[dict[str, Any]]:
    """Run ``episodes`` deterministically; rows keep the order of ``episodes``.

    Episodes end by the environment's own termination or timeout.
    """
    config = dict(env_config)
    if shield_enabled is not None:
        config["shield_enabled"] = bool(shield_enabled)
    count = max(1, min(int(workers), len(episodes)))
    assignments = [list(episodes[index::count]) for index in range(count)]
    specs = [SlotSpec(config, index, episodes=assignment) for index, assignment in enumerate(assignments)]
    vector = make_vector_env(specs, backend)
    rows: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    inference = [0.0] * count
    steps = [0] * count
    try:
        observations = vector.reset()
        while True:
            active = [index for index in range(count) if len(rows[index]) < len(assignments[index])]
            if not active:
                break
            actions = np.zeros((count, 2), dtype=np.float32)
            start = time.perf_counter()
            actions[active] = _predict(policy, observations[active])
            elapsed = (time.perf_counter() - start) / len(active)
            observations, _, _, _, _, infos = vector.step(actions)
            for index in active:
                inference[index] += elapsed
                steps[index] += 1
                info = infos[index]
                if "final_observation" in info:
                    info = {key: value for key, value in info.items() if key != "final_observation"}
                    info["inference_ms"] = 1000.0 * inference[index] / max(1, steps[index])
                    rows[index].append(info)
                    inference[index], steps[index] = 0.0, 0
    finally:
        vector.close()
    ordered: list[dict[str, Any]] = []
    for position in range(len(episodes)):
        slot, item = position % count, position // count
        ordered.append(rows[slot][item])
    return ordered


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = max(1, len(rows))
    failures: dict[str, int] = {}
    for row in rows:
        key = str(row.get("failure_type") or ("SUCCESS" if row.get("success") else "UNFINISHED"))
        failures[key] = failures.get(key, 0) + 1
    interventions = [int(row["shield_counts"]["CLAMP"]) + int(row["shield_counts"]["STOP"]) for row in rows]
    lengths = [max(1, int(row.get("episode_length", 1))) for row in rows]
    return {
        "episodes": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / count,
        "collision_rate": sum(bool(row["collision"]) for row in rows) / count,
        "timeout_rate": sum(bool(row.get("truncated")) for row in rows) / count,
        "mean_clearance": statistics.mean(float(row["minimum_clearance_episode"]) for row in rows) if rows else math.nan,
        "mean_time": statistics.mean(float(row["elapsed_time"]) for row in rows) if rows else math.nan,
        "mean_cost": statistics.mean(float(row.get("episode_cost", 0.0)) for row in rows) if rows else math.nan,
        "shield_intervention_rate": sum(interventions) / sum(lengths) if rows else math.nan,
        "failure_counts": failures,
    }


def selection_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    """Pre-registered lexicographic checkpoint score (same order as M1)."""
    return (float(metrics["success_rate"]), -float(metrics["collision_rate"]),
            -float(metrics["timeout_rate"]), float(metrics["mean_clearance"]), -float(metrics["mean_time"]))


def evaluate_protocol(policy: Any, protocol: str | dict[str, Any], *, robot_profile: str | None = None,
                      env_config: dict[str, Any] | None = None, episodes_per_scenario: int | None = None,
                      workers: int = 1, backend: str = "sync") -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate metrics (per shield mode) and rows of a policy on a protocol."""
    spec = load_protocol(protocol)
    config = env_config if env_config is not None else protocol_env_config(spec, robot_profile)
    if env_config is not None and spec.get("environment_overrides"):
        config = deep_merge(env_config, spec["environment_overrides"])
    episodes = protocol_episodes(spec, episodes_per_scenario)
    rows, metrics = [], {}
    for shield in spec.get("shield_modes", [False]):
        shield_rows = run_episodes(policy, episodes, config, workers=workers, backend=backend,
                                   shield_enabled=bool(shield))
        for row in shield_rows:
            row["shield_enabled"] = bool(shield)
        rows += shield_rows
        metrics[f"shield_{'on' if shield else 'off'}"] = aggregate(shield_rows)
    first = metrics[f"shield_{'on' if spec.get('shield_modes', [False])[0] else 'off'}"]
    return {**first, "by_shield": metrics, "protocol": spec.get("protocol", "v1"),
            "split": spec.get("split", "")}, rows
