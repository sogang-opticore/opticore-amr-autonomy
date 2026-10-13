"""Lockstep vector environments with curriculum sampling.

Each slot owns one ``WarehouseEnv`` and picks its next episode itself, either
by sampling the active curriculum stage (training) or from a fixed episode
list (evaluation). Slots draw from their own seeded streams and all slots
step together, so ``SyncVectorEnv`` and ``SubprocVectorEnv`` produce identical
results for the same specs.
"""
from __future__ import annotations

import multiprocessing as mp
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from amr_lite.config import deep_merge

# Training episode seeds start here so they never overlap the selection
# (20000+) and test (30000+) namespaces of the evaluation protocols.
TRAINING_SEED_BASE = 1_000_000
_SEED_SPAN = 2**31 - 1 - TRAINING_SEED_BASE
# Per-step info is large; only episode-end info (minus reward terms) crosses
# process boundaries.
_DROPPED_INFO_KEYS = frozenset({"reward_components", "robot_pose", "sensor_model"})


@dataclass
class SlotSpec:
    """Everything a slot needs; must stay picklable for subprocess workers."""

    env_config: dict[str, Any]
    index: int
    training_seed: int = 0
    curriculum: list[dict[str, Any]] | None = None
    episodes: list[tuple[str, int]] | None = None
    episode_overrides: list[dict[str, Any] | None] | None = field(default=None)


def stage_for(curriculum: Sequence[dict[str, Any]], progress: float) -> dict[str, Any]:
    for stage in curriculum:
        if progress < float(stage.get("until", 1.0)):
            return stage
    return curriculum[-1]


def training_episode_seed(training_seed: int, slot: int, counter: int) -> int:
    state = np.random.SeedSequence([int(training_seed) % 2**32, slot, counter]).generate_state(1)[0]
    return TRAINING_SEED_BASE + int(state) % _SEED_SPAN


class EpisodeSlot:
    """One environment slot with automatic reset."""

    def __init__(self, spec: SlotSpec):
        from amr_lite.envs import WarehouseEnv

        self._env_type = WarehouseEnv
        self.spec = spec
        self.progress = 0.0
        self.counter = 0
        self.exhausted = False
        self.env = None
        self._env_key: str | None = None
        self.episode_return = 0.0
        self.episode_cost = 0.0
        self.episode_length = 0
        self.actions: list[np.ndarray] = []
        self.env_seconds = 0.0
        self.current: tuple[str, int, str] = ("", 0, "")
        self._rng = np.random.default_rng(np.random.SeedSequence(
            [int(spec.training_seed) % 2**32, spec.index, zlib.crc32(b"scenario-sampler")]))

    def _next_episode(self) -> tuple[str, int, dict[str, Any] | None, str]:
        if self.spec.episodes is not None:
            if self.counter >= len(self.spec.episodes):
                return "", 0, None, ""
            scenario, seed = self.spec.episodes[self.counter]
            overrides = (self.spec.episode_overrides or [None] * len(self.spec.episodes))[self.counter]
            return scenario, int(seed), overrides, "fixed"
        stage = stage_for(self.spec.curriculum or [{"scenarios": {"S0": 1.0}}], self.progress)
        names = sorted(stage["scenarios"])
        weights = np.asarray([float(stage["scenarios"][name]) for name in names], dtype=np.float64)
        scenario = names[int(self._rng.choice(len(names), p=weights / weights.sum()))]
        seed = training_episode_seed(self.spec.training_seed, self.spec.index, self.counter)
        return scenario, seed, stage.get("environment_overrides"), str(stage.get("name", "stage"))

    def reset(self) -> np.ndarray:
        scenario, seed, overrides, stage = self._next_episode()
        self.counter += 1
        if not scenario:
            self.exhausted = True
            return np.zeros(296, dtype=np.float32)
        key = repr(sorted((overrides or {}).items()))
        if self.env is None or key != self._env_key:
            self.env = self._env_type(deep_merge(self.spec.env_config, overrides), scenario)
            self._env_key = key
        observation, _ = self.env.reset(seed=seed, options={"scenario_id": scenario})
        self.current = (scenario, seed, stage)
        self.episode_return = self.episode_cost = self.env_seconds = 0.0
        self.episode_length = 0
        self.actions = []
        return observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, float, bool, bool, dict[str, Any]]:
        if self.exhausted:
            return np.zeros(296, dtype=np.float32), 0.0, 0.0, True, False, {"exhausted": True}
        action = np.asarray(action, dtype=np.float32)
        self.actions.append(action)
        started = time.perf_counter()
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.env_seconds += time.perf_counter() - started
        cost = float(info.get("safety_cost", 0.0))
        self.episode_return += reward
        self.episode_cost += cost
        self.episode_length += 1
        compact: dict[str, Any] = {}
        if terminated or truncated:
            compact = {key: value for key, value in info.items() if key not in _DROPPED_INFO_KEYS}
            compact.update({
                "episode_return": self.episode_return, "episode_cost": self.episode_cost,
                "episode_length": self.episode_length, "truncated": bool(truncated),
                "terminated": bool(terminated),
                "action_smoothness": _smoothness(self.actions),
                "simulation_steps_per_second": self.episode_length / max(self.env_seconds, 1e-9),
                "seed": self.current[1], "stage": self.current[2],
                "final_observation": observation,
            })
            observation = self.reset()
        return observation, float(reward), cost, bool(terminated), bool(truncated), compact

    def set_progress(self, progress: float) -> None:
        self.progress = float(progress)


def _smoothness(actions: list[np.ndarray]) -> float:
    """Mean squared action change; same expression as the sequential evaluator."""
    diffs = np.diff(np.asarray(actions), axis=0) if len(actions) > 1 else np.zeros((1, 2))
    return float(np.mean(np.sum(diffs * diffs, axis=1)))


class SyncVectorEnv:
    def __init__(self, specs: Sequence[SlotSpec]):
        self.slots = [EpisodeSlot(spec) for spec in specs]
        self.num_envs = len(self.slots)

    def reset(self) -> np.ndarray:
        return np.stack([slot.reset() for slot in self.slots])

    def step(self, actions: np.ndarray):
        results = [slot.step(action) for slot, action in zip(self.slots, actions)]
        return _stack(results)

    def set_progress(self, progress: float) -> None:
        for slot in self.slots:
            slot.set_progress(progress)

    def exhausted(self) -> list[bool]:
        return [slot.exhausted for slot in self.slots]

    def close(self) -> None:
        self.slots = []


def _stack(results):
    observations, rewards, costs, terminated, truncated, infos = zip(*results)
    return (np.stack(observations).astype(np.float32), np.asarray(rewards, dtype=np.float64),
            np.asarray(costs, dtype=np.float64), np.asarray(terminated), np.asarray(truncated), list(infos))


def _worker(connection, spec: SlotSpec) -> None:
    # Environment workers never import torch: it is not needed and would add
    # a second of start-up time and a few hundred MB per process.
    slot = EpisodeSlot(spec)
    try:
        while True:
            command, payload = connection.recv()
            if command == "reset":
                connection.send(slot.reset())
            elif command == "step":
                connection.send(slot.step(payload))
            elif command == "progress":
                slot.set_progress(payload)
                connection.send(None)
            elif command == "exhausted":
                connection.send(slot.exhausted)
            elif command == "close":
                connection.send(None)
                break
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        connection.close()


class SubprocVectorEnv:
    """One spawned process per slot; same results as ``SyncVectorEnv``."""

    def __init__(self, specs: Sequence[SlotSpec]):
        context = mp.get_context("spawn")
        self.connections, self.processes = [], []
        for spec in specs:
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, spec), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        self.num_envs = len(self.connections)

    def _broadcast(self, command: str, payloads: Sequence[Any] | None = None) -> list[Any]:
        try:
            for index, connection in enumerate(self.connections):
                connection.send((command, None if payloads is None else payloads[index]))
            return [connection.recv() for connection in self.connections]
        except (EOFError, BrokenPipeError) as error:
            raise RuntimeError(
                "A vector-env worker exited. Workers use the 'spawn' start method, so the "
                "caller must be an importable module or a script guarded by "
                "`if __name__ == '__main__':` (not code piped through stdin); "
                "use vector_env.backend=sync otherwise.") from error

    def reset(self) -> np.ndarray:
        return np.stack(self._broadcast("reset"))

    def step(self, actions: np.ndarray):
        return _stack(self._broadcast("step", list(actions)))

    def set_progress(self, progress: float) -> None:
        self._broadcast("progress", [progress] * self.num_envs)

    def exhausted(self) -> list[bool]:
        return self._broadcast("exhausted")

    def close(self) -> None:
        if not self.connections:
            return
        try:
            self._broadcast("close")
        except (BrokenPipeError, EOFError):
            pass
        for process in self.processes:
            process.join(timeout=5)
        self.connections, self.processes = [], []

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def make_vector_env(specs: Sequence[SlotSpec], backend: str = "sync"):
    if backend == "subprocess" and len(specs) > 1:
        return SubprocVectorEnv(specs)
    if backend not in {"sync", "subprocess"}:
        raise ValueError(f"Unknown vector env backend {backend!r}")
    return SyncVectorEnv(specs)
