"""Regression checks for matched training RNG and raw-stamped no-history deployment."""
import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from amr_lite.cli.train_m4 import CurriculumWorker
from amr_lite.deploy.runtime import PolicyRuntime
from amr_lite.envs.m4_curriculum import curriculum_stages
from amr_lite.learning.observation import repeat_latest_scan


def test_variants_share_mission_and_sensor_draws_at_each_worker_episode():
    missions = curriculum_stages()[0]
    full = CurriculumWorker(missions, 1, "full")
    no_history = CurriculumWorker(missions, 1, "no_history")
    nominal = CurriculumWorker(missions, 1, "no_randomization")
    for episode, seed in enumerate((8, None)):
        observation, _ = full.reset(seed=seed)
        repeated, _ = no_history.reset(seed=seed)
        nominal.reset(seed=seed)
        np.testing.assert_array_equal(repeat_latest_scan(observation), repeated)
        assert full.env.mission_id == nominal.env.mission_id == missions[(1 + episode) % len(missions)]
        assert full.episode_spec == no_history.episode_spec
        assert full.episode_spec["sensor_seed"] == nominal.episode_spec["sensor_seed"]
        assert full.env.dynamics.drive_delay_ticks in (1, 2)
        assert nominal.env.dynamics.drive_delay_ticks == 0


def test_no_history_runtime_uses_raw_unequal_frames_and_preserves_tail():
    raw = np.linspace(-0.1, 0.1, 296 * 2, dtype=np.float32).reshape(2, 296)
    original = raw.copy()
    transformed = repeat_latest_scan(raw)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(raw[:, -8:], transformed[:, -8:])
    for start in (0, 72, 144, 216):
        np.testing.assert_array_equal(transformed[:, start:start + 72], raw[:, 216:288])
    manifest = {"observation_contract": {"history_mode": "newest"}, "policy": {
        "kind": "sb3_ppo", "mode": "direct", "input_transform": "repeat_latest_scan",
        "network": {"type": "mlp", "layers": [{"op": "linear", "weight": "w", "bias": "b"}],
                    "output": "clip"},
    }}
    weights = np.zeros((2, 296), dtype=np.float32)
    weights[0, 0] = 1
    weights[1, -1] = 1
    runtime = PolicyRuntime(manifest, {"w": weights, "b": np.zeros(2, dtype=np.float32)})
    np.testing.assert_array_equal(runtime.predict_batch(raw), raw[:, [216, 295]])
