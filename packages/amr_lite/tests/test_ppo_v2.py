import csv
import json

import numpy as np
import pytest
import torch

from amr_lite.config import deep_merge, load_config, resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.learning.networks import load_policy
from amr_lite.planning import RuleBasedPlanner
from amr_lite.rl.networks import PolicyNetwork
from amr_lite.rl.policy import PPOPolicy, combine_residual
from amr_lite.rl.ppo import _gae, train


def _observations(count: int = 40) -> np.ndarray:
    env = WarehouseEnv(resolve_env_config("opticore_gazebo"), "S3")
    observation, _ = env.reset(seed=3)
    planner, rng = RuleBasedPlanner(env.config), np.random.default_rng(1)
    rows = []
    for _ in range(count):
        rows.append(observation)
        action = np.clip(planner.predict(observation) + rng.normal(0, 0.3, 2), -1, 1)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(seed=4)
    return np.stack(rows)


def _config(**overrides) -> dict:
    base = deep_merge(load_config("ppo_v2"), {
        "network": {"hidden_sizes": [32, 32], "cnn": {"channels": [4, 8], "state_hidden": 8}},
        "vector_env": {"num_envs": 2, "backend": "sync"},
        "evaluation": {"interval_updates": 1, "scenarios": ["S0"], "episodes_per_scenario": 1,
                       "workers": 1, "backend": "sync"},
    })
    return deep_merge(base, overrides)


def test_residual_policy_starts_as_the_rule_baseline():
    observations = _observations()
    network = PolicyNetwork({"hidden_sizes": [32, 32]})
    network.zero_mean_head()
    config = resolve_env_config("opticore_gazebo")
    policy = PPOPolicy(network, "residual", 0.5, RuleBasedPlanner(config))
    expected = np.stack([RuleBasedPlanner(config).predict(item) for item in observations])
    assert np.array_equal(policy.predict_batch(observations), expected)
    assert np.allclose(combine_residual(np.array([[0.9, 0.0]]), np.array([[0.8, -1.0]]), 0.5),
                       [[1.0, -0.5]])


@pytest.mark.parametrize("encoder", ["mlp", "cnn"])
def test_network_shapes_and_cost_head(encoder):
    network = PolicyNetwork({"encoder": encoder, "cost_critic": True, "hidden_sizes": [16, 16],
                             "cnn": {"channels": [4, 8], "state_hidden": 8}})
    mean, log_std, value, cost = network(torch.as_tensor(_observations(5)))
    assert mean.shape == (5, 2) and log_std.shape == (5, 2)
    assert value.shape == (5,) and cost.shape == (5,)
    with pytest.raises(ValueError):
        PolicyNetwork({"encoder": "transformer"})


def test_gae_handles_episode_boundaries():
    rewards = np.array([[1.0], [1.0], [1.0]])
    values = np.array([[0.5], [0.5], [0.5]])
    dones = np.array([[0.0], [1.0], [0.0]])
    advantages = _gae(rewards, values, dones, np.array([2.0]), gamma=0.9, lam=1.0)
    # The step that ends an episode does not see the next episode's value.
    assert advantages[1, 0] == pytest.approx(1.0 - 0.5)
    assert advantages[2, 0] == pytest.approx(1.0 + 0.9 * 2.0 - 0.5)
    assert advantages[0, 0] == pytest.approx((1.0 + 0.9 * 0.5 - 0.5) + 0.9 * advantages[1, 0])


@pytest.mark.parametrize("overrides", [
    {},
    {"policy": {"mode": "direct"}, "network": {"encoder": "cnn"}, "lagrangian": {"enabled": True}},
])
def test_smoke_training_writes_loadable_checkpoints(tmp_path, overrides):
    result = train(tmp_path, _config(**overrides), smoke=True, verbose=False)
    assert result["status"] == "complete" and result["total_timesteps"] >= 512
    rows = list(csv.DictReader((tmp_path / "training.csv").open()))
    assert len(rows) == result["updates"] and rows[-1]["eval_success_rate"] != "nan"
    for name in ("final.pt", "best.pt"):
        policy = load_policy(str(tmp_path / name))
        assert policy.predict(_observations(1)[0]).shape == (2,)
    assert json.loads((tmp_path / "config.json").read_text())["environment"]["robot_profile"] == "opticore_gazebo"


def test_direct_warm_start_from_teacher_dataset(tmp_path):
    from amr_lite.learning.collect_teacher import collect

    collect(tmp_path / "teacher", episodes=2, max_steps=30, seed=3, scenarios=["S0"],
            env_config=resolve_env_config("opticore_gazebo"))
    result = train(tmp_path / "run", _config(policy={"mode": "direct"},
                                             warm_start={"dataset": [str(tmp_path / "teacher")], "epochs": 2,
                                                         "anchor_coef": 0.5}),
                   smoke=True, verbose=False)
    assert result["warm_start_mse"] is not None and result["warm_start_mse"] < 1.0


def test_paused_training_resumes_to_completion(tmp_path):
    config = _config(total_timesteps_smoke=768, evaluation={"interval_updates": 0})
    paused = train(tmp_path, config, smoke=True, verbose=False, max_updates=1)
    assert paused["status"] == "paused" and (tmp_path / "state.pt").exists()
    resumed = train(tmp_path, config, smoke=True, verbose=False, resume=True)
    assert resumed["status"] == "complete" and not (tmp_path / "state.pt").exists()
    rows = list(csv.DictReader((tmp_path / "training.csv").open()))
    assert [int(row["update"]) for row in rows] == list(range(1, resumed["updates"] + 1))
