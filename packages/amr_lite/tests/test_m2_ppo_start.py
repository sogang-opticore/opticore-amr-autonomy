from pathlib import Path

import numpy as np
import pytest
import yaml

from amr_lite.config import resolve_env_config
from amr_lite.envs.scenarios import DynamicObstacle
from amr_lite.envs.warehouse_env import WarehouseEnv


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m2" / "ppo_start.yaml"


def test_rollout_and_reward_profile():
    cfg = yaml.safe_load(CONFIG.read_text())
    contract = yaml.safe_load((CONFIG.parent / "policy_io.yaml").read_text())
    assert cfg["policy_io"] == contract["id"]
    assert cfg["robot_profile"] == contract["robot_profile"]
    assert cfg["versions"] == {"gymnasium": "1.0.0", "stable-baselines3": "2.4.1"}
    batch = cfg["ppo"]["batch_size"]
    assert batch > 1 and cfg["n_envs"] * cfg["ppo"]["n_steps"] % batch == 0
    env = WarehouseEnv(resolve_env_config(cfg["robot_profile"], cfg["environment_overrides"]), "S0")
    _, start = env.reset(seed=cfg["seed"])
    _, reward, term, trunc, end = env.step(np.array([1, 0], dtype=np.float32))
    assert not term and not trunc
    assert reward == pytest.approx(2 * (start["path_remaining_distance"] - end["path_remaining_distance"]) - .01)
    assert sum(end["reward_components"].values()) == pytest.approx(reward)
    assert env.config["shield_strategy"] == "scan"


@pytest.mark.parametrize("event", ["goal", "collision", "timeout"])
def test_terminal_precedence_at_deadline(event):
    cfg = yaml.safe_load(CONFIG.read_text())
    env = WarehouseEnv(resolve_env_config(cfg["robot_profile"], cfg["environment_overrides"]), "S0")
    env.reset(seed=cfg["seed"])
    env.episode_timeout = env.config["policy_dt"] / 10  # Deadline during the first tick.
    if event in {"goal", "collision"}:
        env.scenario.goal = (env.robot.x, env.robot.y)
    if event == "collision":
        # Place a stationary obstacle at the goal to exercise simultaneous contact/arrival.
        env.dynamic_obstacles.append(DynamicObstacle(env.robot.x, env.robot.y, vx=0, vy=0, radius=0.25))
    _, _, term, trunc, info = env.step([-1., 0.])
    assert bool(term) == (event != "timeout")
    assert bool(trunc) == (event == "timeout")
    assert bool(info["success"]) == (event == "goal")
    assert bool(info["collision"]) == (event == "collision")
    terms = info["reward_components"]
    assert terms["goal"] == (20. if event == "goal" else 0.)
    assert terms["collision"] == (-20. if event == "collision" else 0.)
