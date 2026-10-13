"""The default (Phase 0/M1) environment must not change.

Reference values were recorded from the code before the sensor model,
vectorized ray casting and procedural scenarios were added. Tolerances only
absorb last-digit libm differences between platforms.
"""
import numpy as np
import pytest

from amr_lite.config import load_config
from amr_lite.envs import WarehouseEnv
from amr_lite.planning import RuleBasedPlanner

REFERENCE = {
    "S3_rule_legacy": (
        {}, "S3", 5, 400, False,
        {"pose": [10.8055822697, 4.1375207682, -0.6023722389], "reward": 101.8238563608, "steps": 187,
         "tail": [0.0297674, -0.0132671, 0.999912, 0.4644428, -0.034636, 0.0058214, 0.0356355, 0.9993649],
         "scan_sum": 30.14147, "failure": "", "shield": {"PASS": 185, "CLAMP": 2, "STOP": 0}},
    ),
    "D0_noisy_predictive": (
        {"shield_strategy": "predictive"}, "D0", 11, 150, True,
        {"pose": [7.4204976207, 4.0164877883, 0.0786818113], "reward": 16.4014343203, "steps": 150,
         "tail": [0.0812761, -0.1038546, 0.9945925, 0.6565345, -0.1627728, 0.0082439, -0.0786007, 0.9969062],
         "scan_sum": 42.16184, "failure": "", "shield": {"PASS": 143, "CLAMP": 7, "STOP": 0}},
    ),
    "D1_rule_randomized": (
        {"scenario_randomization": {"enabled": True}}, "D1", 20003, 400, False,
        {"pose": [5.9364048608, 3.9330547422, -0.0049761022], "reward": 31.8833060412, "steps": 118,
         "tail": [0.08125, 0.0, 1.0, 0.0, 0.0, 2e-07, 7e-07, 1.0],
         "scan_sum": 35.77698, "failure": "COLLISION_DYNAMIC", "shield": {"PASS": 94, "CLAMP": 9, "STOP": 15}},
    ),
    "S1_noisy_shield_off": (
        {"shield_enabled": False}, "S1", 3, 200, True,
        {"pose": [6.1951478395, 6.208444743, 0.1938983811], "reward": -12.9306355914, "steps": 200,
         "tail": [0.0812749, -0.0858945, 0.9963042, 0.7674453, 0.2215651, -0.0080476, -0.1105308, 0.9938727],
         "scan_sum": 37.39208, "failure": "", "shield": {"PASS": 200, "CLAMP": 0, "STOP": 0}},
    ),
}


@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_default_environment_reproduces_pre_change_rollouts(case):
    overrides, scenario, seed, steps, noisy, expected = REFERENCE[case]
    env = WarehouseEnv(load_config("env", overrides), scenario)
    observation, _ = env.reset(seed=seed, options={"scenario_id": scenario})
    planner = RuleBasedPlanner()
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(steps):
        action = planner.predict(observation)
        if noisy:
            action = np.clip(action + rng.normal(0, 0.6, 2), -1, 1).astype(np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    assert env.step_count == expected["steps"]
    assert info["failure_type"] == expected["failure"]
    assert info["shield_counts"] == expected["shield"]
    assert list(info["robot_pose"]) == pytest.approx(expected["pose"], abs=1e-6)
    assert total == pytest.approx(expected["reward"], abs=1e-5)
    assert observation[288:296].tolist() == pytest.approx(expected["tail"], abs=1e-5)
    assert float(observation[216:288].sum()) == pytest.approx(expected["scan_sum"], abs=1e-3)
