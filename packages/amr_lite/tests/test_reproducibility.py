import numpy as np

from amr_lite.envs import WarehouseEnv


def test_deterministic_reset_and_step():
    first, second = WarehouseEnv(scenario_id="D0"), WarehouseEnv(scenario_id="D0")
    a, _ = first.reset(seed=99)
    b, _ = second.reset(seed=99)
    assert np.array_equal(a, b)
    for _ in range(3):
        a, ra, ta, tra, ia = first.step([0.0, 0.1])
        b, rb, tb, trb, ib = second.step([0.0, 0.1])
        assert np.array_equal(a, b)
        assert (ra, ta, tra, ia["robot_pose"]) == (rb, tb, trb, ib["robot_pose"])

