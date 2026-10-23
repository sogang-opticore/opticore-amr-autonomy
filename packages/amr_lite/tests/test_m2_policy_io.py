from pathlib import Path

import numpy as np
import yaml

from amr_lite.config import resolve_env_config
from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.action import scale_action
from amr_lite.learning.observation import OBSERVATION_TAIL, build_observation
from amr_lite.planning.path_utils import path_state
from amr_lite.sensors.lidar import ContractScanSpec, resample_scan


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "m2" / "policy_io.yaml"


def test_source_profile_and_scan_geometry():
    contract = yaml.safe_load(CONFIG.read_text())
    scan = contract["scan"]
    env = resolve_env_config(contract["robot_profile"])
    raw = env["lidar_model"]
    assert scan["raw_samples"] == raw["raw_samples"]
    assert scan["mount_xy_yaw"] == [raw["mount_x"], raw["mount_y"], raw["mount_yaw"]]
    assert scan["raw_range_m"] == [raw["raw_range_min"], raw["raw_range_max"]]
    assert scan["rate_hz"] * env["sensor_model"]["scan_period"] == 1
    assert scan["bins"] == env["lidar_rays"]
    assert scan["fov_deg"] == env["lidar_fov_deg"]
    assert scan["range_cap_m"] == env["lidar_max_range"]
    ranges = np.full(scan["raw_samples"], np.inf)
    ranges[len(ranges) // 2] = 1.0
    result = resample_scan(
        ranges, raw["raw_angle_min"],
        (raw["raw_angle_max"] - raw["raw_angle_min"]) / (len(ranges) - 1),
        *scan["raw_range_m"], mount_x=raw["mount_x"], contract=ContractScanSpec.from_env_config(env),
    )
    assert result.shape == (scan["bins"],)
    assert np.count_nonzero(result < scan["range_cap_m"]) == 1
    np.testing.assert_allclose(result[scan["bins"] // 2], 1.25, atol=1e-5)


def test_tensor_order_and_physical_targets():
    contract = yaml.safe_load(CONFIG.read_text())
    obs, act, scan = contract["observation"], contract["action"], contract["scan"]
    env = resolve_env_config(contract["robot_profile"])
    assert tuple(obs["tail"]) == OBSERVATION_TAIL
    assert (obs["lookahead_m"], obs["goal_distance_scale_m"], obs["cross_track_scale_m"]) == (
        env["lookahead_distance"], env["goal_distance_scale"], env["max_cross_track_error"])
    rates = (act["max_linear_mps"], act["max_angular_radps"])
    assert rates == (env["v_max"], env["w_max"])
    assert act["control_hz"] * env["policy_dt"] == 1
    robot = RobotState(0, 0, 0, 0.75, -0.75)
    path = path_state([(0, 0), (3, 0)], robot, obs["lookahead_m"])
    scans = [np.full(scan["bins"], idx + 1.) for idx in range(scan["history_frames"])]
    tensor = build_observation(scans, path, robot, scan["range_cap_m"],
                               obs["goal_distance_scale_m"], *rates, obs["cross_track_scale_m"])
    assert tensor.dtype == np.dtype(obs["dtype"])
    assert tensor.shape == tuple(obs["shape"])
    np.testing.assert_allclose(tensor[:-8].reshape(4, 72)[:, 0], [0.125, 0.25, 0.375, 0.5])
    np.testing.assert_allclose(tensor[-8:], [0.65/8, 0, 1, 0.5, -0.5, 0, 0, 1])
    assert scale_action(act["normalized_stop"], *rates) == tuple(act["fault_twist"])
    assert scale_action([0, 0], *rates) == (0.75, 0)
    assert scale_action([1, -1], *rates) == (1.5, -1.5)
