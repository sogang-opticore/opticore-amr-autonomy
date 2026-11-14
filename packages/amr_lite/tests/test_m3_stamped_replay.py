import io
import json
import math

import pytest

from amr_lite.envs.dynamics import RobotState
from amr_lite.learning.stamped_io import StampedPolicyIO
from amr_lite.learning.stamped_replay import ReplayTrace, replay
from amr_lite.sensors.lidar import RawLidarSpec


def test_replay_watches_tensors_commands_faults_and_source_stamps():
    scan = RawLidarSpec()
    robot = RobotState(2, 4, 0)
    path = [(2, 4), (6, 4), (9, 4)]
    stream = io.StringIO()
    trace, inputs = ReplayTrace(stream), StampedPolicyIO()
    trace.reset()
    for index in range(4):
        stamp = index / 10
        args = ("/amr1", "amr1/lidar", stamp, [math.inf] * scan.samples,
                scan.angle_min, scan.angle_increment, scan.range_min, scan.range_max)
        trace.scan(*args)
        assert inputs.ingest_scan(*args)
        obs = inputs.observe(stamp, stamp, robot, path)
        action = [0.0, 1.0] if obs.tensor is not None else None
        command = inputs.actuate(action, robot) if action is not None else None
        trace.tick(stamp, stamp, robot, path, obs, action, command, goal_id="goal", path_revision=1)
    result = replay(io.StringIO(stream.getvalue()))
    assert result["counts"]["valid_observation"] == result["counts"]["physical_command"] == 1
    assert result["faults"] == {"SCAN_WARMUP": 3}
    assert result["max_observation_error"] == 0
    history = [json.loads(line) for line in stream.getvalue().splitlines()]
    history[-1]["observation"]["stamps"][-1] = 0.32
    with pytest.raises(ValueError, match="different observation or scan history"):
        replay(io.StringIO("\n".join(map(json.dumps, history))))
    history[-1]["observation"]["stamps"][-1] = 0.3
    history[-1]["command"]["command"][0] = 0.75
    with pytest.raises(ValueError, match="different physical command"):
        replay(io.StringIO("\n".join(map(json.dumps, history))))
