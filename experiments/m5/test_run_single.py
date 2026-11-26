"""Evidence requirements for a stopped AMR during blockage, input loss or cancellation."""
import pytest

from run_single import stopped_sample


@pytest.fixture
def stopped():
    return {"cmd": (0.0, 0.0), "cmd_stamp": 10.0, "motion_stamp": 9.99,
            "speed": 0.004, "yaw_rate": 0.006}


def test_stopped_base_has_both_command_and_ground_truth(stopped):
    assert stopped_sample(10.0, stopped)
    assert not stopped_sample(10.0, {"cmd": (0.0, 0.0), "cmd_stamp": 10.0})


@pytest.mark.parametrize(("field", "value"), [
    ("cmd", (0.15, 0.0)), ("cmd", (0.0, 0.2)), ("speed", 0.05),
    ("yaw_rate", 0.3), ("yaw_rate", -0.3), ("cmd_stamp", 9.6), ("motion_stamp", 9.6),
    ("cmd_stamp", 10.5), ("motion_stamp", 10.5),
])
def test_stall_rotation_or_stale_evidence_cannot_count_as_stop(stopped, field, value):
    stopped[field] = value
    assert not stopped_sample(10.0, stopped)
