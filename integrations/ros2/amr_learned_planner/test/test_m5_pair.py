"""Offline graph regressions for the measured two-robot goal replacement probe."""
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "experiments/m5"))
from run_pair import authority_graph


def endpoint(robot, name):
    return SimpleNamespace(node_namespace="/" + robot, node_name=name)


def test_rejects_duplicate_clock_private_topic_leaks_and_crossed_drive_subscriptions():
    publishers = {"/clock": [endpoint("amr1", "ros_gz_bridge")]}
    subscribers = {}
    for r in ("amr1", "amr2"):
        for name, channels in (
            ("cmd_arbiter", ["cmd_vel", "dwa/status", "cmd_arbiter/state"]),
            ("dwa_planner", ["dwa_fallback/status", "dwa_fallback/cmd_vel"]),
            ("m3_policy", ["learned_planner/health", "learned_planner/cmd_vel"]),
            ("astar_planner", ["global_path"]),
            ("m2_contact_bridge", ["m2/contacts"]),
        ):
            for channel in channels:
                publishers[f"/{r}/{channel}"] = [endpoint(r, name)]
        publishers[f"/{r}/goal_pose"] = [endpoint("", "m5_fleet_probe")]
        subscribers[f"/{r}/goal_pose"] = [endpoint(r, name) for name in (
            "cmd_arbiter", "dwa_planner", "astar_planner", "m3_policy")]
        subscribers[f"/{r}/cmd_vel"] = [endpoint(r, "ros_gz_bridge")]
        subscribers[f"/{r}/dwa/status"] = [endpoint(r, "astar_planner")]

    node = SimpleNamespace(get_publishers_info_by_topic=publishers.__getitem__,
                           get_subscriptions_info_by_topic=subscribers.__getitem__)
    assert authority_graph(node)["publishers"]["/amr2/cmd_vel"] == ["/amr2/cmd_arbiter"]
    for endpoints, topic, wrong in (
        (publishers, "/clock", endpoint("amr2", "ros_gz_bridge")),
        (publishers, "/amr2/cmd_vel", endpoint("amr2", "dwa_planner")),
        (publishers, "/amr1/dwa_fallback/status", endpoint("amr2", "dwa_planner")),
        (subscribers, "/amr1/goal_pose", endpoint("amr2", "cmd_arbiter")),
        (subscribers, "/amr1/cmd_vel", endpoint("amr2", "ros_gz_bridge")),
    ):
        endpoints[topic].append(wrong)
        assert authority_graph(node) is None
        endpoints[topic].pop()
    subscribers["/amr2/goal_pose"].pop()
    assert authority_graph(node) is None


def test_hold_requires_fresh_ground_truth_zero_command_and_matching_authority():
    from run_pair import hold_acknowledged

    moving = (2.0, 22.0, .12, 0., 16.0, "world")
    stopped = (2.0, 22.0, .01, .02, 16.0, "world")
    row = {"motion": stopped, "cmd": (0., 0.), "cmd_received": 16.0,
           "state": {"goal_id": "hold", "active": "stop", "stamp": 16.0}}
    assert hold_acknowledged(row, "hold", 16.)
    assert not hold_acknowledged(row, "old", 16.)
    assert not hold_acknowledged(row, "hold", 16.01)
    assert not hold_acknowledged(dict(row, motion=moving), "hold", 16.)
    assert not hold_acknowledged(dict(row, cmd=(.1, 0.)), "hold", 16.)
    assert not hold_acknowledged(dict(row, cmd_received=15.9), "hold", 16.)
    assert not hold_acknowledged(dict(row, state=dict(row["state"], active="learned")), "hold", 16.)


def test_contact_is_measured_failure_and_loss_of_evidence_invalidates_the_trial():
    import pytest
    from run_pair import check_telemetry, ProtocolFailure

    row = {"motion": (2., 22., 0., 0., 16., "world"), "scan": (16., "amr1/lidar_link"),
           "cmd_received": 16., "status_received": 16., "health": {"stamp": 16.},
           "state": {"stamp": 16.}, "contact": {"collision": False, "collision_events": 0,
                                                "coverage_epoch": 1, "stamp": 16.}}
    check_telemetry("amr1", row, 16.2, 1, "hold")
    with pytest.raises(RuntimeError, match="telemetry gap") as gap:
        check_telemetry("amr1", row, 16.2, 0, "hold")
    assert not isinstance(gap.value, ProtocolFailure)
    with pytest.raises(RuntimeError, match="telemetry gap"):
        check_telemetry("amr1", row, 16.7, 1, "hold")
    row["contact"]["collision_events"] = 1
    with pytest.raises(ProtocolFailure, match="physical contact"):
        check_telemetry("amr1", row, 16.2, 1, "hold")
