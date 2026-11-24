"""Operational ROS callbacks driven by stamped development fixtures."""
import json

import pytest

import fake_ros

INSTALLED = fake_ros.install()
pytestmark = pytest.mark.skipif(not INSTALLED, reason="requires fake clock")


def string(value):
    msg = fake_ros.String()
    msg.data = value
    return msg


def goal(node, t):
    fake_ros.Clock.time = t
    msg = fake_ros.PoseStamped()
    msg.header.frame_id = "map"
    node.deliver("goal_pose", msg)
    return node.goal_id


def tick(node, t, **updates):
    fake_ros.Clock.time = t
    data = {"stamp": t, "goal_id": node.goal_id, "path_stamp": node.goal_stamp, "odom_stamp": t,
            "shield_scan_stamp": t, "support_ready": True, "odom_xy": [0, 0],
            "command": [0.22, -0.20], "faults": [], "healthy": True, "status": "NORMAL", **updates}
    node.deliver("learned_planner/health", string(json.dumps(data)))
    return fire(node)


def fire(node):
    node.fire(0.05)
    state = json.loads(node.publishers["cmd_arbiter/state"].messages[-1].data)
    msg = node.publishers["cmd_vel"].messages[-1]
    assert node.publishers["dwa/status"].messages
    return state["state"], (msg.linear.x, msg.angular.z)


def dwa(node, t, v=0.8, w=-1.2):
    fake_ros.Clock.time = t
    cmd = fake_ros.Twist()
    cmd.linear.x, cmd.angular.z = v, w
    node.deliver("dwa_fallback/cmd_vel", cmd)


@pytest.fixture
def node():
    from amr_learned_planner.operational_arbiter_node import OperationalArbiter
    fake_ros.Node.parameter_overrides = {}
    fake_ros.Clock.time = 1.
    return OperationalArbiter()


def test_replace_cancel_and_ignore_old_reports(node):
    first = goal(node, 1.)
    moving = ("LEARNED", (0.22, -0.2))
    assert tick(node, 1.1) == moving
    second = goal(node, 1.2)
    assert second != first
    assert tick(node, 1.25, goal_id=first, status="REACHED") == ("WAIT_INPUT", (0., 0.))
    node.deliver("cancel_goal", string(first))
    assert tick(node, 1.3) == moving
    node.deliver("cancel_goal", string(second))
    assert tick(node, 1.4) == ("CANCELLED", (0., 0.))
    goal(node, 1.6)
    assert tick(node, 1.7) == moving
    fake_ros.Clock.time = 0.
    assert fire(node) == ("WAIT_INPUT", (0., 0.))
    assert node.goal_id is None


def test_policy_fault_dwell_fallback_recovery_and_stale_hard_stop(node):
    goal(node, 1.)
    assert tick(node, 1.1)[0] == "LEARNED"
    fault = dict(faults=["INFERENCE_DEADLINE"], healthy=False, status="STOPPED", command=[0., 0.])
    dwa(node, 1.19)
    assert tick(node, 1.2, **fault) == ("FALLBACK_HOLD", (0., 0.))
    assert tick(node, 1.51, **fault) == ("FALLBACK_HOLD", (0., 0.))
    dwa(node, 1.52)
    assert tick(node, 1.55, **fault) == ("FALLBACK", (0.45, -0.6))
    dwa(node, 1.56)
    assert tick(node, 1.6, odom_stamp=0.8, **fault) == ("WAIT_INPUT", (0., 0.))
    assert tick(node, 1.65, **fault) == ("FALLBACK_HOLD", (0., 0.))
    dwa(node, 2.)
    assert tick(node, 2.05, **fault)[0] == "FALLBACK"
    assert tick(node, 2.1)[0] == "FALLBACK"
    for i in range(1, 12):
        t = 2.1 + 0.1 * i
        dwa(node, t)
        last = tick(node, t)
    assert last == ("LEARNED", (0.22, -0.2))
    fake_ros.Clock.time = 3.5
    assert fire(node) == ("WAIT_INPUT", (0., 0.))


def test_replan_requires_new_goal_aligned_path_and_progress(node):
    goal(node, 1.)
    assert tick(node, 1.1)[0] == "LEARNED"
    assert tick(node, 4.15) == ("WAIT_REPLAN", (0., 0.))
    assert node.publishers["dwa/status"].messages[-1].data == "DYNAMIC_BLOCKED"
    assert tick(node, 4.2, path_stamp=4.0)[0] == "WAIT_REPLAN"
    assert tick(node, 4.3, path_stamp=4.25)[0] == "LEARNED"
    assert tick(node, 4.4, odom_xy=[0.18, 0.], path_stamp=4.25)[0] == "LEARNED"
    assert node.request_stamp is None
    assert tick(node, 7.5, odom_xy=[0.18, 0.], path_stamp=4.25)[0] == "WAIT_REPLAN"
    assert tick(node, 19.5, odom_xy=[0.18, 0.], path_stamp=19.4) == ("BLOCKED_TIMEOUT", (0., 0.))
    assert tick(node, 19.6, status="REACHED")[0] == "BLOCKED_TIMEOUT"
    goal(node, 20.)
    assert tick(node, 20.2, status="REACHED") == ("REACHED", (0., 0.))


@pytest.mark.parametrize("bad", [
    dict(support_ready=False), dict(shield_scan_stamp=-1.),
    dict(path_stamp=0.5), dict(odom_xy=[float("nan"), 0.0]),
    dict(faults=["SCAN_STALE"], healthy=False), dict(command=[0.1, float("nan")]),
])
def test_shared_input_failures_stop_both_sources(node, bad):
    goal(node, 1.)
    dwa(node, 1.1)
    assert tick(node, 1.15, **bad) == ("WAIT_INPUT", (0., 0.))


def test_recorder_finishes_only_matching_terminal():
    from amr_learned_planner.core.recorder import EpisodeRecorder
    recorder = EpisodeRecorder()
    recorder.start(1., (1., 0.), goal_id="map:1.000000000")
    snapshot = {"state":"CANCELLED", "mode":"learned", "goal_id":"map:0.000000000"}
    recorder.on_arbiter(1.05, snapshot)
    assert not recorder.rows
    recorder.on_arbiter(1.1, {**snapshot, "goal_id":"map:1.000000000"})
    assert recorder.rows[-1]["termination_reason"] == "CANCELLED"
    assert recorder.rows[-1]["success"] == 0


def test_missing_route_requests_astar_and_ends_when_unresolved(node):
    goal(node, 1.)
    invalid_path = dict(path_waiting=True, support_ready=False,
                        healthy=False, faults=["PATH_UNAVAILABLE"], status="STOPPED")
    assert tick(node, 1.1, **invalid_path)[0] == "WAIT_PATH"
    assert tick(node, 4.2, **invalid_path)[0] == "WAIT_REPLAN"
    assert node.publishers["dwa/status"].messages[-1].data == "DYNAMIC_BLOCKED"
    assert tick(node, 16.2, **invalid_path)[0] == "PATH_TIMEOUT"


@pytest.mark.parametrize("mismatch", [
    dict(faults=["PATH_UNAVAILABLE"], path_waiting=True, support_ready=False),
    dict(faults=["OUTSIDE_DEVELOPMENT_CORRIDOR"], support_ready=True),
])
def test_missing_or_unusable_route_requires_a_post_request_path(node, mismatch):
    goal(node, 1.)
    assert tick(node, 1.1, healthy=False, **mismatch)[0] == "WAIT_PATH"
    assert tick(node, 4.2, healthy=False, **mismatch)[0] == "WAIT_REPLAN"
    assert tick(node, 4.3, path_stamp=4.0)[0] == "WAIT_REPLAN"
    assert tick(node, 4.4, path_stamp=4.35)[0] == "LEARNED"


def test_sensor_glitch_and_odometry_drift_cannot_acknowledge_pending_replan(node):
    goal(node, 1.)
    tick(node, 1.1)
    assert tick(node, 4.2)[0] == "WAIT_REPLAN"
    assert tick(node, 4.3, odom_xy=[0.2, 0.0])[0] == "WAIT_REPLAN"
    assert tick(node, 4.4, healthy=False, faults=["SCAN_STALE"], shield_scan_stamp=2.)[0] == "WAIT_INPUT"
    assert tick(node, 4.5, odom_xy=[0.2, 0.0])[0] == "WAIT_REPLAN"
    assert tick(node, 4.6, odom_xy=[0.2, 0.0], path_stamp=4.55)[0] == "LEARNED"


def test_operational_fault_controls_reject_partial_updates_and_allow_recovery():
    from collections import deque
    from types import SimpleNamespace
    from amr_learned_planner.m3_policy_node import M3PolicyNode

    policy = M3PolicyNode.__new__(M3PolicyNode)
    policy.params = {"operational": False}
    policy.pending_scans = deque()
    policy.namespace = "amr1"
    policy.scan_frames = ("amr1/lidar_link",)
    def setting(name, value):
        return SimpleNamespace(name=name, value=value)

    drop = setting("diagnostic_drop_scans", True)
    latency = setting("diagnostic_inference_delay_ms", 35.0)
    assert not policy.update_diagnostics([drop]).successful
    policy.params["operational"] = True
    assert not policy.update_diagnostics([drop, setting(latency.name, float("nan"))]).successful
    assert drop.name not in policy.params
    assert policy.update_diagnostics([drop, latency]).successful
    scan = fake_ros.LaserScan()
    scan.header.frame_id = "amr1/lidar_link"
    policy.receive_scan(scan)
    assert not policy.pending_scans
    assert policy.update_diagnostics([setting(drop.name, False), setting(latency.name, 0.0)]).successful
    policy.receive_scan(scan)
    assert len(policy.pending_scans) == 1


def test_operational_and_policy_goals_share_clock_epoch_and_future_filter(node):
    from collections import deque
    from amr_learned_planner.m3_policy_node import M3PolicyNode

    policy = M3PolicyNode.__new__(M3PolicyNode)
    policy.goal = policy.path = policy.odom = policy.last_clock = None
    policy.path_revision = 0
    policy.odom_history = deque()
    policy.reset_history = lambda: None

    def dispatch(t, stamp):
        fake_ros.Clock.time = t
        msg = fake_ros.PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp.sec, msg.header.stamp.nanosec = stamp, 0
        node.deliver("goal_pose", msg)
        policy.receive_goal(msg)

    dispatch(10., 10)
    dispatch(10.1, 100)
    assert policy.goal[2] == node.goal_id == "map:10.000000000"
    fake_ros.Clock.time = 10.1
    path = fake_ros.Path()
    path.header.stamp.sec = 999
    policy.receive_path(path)
    assert policy.path is None
    dispatch(0., 0)
    assert policy.goal[2] == node.goal_id == "map:0.000000000"
    assert node.clock_time() == policy.clock_time() == 0.
    assert policy.goal[2] == node.goal_id


def test_goal_scoped_health_counts_each_missed_deadline_and_stale_input_fault():
    from amr_learned_planner.core.recorder import EpisodeRecorder
    recorder = EpisodeRecorder()
    recorder.start(1., (1., 0.), goal_id="map:1.000000000")
    fault = {"goal_id":"map:1.000000000", "stamp":1.05,
             "faults":["INFERENCE_DEADLINE"], "inference_ms":29.25}
    recorder.on_health(1.05, {**fault, "goal_id":"old"})
    recorder.on_health(1.05, fault)
    recorder.on_health(1.06, fault)
    recorder.on_health(1.15, {**fault, "stamp":1.14, "faults":["ODOM_STALE"], "inference_ms":None})
    row = recorder.finish(1.2, "INTERRUPTED")
    assert row["deadline_misses"] == 1 and row["max_inference_ms"] == 29.25
    assert row["faults"] == "INFERENCE_DEADLINE;ODOM_STALE"
