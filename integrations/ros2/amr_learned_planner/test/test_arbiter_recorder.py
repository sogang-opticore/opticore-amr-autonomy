import math

import numpy as np
import pytest

from amr_learned_planner.core.arbiter import ArbiterConfig, CommandArbiter
from amr_learned_planner.core.recorder import EpisodeRecorder
from amr_learned_planner.core.sequence import GoalSequence


def _feed(arbiter: CommandArbiter, now: float, learned=(0.5, 0.1), dwa=(0.3, -0.1), healthy=True):
    outputs = [arbiter.on_cmd("learned", now, *learned), arbiter.on_cmd("dwa", now, *dwa)]
    arbiter.on_health(now, healthy, [] if healthy else ["DEADLINE_MISS"])
    outputs.append(arbiter.tick(now))
    return [output.cmd for output in outputs if output.cmd is not None]


@pytest.mark.parametrize("mode", ["dwa", "shadow"])
def test_dwa_and_shadow_modes_forward_only_the_dwa(mode):
    arbiter = CommandArbiter(ArbiterConfig(mode=mode))
    for step in range(100):
        assert _feed(arbiter, 0.05 * step) == [(0.3, -0.1)]
    assert arbiter.on_status("learned", 5.0, "NORMAL").statuses == []
    assert arbiter.on_status("dwa", 5.0, "REJOIN").statuses == ["REJOIN"]
    assert arbiter.state == "DWA" and arbiter.fallbacks == 0


def test_learned_mode_starts_on_dwa_and_hands_over_when_healthy():
    arbiter = CommandArbiter(ArbiterConfig(mode="learned", min_fallback_sec=2.0, recover_after_sec=1.0))
    assert arbiter.state == "FALLBACK"
    arbiter.on_status("learned", 0.0, "NORMAL")
    handed_over = None
    for step in range(60):
        now = 0.05 * step
        commands = _feed(arbiter, now)
        if arbiter.state == "LEARNED" and handed_over is None:
            handed_over = now
        if handed_over is None:
            assert commands == [(0.3, -0.1)]
    assert handed_over == pytest.approx(2.0)
    assert _feed(arbiter, 3.05) == [(0.5, 0.1)]


def test_fault_holds_a_stop_then_falls_back_and_recovers():
    arbiter = CommandArbiter(ArbiterConfig(mode="learned", fallback_hold_sec=0.3, min_fallback_sec=1.0,
                                           recover_after_sec=0.5))
    arbiter.state, arbiter.healthy_since = "LEARNED", 0.0
    arbiter.on_status("dwa", 0.0, "NORMAL")
    _feed(arbiter, 0.0)
    trace = []
    for step in range(1, 80):
        now = 0.05 * step
        healthy = not 0.5 <= now < 1.0
        trace.append((round(now, 2), arbiter.state, _feed(arbiter, now, healthy=healthy)))
    states = [state for _, state, _ in trace]
    assert "FALLBACK_HOLD" in states and "FALLBACK" in states and states[-1] == "LEARNED"
    hold = [commands for _, state, commands in trace if state == "FALLBACK_HOLD"]
    # Only stops while holding; the tick that ends the hold hands over to the DWA at once.
    assert all(commands == [(0.0, 0.0)] for commands in hold[:-1])
    assert hold[-1] == [(0.3, -0.1)]
    fallback = [(now, commands) for now, state, commands in trace if state == "FALLBACK"]
    assert all(commands == [(0.3, -0.1)] for _, commands in fallback[:-1])
    # min_fallback_sec after entering FALLBACK at 0.8 s, the learned planner takes over at once.
    assert fallback[-1] == (1.8, [(0.3, -0.1), (0.5, 0.1)])
    assert arbiter.fallbacks == 1


def test_stale_active_source_stops_once_and_nan_commands_count_as_missing():
    arbiter = CommandArbiter(ArbiterConfig(mode="dwa", cmd_timeout=0.25))
    arbiter.on_cmd("dwa", 0.0, 0.4, 0.0)
    assert arbiter.tick(0.1).cmd is None
    first, second = arbiter.tick(0.4), arbiter.tick(0.45)
    assert first.cmd == second.cmd == (0.0, 0.0)
    assert first.statuses == ["STOPPED"] and second.statuses == []
    assert arbiter.on_cmd("dwa", 0.5, math.nan, 0.0).cmd is None
    assert arbiter.tick(0.8).cmd == (0.0, 0.0)
    with pytest.raises(ValueError):
        CommandArbiter(ArbiterConfig(mode="auto"))


def test_recovery_can_be_disabled():
    arbiter = CommandArbiter(ArbiterConfig(mode="learned", allow_recovery=False))
    for step in range(200):
        _feed(arbiter, 0.05 * step)
    assert arbiter.state == "FALLBACK"


def test_recorder_success_timeout_collision_and_fallbacks(tmp_path):
    recorder = EpisodeRecorder(label="learned", timeout_sec=10.0, lidar_mount=(0.0, 0.0, 0.0))
    angles = np.linspace(-math.pi, math.pi, 8, endpoint=False)
    recorder.start(0.0, (3.0, 0.0), "WH-x-00", 0)
    for step in range(31):
        recorder.on_pose(0.1 * step, 0.1 * step, 0.0)
    recorder.on_scan(1.0, np.full(8, 1.0), float(angles[0]), float(angles[1] - angles[0]), 0.1, 10.0)
    recorder.on_arbiter(1.0, {"mode": "learned", "state": "FALLBACK_HOLD"})
    recorder.on_arbiter(2.0, {"mode": "learned", "state": "LEARNED"})
    recorder.on_health(2.0, {"shield_mode": "CLAMP", "inference_ms": 0.4, "deadline_misses": 0})
    recorder.on_health(2.1, {"shield_mode": "PASS", "inference_ms": 0.2, "deadline_misses": 1})
    recorder.on_status(3.0, "GOAL_REACHED")
    success = recorder.rows[-1]
    assert success["success"] == 1 and success["collision"] == 0
    assert success["path_length"] == pytest.approx(3.0) and success["episode_time"] == pytest.approx(3.0)
    assert success["minimum_clearance"] == pytest.approx(1.0 - 0.36)
    assert success["fallback_count"] == 1 and success["fallback_time"] == pytest.approx(1.0)
    assert success["shield_intervention_rate"] == pytest.approx(0.5) and success["deadline_misses"] == 1
    recorder.start(5.0, (9.0, 9.0), "WH-x-01", 0)
    recorder.on_scan(6.0, np.full(8, 0.2), float(angles[0]), float(angles[1] - angles[0]), 0.1, 10.0)
    assert recorder.tick(14.0) is None
    timed_out = recorder.tick(15.5)
    assert timed_out["failure_type"] == "COLLISION_PROXY" and timed_out["collision"] == 1
    recorder.start(20.0, (1.0, 1.0))
    assert recorder.tick(31.0)["failure_type"] == "TIMEOUT"
    lines = recorder.write_csv(tmp_path / "episodes.csv").read_text().splitlines()
    assert len(lines) == 4 and lines[0].startswith("policy,scenario_id,seed,success")


def test_goal_sequence_orders_legs_aborts_on_failure_and_repeats_runs():
    spec = {"name": "demo", "goal_timeout_sec": 30.0,
            "legs": [{"goal": {"x": 1.0, "y": 0.0}}, {"goal": {"x": 2.0, "y": 0.0}}, {"goal": {"x": 3.0, "y": 0.0}}]}
    sequence = GoalSequence(spec, runs=2, settle_sec=1.0)
    first = sequence.next_goal(0.0)
    assert (first.scenario_id, first.run) == ("WH-demo-00", 0) and sequence.next_goal(0.1) is None
    sequence.on_result(5.0, True)
    assert sequence.next_goal(5.5) is None
    assert sequence.next_goal(6.0).scenario_id == "WH-demo-01"
    sequence.on_result(9.0, False)  # aborts leg 02 of run 0
    assert [item["scenario_id"] for item in sequence.skipped] == ["WH-demo-02"]
    second_run = sequence.next_goal(10.0)
    assert (second_run.scenario_id, second_run.run) == ("WH-demo-00", 1)
    for leg in range(3):
        sequence.on_result(20.0 + leg, True)
        sequence.next_goal(30.0 + leg)
    assert sequence.done and sequence.timeout_sec == 30.0
