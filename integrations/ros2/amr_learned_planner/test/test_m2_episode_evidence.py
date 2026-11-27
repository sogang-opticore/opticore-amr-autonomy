import json

from amr_learned_planner.core.recorder import EpisodeRecorder


def active(rec, stamp, point, goal_id):
    rec.start(stamp, point, goal_id=goal_id)
    rec.on_path(stamp, point, "map")
    rec.on_status(stamp, "NORMAL")


def test_preempted_goal_latched_status_wrong_path_far_pose_and_old_pose():
    rec = EpisodeRecorder(timeout_sec=5)
    active(rec, 1, (0, 0), "first")
    rec.on_pose(1, 0, 0)
    rec.start(2, (1, 0), goal_id="second")
    assert rec.rows[0]["termination_reason"] == "PREEMPTED"
    rec.start(2, (1, 0), goal_id="second")
    rec.on_pose(2, 1, 0)
    rec.on_status(2, "REACHED")
    rec.on_path(1.8, (1, 0), "map")
    rec.on_path(2, (0, 0), "map")
    rec.on_status(2, "NORMAL")
    rec.on_status(2.1, "REACHED")
    assert not rec.current.armed and rec.current.terminal_at is None
    rec.on_path(2.2, (1, 0), "map")
    rec.on_status(2.3, "NORMAL")
    rec.on_pose(2.4, 0, 0)
    rec.on_pose(2.4, 1, 0, frame="odom")
    rec.on_status(2.5, "REACHED")
    assert rec.current.terminal_at is None
    rec.on_pose(2.6, 1, 0)
    rec.on_status(3.2, "REACHED")
    rec.on_pose(7.0, 1, 0)
    rec.on_status(7.0, "REACHED")
    assert rec.tick(7.3)["termination_reason"] == "TIMEOUT"
    assert len({r["episode_id"] for r in rec.rows}) == 2


def test_observed_contact_while_stationary_and_dropped_positive_packet(tmp_path):
    rec = EpisodeRecorder(events_path=tmp_path / "e.jsonl")
    rec.on_contact(1, False, "gazebo_contacts", count=0, epoch=0)
    rec.start(1, (0, 0), goal_id="one")
    rec.on_contact(1.1, False, "gazebo_contacts", count=1, epoch=0)
    assert rec.rows[0]["termination_reason"] == "COLLISION" and rec.rows[0]["collision"] == 1
    rec.start(2, (1, 0), goal_id="two")
    rec.on_contact(2.1, True, "gazebo_contacts", pairs=[["robot::chassis", "box"]], count=2, epoch=0)
    assert rec.rows[1]["collision_known"] == 1
    rec.close(3)
    events = [json.loads(line) for line in (tmp_path / "e.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events if e["goal_id"] == "two"] == ["goal", "contact", "end"]


def test_stream_gap_or_no_initial_coverage_prevents_clean_arrival():
    for initial in (None, 1.0):
        rec = EpisodeRecorder()
        if initial:
            rec.on_contact(initial, False, "gazebo_contacts", count=0, epoch=0)
        active(rec, 1, (0, 0), "run")
        rec.on_contact(1.5, False, "gazebo_contacts", count=0, epoch=0)
        rec.on_pose(1.5, 0, 0)
        rec.on_status(1.5, "GOAL_REACHED")
        assert (rec.rows[0]["arrived"], rec.rows[0]["collision"], rec.rows[0]["success"]) == (1, None, 0)


def test_delayed_contact_to_arrival_is_logged_before_success_and_clock_reset_ends_run():
    rec = EpisodeRecorder()
    rec.on_contact(0.9, False, "gazebo_contacts", count=0, epoch=0)
    active(rec, 1, (0, 0), "first")
    rec.on_pose(1.05, 0, 0)
    rec.on_status(1.05, "REACHED")
    assert rec.rows == []
    rec.on_contact(1.05, True, "gazebo_contacts", count=0, epoch=0)
    assert rec.rows[0]["termination_reason"] == "COLLISION"
    rec.start(2, (2, 0), goal_id="second")
    assert rec.tick(0)["termination_reason"] == "CLOCK_RESET"


def test_paired_report_keeps_unknown_collision_rate_unmeasured():
    import math
    from amr_lite.experiments.paired import gap_report

    row = {"scenario_id": "static", "success": 0, "collision": "", "minimum_clearance": "nan", "failure_type": "CONTACT_UNKNOWN"}
    report = gap_report([row], [row])
    assert math.isnan(report["overall"]["reference_collision"])
    assert math.isnan(report["legs"][0]["candidate_collision"])


def test_physics_evidence_epoch_change_even_within_nominal_sample_period():
    rec = EpisodeRecorder()
    rec.on_contact(1.0, False, "gazebo_contacts", count=0, epoch=3)
    active(rec, 1.0, (0, 0), "station")
    rec.on_contact(1.05, False, "gazebo_contacts", count=0, epoch=4)
    rec.on_pose(1.05, 0, 0)
    rec.on_status(1.05, "REACHED")
    assert rec.rows[-1]["failure_type"] == "CONTACT_UNKNOWN"
