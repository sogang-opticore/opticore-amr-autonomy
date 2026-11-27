"""Exercise M5 batch verdicts without starting ROS or Gazebo."""
import json
import subprocess

import pytest

import run_batch


def leg(reason="ARRIVED", *, known="1", collision="0", events=()):
    return {"row": {"termination_reason": reason, "success": "1" if reason == "ARRIVED" else "0",
                    "collision_known": known, "collision": collision},
            "physical_goal_error_m": 0.18, "states": {"LEARNED": 4},
            "events": {name: {"stamp": i + 1} for i, name in enumerate(events)}}


def trial(legs, expected=None):
    count = len(legs) if expected is None else expected
    return {"status": "recorded", "case": {"goals": [[i, 0] for i in range(count)]}, "legs": legs}


def drive(tmp_path, monkeypatch, case, payload, code):
    out = tmp_path / "trial"

    def driver(command, **kwargs):
        assert str(out) == command[command.index("--out") + 1]
        out.mkdir()
        if payload is not None:
            (out / "run.json").write_text(payload)
        return subprocess.CompletedProcess(command, code)

    monkeypatch.setattr(run_batch.subprocess, "run", driver)
    return run_batch.run_case(out, tmp_path / "models", case, 7, 0)


def test_paired_suite_and_full_batch_share_the_same_candidates():
    seed_set = {7, 19, 31}
    assert set(run_batch.cases("paired")) == {(case, seed) for case in ("baseline", "blockage") for seed in seed_set}
    assert set(run_batch.cases("all")) == (
        {(case, seed) for case in run_batch.SINGLE_CASES for seed in seed_set} | {("pair", 7), ("fleet", 7)})
    assert len(run_batch.cases("all")) == len(set(run_batch.cases("all")))


def test_all_issued_legs_must_finish_collision_free():
    assert run_batch.assess_single(trial([leg(), leg()]), "sequence")["passed"]
    assert not run_batch.assess_single(trial([leg()], expected=2), "sequence")["passed"]
    assert not run_batch.assess_single(trial([], expected=2), "sequence")["passed"]
    for bad in (leg("TIMEOUT"), leg("COLLISION", collision="1"), leg(known="0"), leg(collision="1")):
        assert not run_batch.assess_single(trial([leg(), bad]), "sequence")["passed"]


def test_blockage_needs_all_four_motion_and_actor_witnesses():
    events = ("moving_before_obstacle", "stopped_before_release", "actor_removed", "advanced_after_release")
    assert run_batch.assess_single(trial([leg(events=events)]), "blockage")["passed"]
    for excluded in events:
        remaining = [event for event in events if event != excluded]
        assert not run_batch.assess_single(trial([leg(events=remaining)]), "blockage")["passed"]
    assert not run_batch.assess_single(trial([leg("TIMEOUT", events=events)]), "blockage")["passed"]


def test_faults_require_measured_cancel_then_collision_free_replacement():
    events = ("drop_scans", "stale_stop", "restored_scans_motion", "fallback_motion", "learned_recovery",
              "cancel_sent", "cancelled_stop")
    first = leg("CANCELLED", events=events)
    assert run_batch.assess_single(trial([first, leg()]), "faults")["passed"]
    for excluded in events:
        partial = leg("CANCELLED", events=[event for event in events if event != excluded])
        assert not run_batch.assess_single(trial([partial, leg()]), "faults")["passed"]
    for bad_first in (leg("TIMEOUT", events=events), leg("CANCELLED", known="0", events=events),
                      leg("CANCELLED", collision="1", events=events)):
        assert not run_batch.assess_single(trial([bad_first, leg()]), "faults")["passed"]
    assert not run_batch.assess_single(trial([first, leg(known="0")]), "faults")["passed"]
    assert not run_batch.assess_single(trial([first], expected=2), "faults")["passed"]


def test_measured_timeout_is_a_recorded_negative(tmp_path, monkeypatch):
    result = drive(tmp_path, monkeypatch, "baseline", json.dumps(trial([leg("TIMEOUT")])), 0)
    assert result["status"] == "recorded" and result["passed"] is False
    assert result["leg_outcomes"] == ["TIMEOUT"]


def test_timeout_with_success_flag_cannot_pass():
    item = leg("TIMEOUT")
    item["row"]["success"] = "1"
    assert not run_batch.assess_single(trial([item]), "baseline")["passed"]


def test_arrival_requires_world_position_and_candidate_control():
    for patch in ({"physical_goal_error_m": 0.26}, {"physical_goal_error_m": float("nan")},
                  {"states": {"FALLBACK": 3}}):
        assert not run_batch.assess_single(trial([{**leg(), **patch}]), "baseline")["passed"]


def test_suite_exit_code_distinguishes_mission_failure_and_incomplete_evidence():
    passed = {"status": "recorded", "passed": True}
    failed = {"status": "recorded", "passed": False}
    invalid = {"status": "invalid"}
    assert run_batch.verdict([passed]) == ("passed", 0)
    assert run_batch.verdict([passed, failed]) == ("failed", 3)
    assert run_batch.verdict([failed, invalid]) == ("invalid", 1)


@pytest.mark.parametrize(("payload", "exit_code"), [(None, 0), ("{", 0), ('{"status":"recorded"}', 0),
                                                    ('{"status":"unverified"}', 0), ('{"status":"recorded"}', 1)])
def test_missing_and_invalid_driver_output(tmp_path, monkeypatch, payload, exit_code):
    assert drive(tmp_path, monkeypatch, "baseline", payload, exit_code)["status"] == "invalid"


def test_missing_contact_field_invalidates_record(tmp_path, monkeypatch):
    item = leg()
    del item["row"]["collision_known"]
    assert drive(tmp_path, monkeypatch, "baseline", json.dumps(trial([item])), 0)["status"] == "invalid"


@pytest.mark.parametrize(("driver_status", "exit_code", "expected", "passed"),
                         [("verified", 0, "recorded", True), ("failed", 1, "recorded", False),
                          ("unverified", 0, "invalid", None), ("verified", 1, "invalid", None),
                          ("recorded", 0, "invalid", None)])
def test_pair_uses_pair_driver_verdict(tmp_path, monkeypatch, driver_status, exit_code, expected, passed):
    result = drive(tmp_path, monkeypatch, "pair", json.dumps({"status": driver_status}), exit_code)
    assert result["status"] == expected
    assert result.get("passed") is passed


def test_retry_one_candidate_preserves_domain_and_finishes_subset(tmp_path, monkeypatch):
    output = tmp_path / "retry"
    monkeypatch.setattr(run_batch.sys, "argv", ["run_batch", "--suite", "sequence", "--seed", "31",
                                                "--out", str(output), "--candidates", str(tmp_path)])
    def record(folder, candidates, case, seed, index):
        assert (folder.name, case, seed, index) == ("03-sequence-31", "sequence", 31, 2)
        return {"status": "recorded", "passed": False}

    monkeypatch.setattr(run_batch, "run_case", record)
    assert run_batch.main() == 3
    report = json.loads((output / "summary.json").read_text())
    assert report["expected_trials"] == report["finished_trials"] == 1
    assert report["suite_verdict"] == "failed" and report["finished_at_utc"]


def test_cancel_without_contact_coverage_requires_new_measurement(tmp_path, monkeypatch):
    report = trial([leg("CANCELLED", known="0"), leg()])
    result = drive(tmp_path, monkeypatch, "faults", json.dumps(report), 0)
    assert result["status"] == "invalid" and result["leg_outcomes"] == ["CANCELLED", "ARRIVED"]
