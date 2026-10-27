import json
import sys
from pathlib import Path

import pytest
from summarize_baseline import resolve, summarize

ROS_PACKAGE = Path(__file__).resolve().parents[2] / "integrations/ros2/amr_learned_planner"
sys.path.insert(0, str(ROS_PACKAGE))
from amr_learned_planner.core.recorder import EpisodeRecorder


def trial(path, disposition="success", mission="train-static", planner="b0"):
    """Generate recorder evidence, with a dummy bag envelope for these offline unit tests."""
    path.mkdir()
    case = resolve(mission)
    run = {**case, "seed": 0, "goal_stamp": 1.0, "run_id": path.name, "scenario_id": mission,
           "planner": planner, "status": "recorded", "installed_hashes": {"maps/warehouse_map.pgm": case["map_sha256"]}}
    if disposition == "runner_error":
        run.update(status="invalid", error="fixture startup failure")
    if disposition == "incomplete":
        run.update(status="running")
    (path / "run.json").write_text(json.dumps(run))
    if run["status"] != "recorded":
        return path
    (path / "bag").mkdir()
    topics = ["/clock", "/amr1/goal_pose", "/amr1/ground_truth", "/amr1/m2/contacts", "/amr1/cmd_vel"]
    bag_metadata = {"relative_file_paths": ["bag_0.db3"], "topics_with_message_count": [
        {"topic_metadata": {"name": name}, "message_count": 1} for name in topics]}
    (path / "bag/metadata.yaml").write_text(json.dumps({"rosbag2_bagfile_information": bag_metadata}))
    (path / "bag/bag_0.db3").write_bytes(b"unit test placeholder")
    (path / "actor.jsonl").write_text("" if case["actor"] is None else '{"action":"create"}\n')
    out = path / "episodes.csv"
    rec = EpisodeRecorder(label=planner, run_id=run["run_id"], timeout_sec=case["timeout_sec"],
                          events_path=out.with_suffix(".events.jsonl"))
    observed = disposition not in {"timeout_contact_unknown", "contact_unknown"}
    if observed:
        rec.on_contact(1., False, "gazebo_contacts", count=0, epoch=0)
    rec.start(1., tuple(case["goal"]), mission, seed=0, frame="map", goal_id="map:1.000000000")
    rec.on_path(1.01, tuple(case["goal"]), "map")
    rec.on_status(1.02, "NORMAL")
    if disposition in {"timeout_contact_unknown", "timeout"}:
        if observed:
            for i in range(1, 481):
                rec.on_contact(1. + i / 4, False, "gazebo_contacts", count=0, epoch=0)
        rec.tick(121.5)
    elif disposition == "collision":
        rec.on_contact(1.125, False, "gazebo_contacts", count=1, epoch=0)
    else:
        rec.on_pose(1.05, *case["goal"], frame="map")
        rec.on_status(1.1, "REACHED")
        if observed:
            rec.on_contact(1.125, False, "gazebo_contacts", count=0, epoch=0)
        rec.tick(1.5)
    rec.close(122.)
    rec.write_csv(out)
    return path


def inventory(*dirs):
    return summarize(dirs, {"train"}, {0})


def test_empty_and_partial_plan_have_null_rates(tmp_path):
    assert inventory()["planners"]["b0"]["counts"] == {"missing": 4}
    assert inventory()["planners"]["b0"]["completion_rate"] is None
    one = inventory(trial(tmp_path / "one"))["planners"]["b0"]
    assert one["counts"] == {"missing": 3, "success": 1} and one["completion_rate"] is None


@pytest.mark.parametrize("outcome", ["timeout", "collision", "contact_unknown", "timeout_contact_unknown",
                                     "runner_error", "incomplete"])
def test_actual_recorder_dispositions(tmp_path, outcome):
    result = inventory(trial(tmp_path / "sample", outcome))
    assert result["input_directories"][0]["disposition"] == outcome
    assert result["planners"]["b0"]["completion_rate"] is None


def test_complete_registered_conditions_and_separate_planners(tmp_path):
    names = ["train-static", "train-crossing", "train-headon", "train-blockage"]
    runs = [trial(tmp_path / name, "collision" if "headon" in name else "success", mission=name) for name in names]
    result = inventory(*runs)["planners"]
    assert result["b0"]["completion_rate"] == 0.75
    assert result["b1"]["scored"] == 0 and result["b1"]["completion_rate"] is None


def test_holdout_metadata_excluded_before_opening_recorder(tmp_path):
    out = tmp_path / "holdout"
    out.mkdir()
    (out / "run.json").write_text('{"id":"test-headon", "split":"test"}')
    result = inventory(out)
    assert result["input_directories"][0]["disposition"] == "excluded"
    assert result["planners"]["b0"]["counts"] == {"missing": 4}


@pytest.mark.parametrize("corrupt", ["bag", "event", "map", "identity", "header", "terminal"])
def test_missing_or_conflicting_evidence_rejected(tmp_path, corrupt):
    out = trial(tmp_path / "sample")
    meta = out / "run.json"
    events = out / "episodes.events.jsonl"
    if corrupt == "bag":
        (out / "bag/bag_0.db3").unlink()
    elif corrupt == "map":
        source = json.loads(meta.read_text())
        source["installed_hashes"]["maps/warehouse_map.pgm"] = "wrong"
        meta.write_text(json.dumps(source))
    elif corrupt == "identity":
        source = json.loads(meta.read_text())
        source["planner"] = "b1"
        meta.write_text(json.dumps(source))
    elif corrupt == "header":
        events.write_text("broken json")
    else:
        to_remove = "contact" if corrupt == "event" else "end"
        events.write_text("".join(line for line in events.read_text().splitlines(keepends=True)
                                  if json.loads(line)["kind"] != to_remove))
    assert inventory(out)["input_directories"][0]["disposition"] == "rejected"


def test_duplicate_condition_excludes_both_even_with_different_outcomes(tmp_path):
    a, b = trial(tmp_path / "a"), trial(tmp_path / "b", "collision")
    result = inventory(a, b)
    assert result["planners"]["b0"]["counts"] == {"missing": 3, "rejected": 1}
    assert all(e["disposition"] == "rejected" for e in result["input_directories"])


@pytest.mark.parametrize("mutation", ["remove_active", "active_before_path", "disarm", "backdated"])
def test_arrival_requires_fresh_activity_for_matching_path(tmp_path, mutation):
    out = trial(tmp_path / "sample")
    raw = out / "episodes.events.jsonl"
    events = [json.loads(line) for line in raw.read_text().splitlines()]
    idx = next(i for i, e in enumerate(events) if e.get("value") == "NORMAL")
    active = events.pop(idx)
    if mutation == "active_before_path":
        events.insert(next(i for i, e in enumerate(events) if e["kind"] == "path"), active)
    if mutation == "disarm":
        path = next(e for e in events if e["kind"] == "path")
        events[idx:idx] = [active, {**path, "matching": False}, path]
    if mutation == "backdated":
        events.insert(idx, {**active, "time": 0.99})
    raw.write_text("".join(json.dumps(e) + "\n" for e in events))
    assert inventory(out)["input_directories"][0]["disposition"] == "rejected"


def test_retry_after_infrastructure_failure(tmp_path):
    failed, retry = trial(tmp_path / "blocked", "runner_error"), trial(tmp_path / "retry")
    path = failed / "run.json"
    run = json.loads(path.read_text())
    del run["goal_stamp"]
    path.write_text(json.dumps(run))
    for dirs in ((failed, retry), (retry, failed)):
        report = inventory(*dirs)
        assert report["planners"]["b0"]["counts"] == {"missing": 3, "success": 1}
        assert sorted(e["disposition"] for e in report["input_directories"]) == ["runner_error", "success"]
    (failed / "episodes.events.jsonl").write_text("partial recorder output")
    assert inventory(failed, retry)["planners"]["b0"]["counts"] == {"missing": 3, "rejected": 1}


def test_duplicate_run_id_prevents_retry(tmp_path):
    failed, retry = trial(tmp_path / "failed", "runner_error"), trial(tmp_path / "retry")
    meta = failed / "run.json"
    run = json.loads(meta.read_text())
    del run["goal_stamp"]
    run["run_id"] = retry.name
    meta.write_text(json.dumps(run))
    assert inventory(failed, retry)["planners"]["b0"]["counts"] == {"missing": 3, "rejected": 1}
