import pytest

from amr_lite.config import resolve_env_config
from amr_lite.experiments.paired import gap_report, load_policy_spec, read_rows, run_lite_paired, write_gap_report, write_rows


def test_lite_paired_rows_match_the_gazebo_runner_schema(tmp_path):
    policy = load_policy_spec("rule", resolve_env_config("opticore_gazebo"))
    rows = run_lite_paired(policy, "warehouse_v1", [0], label="rule", legs=[5])
    assert [(row["scenario_id"], row["seed"], row["policy"]) for row in rows] == [("WH-warehouse_v1-05", 0, "rule")]
    assert rows[0]["success"] == 1 and rows[0]["episode_time"] < 120.0
    stored = read_rows(write_rows(tmp_path / "lite.csv", rows))
    assert list(stored[0]) == ["policy", "scenario_id", "seed", "success", "collision", "truncated", "failure_type",
                               "episode_time", "path_length", "minimum_clearance", "shield_intervention_rate"]
    with pytest.raises(ValueError):
        load_policy_spec("teleop", {})


def test_gap_report_compares_legs(tmp_path):
    def row(policy, leg, seed, success, time, failure=""):
        return {"policy": policy, "scenario_id": leg, "seed": seed, "success": success, "collision": 0,
                "truncated": int(failure == "TIMEOUT"), "failure_type": failure, "episode_time": time,
                "path_length": 10.0, "minimum_clearance": 0.3, "shield_intervention_rate": 0.0}
    lite = [row("lite", "WH-a-00", s, 1, 20.0) for s in range(4)] + [row("lite", "WH-a-01", s, 1, 30.0) for s in range(4)]
    gazebo = [row("gz", "WH-a-00", 0, 1, 22.0), row("gz", "WH-a-00", 1, 1, 22.0),
              row("gz", "WH-a-01", 0, 0, 120.0, "TIMEOUT"), row("gz", "WH-a-01", 1, 1, 33.0)]
    report = gap_report(lite, gazebo)
    assert report["overall"]["success_delta"] == pytest.approx(-0.25)
    legs = {leg["scenario_id"]: leg for leg in report["legs"]}
    assert legs["WH-a-01"]["candidate_failures"] == {"TIMEOUT": 1}
    assert report["overall"]["time_ratio"] == pytest.approx((22.0 + 33.0) / (20.0 + 30.0))
    text = write_gap_report(tmp_path / "GAP.md", report, "lite", "gz").read_text()
    assert "WH-a-01" in text and "TIMEOUT 1" in text
