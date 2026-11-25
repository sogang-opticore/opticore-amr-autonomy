from copy import deepcopy

import pytest

from summarize import expected_reports, paired, summarize
from evaluate import conditions


def sample_reports():
    reports = []
    for suite, name, seed, shield in sorted(expected_reports()):
        rows = [{"mission": mission, "sensor_seed": sensor_seed, "profile": profile,
                 "outcome": "timeout", "stalled_s": 4.0, "ticks": 2400,
                 "interventions": 3, "elapsed_s": 120.0, "route_rejoined": None,
                 "stopped_at_release": False if mission == "train-blockage" else None,
                 "release_to_restart_s": None}
                for mission, sensor_seed, profile in conditions(suite)]
        reports.append({"schema": "opticore-m4-development-v1", "suite": suite, "variant": name,
                        "training_seed": seed, "shield": shield, "rows": rows,
                        "source_sha256": "evaluator", "dependencies_sha256": {},
                        "bundle_sha256": f"{name}-{seed}-bundle", "checkpoint_sha256": f"{name}-{seed}-ckpt",
                        "manifest_sha256": "cases", "map_sha256": "map", "profile_seed": 42, "profiles": {}})
    return reports


def test_gate_uses_outcomes_and_all_training_seeds():
    reports = sample_reports()
    full = [r for r in reports if (r["suite"], r["variant"], r["shield"]) == ("primary", "full", True)]
    for run in full:
        row = next(r for r in run["rows"] if r["mission"] == "train-crossing")
        row["outcome"] = "success"
    summary = summarize(reports)
    assert summary["candidate"] == "full"
    assert summary["configurations"]["full"]["gate"]["dynamic_delta_pp"] == pytest.approx(100 / 12)
    assert summary["configurations"]["full"]["overall"]["n"] == 48
    assert summary["total_episodes"] == 312
    # A missing observation must never improve the candidate's rate.
    full[0]["rows"][0]["outcome"] = "invalid"
    assert summarize(reports)["candidate"] is None


def test_pair_matching_does_not_use_row_order_or_failed_run_times():
    refs = sample_reports()[0]["rows"]
    current = list(reversed(deepcopy(refs)))
    refs[0].update(outcome="success", elapsed_s=10.0)
    current[-1].update(outcome="success", elapsed_s=11.5)
    current[-2].update(outcome="success", elapsed_s=8.0)
    result = paired(current, refs)
    assert result["success_gains"] == 1 and result["both_success_n"] == 1
    assert result["mean_elapsed_delta_s_both_success"] == pytest.approx(1.5)


@pytest.mark.parametrize("change", ["repeat", "missing", "conditions", "profile", "changed_bundle"])
def test_rejects_mislabeled_or_incomplete_matrix(change):
    reports = sample_reports()
    if change == "repeat":
        reports.append(reports[-1])
    elif change == "missing":
        reports.pop()
    elif change == "conditions":
        reports[0]["rows"][0] = reports[0]["rows"][1]
    elif change == "profile":
        reports[0]["profiles"] = {"mistake": True}
    else:
        reports[0]["bundle_sha256"] = "different"
    with pytest.raises(ValueError):
        summarize(reports)
