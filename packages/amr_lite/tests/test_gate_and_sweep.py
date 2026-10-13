import json

import pytest

from amr_lite.config import load_config
from amr_lite.evaluation.metrics import paired_comparisons
from amr_lite.experiments.gate import evaluate_gate, write_gate_report
from amr_lite.experiments.sweep import Sweep, expand_trials, rank_variants


def _row(family, role, training_seed, scenario, seed, success, shield=False, failure="", time=10.0):
    return {"policy": f"{family}_{training_seed}_{role}", "policy_family": family, "checkpoint_role": role,
            "training_seed": training_seed, "scenario_id": scenario, "seed": seed, "shield_enabled": shield,
            "success": int(success), "collision": int(not success and failure.startswith("COLLISION")),
            "truncated": int(not success and not failure.startswith("COLLISION")),
            "failure_type": "" if success else failure, "episode_time": time, "steps": 100,
            "minimum_clearance": 0.3}


def _dataset(candidate_dynamic_success: bool, candidate_failure: str = "COLLISION_DYNAMIC",
             candidate_time: float = 9.0):
    rows = []
    for seed in range(20):
        for scenario in ("S3", "D1"):
            rows.append(_row("lite_rule_baseline", "fixed", "", scenario, seed, scenario == "S3",
                             failure="COLLISION_DYNAMIC"))
            for training_seed in (7, 17):
                success = scenario == "S3" or candidate_dynamic_success
                rows.append(_row("cand", "best", training_seed, scenario, seed, success, failure=candidate_failure,
                                 time=candidate_time))
                rows.append(_row("cand", "final", training_seed, scenario, seed, success, failure=candidate_failure))
                rows.append(_row("cand", "best", training_seed, scenario, seed, success, shield=True,
                                 failure=candidate_failure))
    return rows


def test_gate_passes_a_candidate_that_solves_dynamic_scenarios(tmp_path):
    rows = _dataset(candidate_dynamic_success=True)
    gate = evaluate_gate(rows, paired_comparisons(rows, bootstrap_resamples=50), "cand", "best")
    assert gate["passed"], gate["checks"]
    report = write_gate_report(tmp_path / "GATE.md", gate).read_text()
    assert "PASS" in report and "success_delta_ci_low" in report


def test_gate_fails_without_improvement_or_with_behaviour_flags():
    # Same success and the same speed as the reference: neither improvement criterion holds.
    no_improvement = _dataset(candidate_dynamic_success=False, candidate_time=10.0)
    gate = evaluate_gate(no_improvement, paired_comparisons(no_improvement, bootstrap_resamples=50), "cand")
    checks = {check["name"]: check for check in gate["checks"]}
    assert not gate["passed"] and checks["improvement_dynamic_success_delta"]["passed"] is False
    circling = _dataset(candidate_dynamic_success=False, candidate_failure="CIRCLING", candidate_time=10.0)
    gate = evaluate_gate(circling, paired_comparisons(circling, bootstrap_resamples=50), "cand")
    checks = {check["name"]: check for check in gate["checks"]}
    assert checks["behaviour_circling_rate"]["passed"] is False
    with pytest.raises(ValueError):
        evaluate_gate(circling, [], "missing_family")


def test_expand_trials_applies_variants_grid_budget_and_seeds():
    spec = {"name": "t", "base_config": "ppo_v2", "training_seeds": [7, 17],
            "budget": {"total_timesteps": 1234},
            "variants": {"residual": {"policy": {"mode": "residual"}}, "direct": {"policy": {"mode": "direct"}}},
            "grid": {"network.encoder": ["mlp", "cnn"]}}
    trials = expand_trials(spec)
    assert len(trials) == 8
    trial = next(item for item in trials if item["id"] == "direct__encoder=cnn-s17")
    assert trial["config"]["policy"]["mode"] == "direct" and trial["config"]["network"]["encoder"] == "cnn"
    assert trial["config"]["seed"] == 17 and trial["config"]["total_timesteps"] == 1234
    assert load_config("ppo_v2")["network"]["encoder"] == "mlp"  # base config untouched


def test_rank_variants_applies_constraints_before_success():
    records = [
        {"variant": "risky", "seed": 7, "role": "best", "success_rate": 0.95, "collision_rate": 0.2,
         "timeout_rate": 0.0, "mean_time": 9.0, "shield_intervention_rate": 0.0},
        {"variant": "safe", "seed": 7, "role": "best", "success_rate": 0.85, "collision_rate": 0.05,
         "timeout_rate": 0.0, "mean_time": 10.0, "shield_intervention_rate": 0.0},
        {"variant": "safe", "seed": 7, "role": "final", "success_rate": 0.80, "collision_rate": 0.05,
         "timeout_rate": 0.0, "mean_time": 10.0, "shield_intervention_rate": 0.0},
    ]
    leaderboard = rank_variants(records, {"constraints": {"collision_rate_max": 0.1}})
    assert [row["variant"] for row in leaderboard] == ["safe", "risky"]
    assert leaderboard[0]["eligible"] and not leaderboard[1]["eligible"]
    assert leaderboard[0]["max_best_final_gap"] == pytest.approx(0.05)


def test_sweep_trains_selects_and_tests_only_once(tmp_path):
    spec = {"name": "tiny", "trainer": "ppo_v2", "base_config": "ppo_v2", "training_seeds": [7],
            "variants": {"residual": {"network": {"hidden_sizes": [16, 16]}}},
            "selection": {"protocol": "evaluation_selection_v2"}, "test": {"protocol": "evaluation_test_v2"}}
    sweep = Sweep(spec, tmp_path, smoke=True)
    sweep.train()
    assert sweep.trial_complete(expand_trials(spec)[0])
    selection = sweep.select()
    assert selection["winner"] == "residual"
    assert (tmp_path / "leaderboard.csv").exists() and (tmp_path / "SELECTION.md").exists()
    result = sweep.test()
    assert set(json.loads((tmp_path / "test" / "gate.json").read_text())) >= {"passed", "checks"}
    assert result["winner"] == "residual"
    with pytest.raises(RuntimeError):
        sweep.test()
