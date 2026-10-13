"""Experiment sweeps: variants x training seeds -> selection -> one-time test -> G4 gate.

A sweep spec (``configs/sweeps/*.yaml``) names a trainer, a base config,
variants (and an optional grid), training seeds and the pre-registered
selection rule. Stages:

``plan``   expand trials and write ``plan.json``
``train``  run unfinished trials as subprocesses (parallel, resumable)
``select`` evaluate trial checkpoints on the selection protocol and rank variants
``test``   evaluate the chosen variant and the rule baseline on the test protocol
           once, with paired CIs, then apply the G4 gate
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

from amr_lite.config import ROOT, deep_merge, load_config

TRAINERS = {
    "ppo_v2": {"module": "amr_lite.cli.train_rl", "checkpoints": {"best": "best.pt", "final": "final.pt"}},
    "dagger": {"module": "amr_lite.cli.dagger", "checkpoints": {"best": "best.pt"}},
}


def _set_path(config: dict[str, Any], dotted: str, value: Any) -> None:
    cursor = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def expand_trials(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Variant x grid x seed trials with fully resolved trainer configs."""
    base = load_config(str(spec["base_config"]))
    variants = dict(spec.get("variants") or {"default": {}})
    grid = dict(spec.get("grid") or {})
    combos = [dict(zip(grid, values)) for values in itertools.product(*grid.values())] if grid else [{}]
    trials = []
    for variant_name, overrides in variants.items():
        for combo in combos:
            name = variant_name + "".join(f"__{key.split('.')[-1]}={value}" for key, value in combo.items())
            for seed in spec["training_seeds"]:
                config = deep_merge(base, overrides)
                for key, value in combo.items():
                    _set_path(config, key, value)
                for key, value in (spec.get("budget") or {}).items():
                    _set_path(config, key, value)
                config["seed"] = int(seed)
                trials.append({"variant": name, "seed": int(seed), "id": f"{name}-s{seed}", "config": config})
    return trials


class Sweep:
    def __init__(self, spec: dict[str, Any], output: str | Path | None = None, smoke: bool = False):
        self.spec, self.smoke = spec, smoke
        if spec.get("trainer", "ppo_v2") not in TRAINERS:
            raise ValueError(f"Unknown trainer {spec.get('trainer')!r}")
        self.trainer = TRAINERS[spec.get("trainer", "ppo_v2")]
        self.output = Path(output or ROOT / "artifacts" / "sweeps" / spec["name"])

    # ---------------------------------------------------------------- plan
    def plan(self) -> list[dict[str, Any]]:
        trials = expand_trials(self.spec)
        self.output.mkdir(parents=True, exist_ok=True)
        for trial in trials:
            directory = self.trial_dir(trial)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "trial_config.json").write_text(json.dumps(trial["config"], indent=2, default=str),
                                                         encoding="utf-8")
        (self.output / "plan.json").write_text(json.dumps(
            {"spec": self.spec, "smoke": self.smoke,
             "trials": [{key: trial[key] for key in ("id", "variant", "seed")} for trial in trials]},
            indent=2, default=str), encoding="utf-8")
        return trials

    def trial_dir(self, trial: dict[str, Any]) -> Path:
        return self.output / "trials" / trial["id"]

    def trial_complete(self, trial: dict[str, Any]) -> bool:
        result = self.trial_dir(trial) / "result.json"
        if not result.exists():
            return False
        payload = json.loads(result.read_text(encoding="utf-8"))
        return payload.get("status", "complete") == "complete" and (
            self.trial_dir(trial) / self.trainer["checkpoints"]["best"]).exists()

    # --------------------------------------------------------------- train
    def _run_trial(self, trial: dict[str, Any]) -> int:
        directory = self.trial_dir(trial)
        command = [sys.executable, "-m", self.trainer["module"], "--config",
                   str(directory / "trial_config.json"), "--output", str(directory)]
        if self.smoke:
            command.append("--smoke")
        if (directory / "state.pt").exists():
            command.append("--resume")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), environment.get("PYTHONPATH")]))
        environment.setdefault("MPLCONFIGDIR", str(ROOT / "artifacts" / ".matplotlib"))
        with (directory / "train.log").open("a", encoding="utf-8") as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                  cwd=str(ROOT), check=False).returncode

    def train(self, parallel: int | None = None) -> dict[str, int]:
        trials = self.plan()
        pending = [trial for trial in trials if not self.trial_complete(trial)]
        workers = max(1, int(parallel or self.spec.get("parallel_trials", 1)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            codes = dict(zip([trial["id"] for trial in pending], pool.map(self._run_trial, pending)))
        failed = {name: code for name, code in codes.items() if code != 0}
        if failed:
            raise RuntimeError(f"Trials failed (see train.log): {failed}")
        return codes

    # -------------------------------------------------------------- select
    def _selection_protocol(self) -> tuple[dict[str, Any], dict[str, Any]]:
        selection = dict(self.spec.get("selection") or {})
        protocol = load_config(str(selection.get("protocol", "evaluation_selection_v2")))
        if self.smoke:
            protocol.update({"scenarios": list(protocol["scenarios"])[:2], "episodes_per_scenario": 1})
        return selection, protocol

    def select(self) -> dict[str, Any]:
        from amr_lite.learning.networks import load_policy
        from amr_lite.rl.evaluation import evaluate_protocol

        trials = expand_trials(self.spec)
        missing = [trial["id"] for trial in trials if not self.trial_complete(trial)]
        if missing:
            raise RuntimeError(f"Train these trials first: {missing}")
        selection, protocol = self._selection_protocol()
        workers = 1 if self.smoke else int(selection.get("workers", 4))
        backend = "sync" if self.smoke else str(selection.get("backend", "subprocess"))
        roles = ["best"] + (["final"] if selection.get("evaluate_final", True)
                            and "final" in self.trainer["checkpoints"] else [])
        records = []
        for trial in trials:
            for role in roles:
                cache = self.trial_dir(trial) / f"selection_{role}.json"
                if cache.exists():
                    metrics = json.loads(cache.read_text(encoding="utf-8"))
                else:
                    policy = load_policy(str(self.trial_dir(trial) / self.trainer["checkpoints"][role]))
                    metrics, _ = evaluate_protocol(policy, protocol, workers=workers, backend=backend)
                    metrics = {key: value for key, value in metrics.items() if key != "by_shield"}
                    cache.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                records.append({"variant": trial["variant"], "seed": trial["seed"], "role": role, **metrics})
        leaderboard = rank_variants(records, selection.get("rule") or {})
        _write_csv(self.output / "selection_trials.csv", records)
        _write_csv(self.output / "leaderboard.csv", leaderboard)
        winner = next((row["variant"] for row in leaderboard if row["eligible"]), None)
        result = {"winner": winner, "protocol": protocol.get("protocol", "v1"), "leaderboard": leaderboard}
        (self.output / "selection.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        _write_selection_report(self.output / "SELECTION.md", self.spec, result)
        return result

    # ---------------------------------------------------------------- test
    def test(self, force: bool = False) -> dict[str, Any]:
        from amr_lite.evaluation import evaluate
        from amr_lite.evaluation.metrics import paired_comparisons
        from amr_lite.experiments.gate import evaluate_gate, write_gate_report
        from amr_lite.experiments.manifest import sha256_file
        from amr_lite.learning.networks import load_policy
        from amr_lite.planning.rule_baseline import RuleBasedPlanner
        from amr_lite.rl.evaluation import protocol_env_config

        test_dir = self.output / "test"
        if (test_dir / "DONE").exists() and not force:
            raise RuntimeError("The test split was already evaluated for this sweep. Re-running after seeing "
                               "results needs a new protocol version (use --force only to repair a crash).")
        selection_file = self.output / "selection.json"
        if not selection_file.exists():
            raise RuntimeError("Run the select stage first")
        winner = json.loads(selection_file.read_text(encoding="utf-8"))["winner"]
        if winner is None:
            raise RuntimeError("No variant met the selection constraints; nothing to test")
        settings = dict(self.spec.get("test") or {})
        protocol = load_config(str(settings.get("protocol", "evaluation_test_v2")))
        protocol["workers"] = 1 if self.smoke else int(settings.get("workers", 4))
        protocol["backend"] = "sync" if self.smoke else str(settings.get("backend", "subprocess"))
        if self.smoke:
            protocol.update({"scenarios": list(protocol["scenarios"])[:2], "episodes_per_scenario": 1,
                             "bootstrap_resamples": 50})
        env_config = protocol_env_config(protocol)
        baseline = RuleBasedPlanner(env_config, load_config("baseline"))
        baseline.policy_family, baseline.checkpoint_role, baseline.training_seed = "lite_rule_baseline", "fixed", ""
        policies: dict[str, Any] = {"lite_rule_baseline": baseline}
        roles = ["best"] + (["final"] if settings.get("include_final", True)
                            and "final" in self.trainer["checkpoints"] else [])
        for trial in (item for item in expand_trials(self.spec) if item["variant"] == winner):
            for role in roles:
                path = self.trial_dir(trial) / self.trainer["checkpoints"][role]
                policy = load_policy(str(path))
                policy.policy_family, policy.checkpoint_role = winner, role
                policy.training_seed, policy.checkpoint_sha256 = trial["seed"], sha256_file(path)
                policies[f"{trial['id']}_{role}"] = policy
        rows, summary = evaluate(policies, test_dir, protocol)
        comparisons = paired_comparisons(rows, bootstrap_resamples=int(protocol.get("bootstrap_resamples", 2000)),
                                         bootstrap_seed=int(protocol.get("bootstrap_seed", 20260914)))
        gate = evaluate_gate(rows, comparisons, winner, "best", load_config(str(settings.get("gate", "gates/g4"))))
        (test_dir / "gate.json").write_text(json.dumps(gate, indent=2, default=str), encoding="utf-8")
        write_gate_report(test_dir / "GATE_REPORT.md", gate)
        (test_dir / "DONE").write_text(json.dumps({"winner": winner, "smoke": self.smoke}), encoding="utf-8")
        return {"winner": winner, "gate": gate, "summary": summary}


def rank_variants(records: list[dict[str, Any]], rule: dict[str, Any]) -> list[dict[str, Any]]:
    """Pre-registered ranking: constraints on medians, then median and worst success, collisions, time."""
    constraints = rule.get("constraints") or {}
    rows = []
    for variant in sorted({record["variant"] for record in records}):
        best = [r for r in records if r["variant"] == variant and r["role"] == "best"]
        final = {r["seed"]: r for r in records if r["variant"] == variant and r["role"] == "final"}
        success = [float(r["success_rate"]) for r in best]
        row = {
            "variant": variant, "seeds": len(best),
            "median_success": statistics.median(success), "worst_success": min(success),
            "median_collision": statistics.median(float(r["collision_rate"]) for r in best),
            "median_timeout": statistics.median(float(r["timeout_rate"]) for r in best),
            "median_time": statistics.median(float(r["mean_time"]) for r in best),
            "median_shield_intervention_rate": statistics.median(float(r["shield_intervention_rate"]) for r in best),
            "max_best_final_gap": max((float(r["success_rate"]) - float(final[r["seed"]]["success_rate"])
                                       for r in best if r["seed"] in final), default=float("nan")),
        }
        row["eligible"] = (row["median_collision"] <= float(constraints.get("collision_rate_max", 1.0))
                           and row["median_timeout"] <= float(constraints.get("timeout_rate_max", 1.0)))
        rows.append(row)
    rows.sort(key=lambda r: (r["eligible"], r["median_success"], r["worst_success"],
                             -r["median_collision"], -r["median_time"]), reverse=True)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row if not isinstance(row[key], dict)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_selection_report(path: Path, spec: dict[str, Any], result: dict[str, Any]) -> None:
    lines = [f"# Sweep {spec['name']} — selection", "",
             f"- Protocol: {result['protocol']} selection split", f"- Winner: **{result['winner']}**", "",
             "| Variant | Seeds | Median success | Worst | Median collision | Median timeout | "
             "Median time (s) | Max best-final gap | Eligible |",
             "|---|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    for row in result["leaderboard"]:
        lines.append(f"| {row['variant']} | {row['seeds']} | {row['median_success']:.3f} | "
                     f"{row['worst_success']:.3f} | {row['median_collision']:.3f} | {row['median_timeout']:.3f} | "
                     f"{row['median_time']:.2f} | {row['max_best_final_gap']:.3f} | "
                     f"{'yes' if row['eligible'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sweep_spec(name_or_path: str) -> dict[str, Any]:
    path = Path(name_or_path)
    if path.suffix in {".yaml", ".yml"} and path.exists():
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8"))
    return load_config(f"sweeps/{name_or_path}")


def apply_smoke_budget(spec: dict[str, Any]) -> dict[str, Any]:
    """Shrink a spec to two seeds and one variant per grid value for smoke checks."""
    smoke = deepcopy(spec)
    smoke["training_seeds"] = list(spec["training_seeds"])[:2]
    smoke["parallel_trials"] = 1
    return smoke
