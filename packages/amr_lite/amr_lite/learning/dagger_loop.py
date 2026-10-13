"""BC -> DAgger rounds with selection-split evaluation (ROADMAP M4, experiment E1).

Round 0 behaviour-clones teacher episodes. Each later round runs the
previous learner, executes the teacher action with probability
``beta * beta_decay**(round - 1)``, relabels every visited state with the
teacher, aggregates all data and retrains from scratch. The round with the
best pre-registered selection score is copied to ``best.pt``.

The teacher is ``lite_rule_baseline``: DAgger reduces covariate shift but
cannot exceed the teacher; improving on it is the job of PPO v2.
"""
from __future__ import annotations

import csv
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

from amr_lite.config import load_config, resolve_env_config
from amr_lite.rl.evaluation import evaluate_protocol, selection_score

from .collect_teacher import collect
from .networks import load_policy
from .train_bc import train as train_bc


def run_dagger(output: str | Path, config: dict[str, Any] | None = None, *, smoke: bool = False,
               verbose: bool = True) -> dict[str, Any]:
    cfg = deepcopy(config or load_config("dagger"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env_config = resolve_env_config(cfg.get("robot_profile"), cfg.get("environment_overrides"))
    teacher, dagger, bc = dict(cfg["teacher"]), dict(cfg["dagger"]), dict(cfg["bc"])
    evaluation = dict(cfg["evaluation"])
    rounds = int(cfg["rounds"])
    if smoke:
        teacher.update({"episodes": 2, "max_steps": 40})
        dagger.update({"episodes": 1, "max_steps": 40})
        bc.update({"epochs_full": 1, "hidden_size": 32})
        evaluation.update({"episodes_per_scenario": 1, "workers": 1, "backend": "sync", "scenarios": ["S0"]})
        rounds = min(rounds, 1)
    bc.setdefault("epochs_smoke", 1)
    bc["seed"] = int(cfg["seed"])
    protocol = load_config(str(evaluation["protocol"]))
    if "scenarios" in evaluation:
        protocol["scenarios"] = list(evaluation["scenarios"])
    datasets: list[Path] = []
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for round_index in range(rounds + 1):
        directory = output / f"round_{round_index}"
        if round_index == 0:
            dataset = directory / "teacher"
            files = collect(dataset, int(teacher["episodes"]), int(teacher["max_steps"]), int(cfg["seed"]),
                            list(teacher["scenarios"]), env_config=env_config)
            beta = 1.0
        else:
            dataset = directory / "dagger"
            beta = float(dagger["beta"]) * float(dagger.get("beta_decay", 1.0)) ** (round_index - 1)
            learner = load_policy(str(output / f"round_{round_index - 1}" / "bc.pt"))
            files = collect(dataset, int(dagger["episodes"]), int(dagger["max_steps"]),
                            int(cfg["seed"]) + 1000 * round_index, list(dagger["scenarios"]),
                            learner=learner, dagger_risk_only=bool(dagger.get("risk_only", False)),
                            env_config=env_config, beta=beta)
        if files:
            datasets.append(dataset)
        checkpoint = directory / "bc.pt"
        metrics = train_bc([str(item) for item in datasets], checkpoint, smoke=False, config=bc)
        selection, _ = evaluate_protocol(
            load_policy(str(checkpoint)), protocol, robot_profile=cfg.get("robot_profile"),
            episodes_per_scenario=evaluation.get("episodes_per_scenario"),
            workers=int(evaluation.get("workers", 1)), backend=str(evaluation.get("backend", "sync")))
        record = {"round": round_index, "beta": beta, "chunks": len(files), "datasets": len(datasets),
                  "checkpoint": str(checkpoint), **{f"bc_{key}": value for key, value in metrics.items()},
                  **{f"eval_{key}": value for key, value in selection.items()
                     if key not in {"by_shield", "failure_counts"}}}
        history.append(record)
        if best is None or selection_score(selection) > selection_score(best["selection"]):
            best = {"round": round_index, "checkpoint": checkpoint, "selection": selection}
        if verbose:
            print(f"DAgger round {round_index}/{rounds} | beta={beta:.2f} | samples={metrics['train_samples']} "
                  f"| val_mse={metrics['mse']:.4f} | success={selection['success_rate']:.3f} "
                  f"| collision={selection['collision_rate']:.3f}", flush=True)
    assert best is not None
    shutil.copyfile(best["checkpoint"], output / "best.pt")
    with (output / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    result = {"output": str(output), "best_round": best["round"], "best_checkpoint": str(output / "best.pt"),
              "best_selection": best["selection"], "datasets": [str(item) for item in datasets],
              "robot_profile": env_config.get("robot_profile"), "smoke": smoke}
    (output / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result
