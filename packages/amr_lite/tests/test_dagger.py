import csv

import numpy as np

from amr_lite.config import resolve_env_config
from amr_lite.learning.collect_teacher import collect
from amr_lite.learning.dagger_loop import run_dagger
from amr_lite.learning.dataset import load_dataset
from amr_lite.learning.networks import load_policy
from amr_lite.learning.train_bc import train as train_bc


class _AlwaysForward:
    def predict(self, observation, deterministic=True):
        return np.asarray([1.0, 0.0], dtype=np.float32)


def test_beta_one_executes_the_teacher(tmp_path):
    teacher = collect(tmp_path / "teacher", episodes=1, max_steps=25, seed=4, scenarios=["S3"])
    mixed = collect(tmp_path / "mixed", episodes=1, max_steps=25, seed=4, scenarios=["S3"],
                    learner=_AlwaysForward(), beta=1.0)
    assert np.array_equal(load_dataset(tmp_path / "teacher").observations,
                          load_dataset(tmp_path / "mixed").observations)
    assert teacher and mixed


def test_risk_weighting_changes_training_only_when_enabled(tmp_path):
    collect(tmp_path / "data", episodes=2, max_steps=40, seed=4, scenarios=["S3"])
    assert load_dataset(tmp_path / "data").minimum_clearance is not None
    base = {"seed": 1, "hidden_size": 16, "batch_size": 8, "learning_rate": 0.001,
            "epochs_smoke": 1, "epochs_full": 1, "validation_fraction": 0.5}
    plain = train_bc(tmp_path / "data", tmp_path / "plain.pt", config=base)
    unit = train_bc(tmp_path / "data", tmp_path / "unit.pt", config={**base, "risk_weight": 1.0})
    risky = train_bc(tmp_path / "data", tmp_path / "risky.pt",
                     config={**base, "risk_weight": 5.0, "risk_clearance": 2.0})
    assert plain == unit
    assert risky["mse"] != plain["mse"]


def test_dagger_smoke_selects_a_round(tmp_path):
    result = run_dagger(tmp_path, smoke=True, verbose=False)
    rows = list(csv.DictReader((tmp_path / "rounds.csv").open()))
    assert [int(row["round"]) for row in rows] == [0, 1]
    assert float(rows[1]["beta"]) == 0.5
    assert result["best_round"] in (0, 1)
    assert load_policy(str(tmp_path / "best.pt")).predict(np.zeros(296, dtype=np.float32)).shape == (2,)
    assert result["robot_profile"] == "opticore_gazebo"
    assert resolve_env_config("opticore_gazebo")["w_max"] == 1.5
