import json
import subprocess
import sys

import numpy as np
import pytest

from amr_lite.config import ROOT, deep_merge, load_config, resolve_env_config
from amr_lite.deploy.export import export_bundle, golden_observations
from amr_lite.deploy.runtime import BundleError, PolicyRuntime
from amr_lite.learning.collect_teacher import collect
from amr_lite.learning.networks import load_policy
from amr_lite.learning.train_bc import train as train_bc
from amr_lite.learning.train_ppo import train as train_m1_ppo
from amr_lite.rl.ppo import train as train_ppo_v2


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    root = tmp_path_factory.mktemp("checkpoints")
    collect(root / "data", episodes=2, max_steps=30, seed=3, scenarios=["S0", "S3"])
    train_bc(root / "data", root / "bc.pt", config={"seed": 1, "hidden_size": 16, "batch_size": 8,
                                                     "learning_rate": 0.001, "epochs_smoke": 1,
                                                     "epochs_full": 1, "validation_fraction": 0.5})
    train_m1_ppo(root / "stable.pt", smoke=False, warm_start_dataset=root / "data", warm_start_only=True,
                 stable_architecture=True,
                 config={"seed": 1, "hidden_size": 16, "learning_rate": 0.0003, "minibatch_size": 8,
                         "warm_start_learning_rate": 0.001, "warm_start_epochs": 1, "warm_start_log_std": -1.5})
    base = deep_merge(load_config("ppo_v2"), {
        "network": {"hidden_sizes": [32, 32], "cnn": {"channels": [4, 8], "state_hidden": 8}},
        "vector_env": {"num_envs": 2, "backend": "sync"}, "evaluation": {"interval_updates": 0}})
    train_ppo_v2(root / "residual", base, smoke=True, verbose=False)
    train_ppo_v2(root / "cnn", deep_merge(base, {"policy": {"mode": "direct"}, "network": {"encoder": "cnn"},
                                                 "environment_overrides": {"history_mode": "newest"}}),
                 smoke=True, verbose=False)
    return {"bc": root / "bc.pt", "ppo_stable": root / "stable.pt",
            "residual": root / "residual" / "final.pt", "cnn": root / "cnn" / "final.pt"}


@pytest.mark.parametrize("name", ["bc", "ppo_stable", "residual", "cnn"])
def test_numpy_runtime_reproduces_torch_policy(checkpoints, tmp_path, name):
    manifest = export_bundle(checkpoints[name], tmp_path / "bundle", golden_count=16)
    runtime = PolicyRuntime.load(tmp_path / "bundle")
    assert runtime.verify_golden() <= 1e-5
    observations = golden_observations(resolve_env_config("opticore_gazebo"), 24, seed=11)
    torch_actions = load_policy(str(checkpoints[name])).predict_batch(observations)
    assert np.max(np.abs(runtime.predict_batch(observations) - torch_actions)) <= 1e-5
    assert manifest["observation_contract"]["version"] == "amr-lite-obs-v1"
    expected_source = "checkpoint" if name in {"residual", "cnn"} else "assumed"
    assert manifest["robot_profile"]["source"] == expected_source


def test_bundle_records_mode_profile_and_history(checkpoints, tmp_path):
    residual = export_bundle(checkpoints["residual"], tmp_path / "residual", golden_count=8)
    assert residual["policy"]["mode"] == "residual"
    assert residual["policy"]["base_policy"]["env_config"]["w_max"] == 1.5
    assert residual["robot_profile"]["name"] == "opticore_gazebo"
    export_bundle(checkpoints["cnn"], tmp_path / "cnn", golden_count=8)
    cnn = PolicyRuntime.load(tmp_path / "cnn")
    assert cnn.network["type"] == "scan_cnn"
    assert cnn.observation_overrides == {"history_mode": "newest"}


def _rewrite(bundle, mutate):
    manifest = json.loads((bundle / "bundle.json").read_text())
    mutate(manifest)
    (bundle / "bundle.json").write_text(json.dumps(manifest))


def test_tampered_or_incompatible_bundles_are_refused(checkpoints, tmp_path):
    bundle = tmp_path / "bundle"
    export_bundle(checkpoints["residual"], bundle, golden_count=8)
    original = (bundle / "bundle.json").read_text()

    _rewrite(bundle, lambda m: m["observation_contract"].update(version="amr-lite-obs-v0"))
    with pytest.raises(BundleError) as error:
        PolicyRuntime.load(bundle)
    assert error.value.code == "CONTRACT_MISMATCH"

    (bundle / "bundle.json").write_text(original)
    with np.load(bundle / "golden.npz") as golden:
        observations, actions = golden["observations"], golden["actions"]
    np.savez(bundle / "golden.npz", observations=observations, actions=actions + 0.01)
    with pytest.raises(BundleError) as error:
        PolicyRuntime.load(bundle)
    assert error.value.code == "BUNDLE_LOAD_FAILED"  # SHA-256 no longer matches

    import hashlib
    _rewrite(bundle, lambda m: m["files"].update(
        {"golden.npz": hashlib.sha256((bundle / "golden.npz").read_bytes()).hexdigest()}))
    with pytest.raises(BundleError) as error:
        PolicyRuntime.load(bundle)
    assert error.value.code == "GOLDEN_PARITY_FAILED"
    assert PolicyRuntime.load(bundle, verify=False).predict(observations[0]).shape == (2,)


def test_runtime_does_not_import_torch(checkpoints, tmp_path):
    export_bundle(checkpoints["residual"], tmp_path / "bundle", golden_count=8)
    script = ("import sys; from amr_lite.deploy.runtime import PolicyRuntime; "
              f"PolicyRuntime.load({str(tmp_path / 'bundle')!r}); print('torch' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT,
                            env={"PYTHONPATH": str(ROOT), "PATH": ""}, check=True)
    assert result.stdout.strip() == "False"
