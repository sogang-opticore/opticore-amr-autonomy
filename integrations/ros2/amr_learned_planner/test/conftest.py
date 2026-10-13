"""Make the package and AMR-Lite importable when tests run from the repository.

Under colcon the package is installed and AMR-Lite must be installed with
``pip install --no-deps -e packages/amr_lite``; the path setup below is then
a no-op.
"""
import sys
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE_ROOT.parents[2]
for path in (PACKAGE_ROOT, REPOSITORY / "packages" / "amr_lite"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    """A real exported bundle: a residual PPO v2 smoke policy on opticore_gazebo."""
    pytest.importorskip("torch")
    from amr_lite.config import deep_merge, load_config
    from amr_lite.deploy.export import export_bundle
    from amr_lite.rl.ppo import train

    root = tmp_path_factory.mktemp("bundle")
    config = deep_merge(load_config("ppo_v2"), {"network": {"hidden_sizes": [32, 32]},
                                                "vector_env": {"num_envs": 2, "backend": "sync"},
                                                "evaluation": {"interval_updates": 0}})
    result = train(root / "run", config, smoke=True, verbose=False)
    export_bundle(result["final_checkpoint"], root / "bundle", golden_count=16)
    return root / "bundle"
