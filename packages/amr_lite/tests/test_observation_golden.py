"""Lite must keep producing the stored observation vectors (ARCHITECTURE §12).

The ROS integration test suite checks the same file from the ROS side. If
this fails because the observation pipeline changed on purpose, bump
OBSERVATION_CONTRACT_VERSION and regenerate with tests/golden/make_observation_golden.py.
"""
from pathlib import Path

import numpy as np

from amr_lite.learning.observation import OBSERVATION_CONTRACT_VERSION

GOLDEN = Path(__file__).with_name("golden") / "observation_v1.npz"


def test_lite_reproduces_golden_observations():
    from tests.golden.make_observation_golden import generate

    stored = np.load(GOLDEN)
    assert str(stored["contract_version"]) == OBSERVATION_CONTRACT_VERSION
    fresh = generate()
    assert np.max(np.abs(fresh["observation"] - stored["observation"])) <= 1e-5
    assert np.array_equal(fresh["refresh"], stored["refresh"])
    assert np.allclose(fresh["path"], stored["path"])
