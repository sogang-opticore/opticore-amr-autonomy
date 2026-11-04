from io import StringIO
import json

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from amr_lite.cli.train_m3 import CaptureRollouts


def test_terminal_row_keeps_sb3_bootstrap_observation(tmp_path):
    stream = StringIO()
    callback = CaptureRollouts(stream, tmp_path, checkpoint_every=1024)
    callback.num_timesteps = 2400
    callback.locals = {
        "dones": [True, False],
        "infos": [{"disposition": "timeout", "TimeLimit.truncated": True,
                   "terminal_observation": np.full(296, 0.25, dtype=np.float32)}, {}],
    }

    assert callback._on_step()
    row = json.loads(stream.getvalue())
    assert row["step"] == 2400 and row["worker"] == 0
    assert row["disposition"] == "timeout" and row["TimeLimit.truncated"]
    assert row["terminal_observation"] == [0.25] * 296
    assert callback.episodes == 1 and callback.invalid == 0


def test_invalid_transition_is_logged_and_aborts_rollout(tmp_path):
    stream = StringIO()
    callback = CaptureRollouts(stream, tmp_path, checkpoint_every=1024)
    callback.num_timesteps = 16
    callback.locals = {
        "dones": [True],
        "infos": [{"disposition": "invalid", "fault": "INVALID_ACTION",
                   "terminal_observation": np.zeros(296, dtype=np.float32)}],
    }

    with pytest.raises(RuntimeError, match="invalid training transition at step 16: INVALID_ACTION"):
        callback._on_step()
    assert json.loads(stream.getvalue())["disposition"] == "invalid"
    assert callback.episodes == 1 and callback.invalid == 1
