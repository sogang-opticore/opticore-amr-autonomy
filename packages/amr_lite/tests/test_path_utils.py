import math

import numpy as np
import pytest

from amr_lite.envs.dynamics import RobotState
from amr_lite.envs.geometry import point_segment_projection
from amr_lite.planning.path_utils import nearest_segment, path_state


def _brute_force(path, x, y, first=0, last=None):
    last = len(path) - 2 if last is None else last
    best = min(range(first, last + 1), key=lambda i: math.hypot(
        x - point_segment_projection(x, y, *path[i], *path[i + 1])[0],
        y - point_segment_projection(x, y, *path[i], *path[i + 1])[1]))
    return best


def test_vectorized_search_on_dense_paths_matches_brute_force():
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 4.0 * math.pi, 200)
    path = [(float(5 + 3 * math.cos(v)), float(4 + 2 * math.sin(v) + 0.1 * v)) for v in t]
    for _ in range(100):
        x, y = float(rng.uniform(1, 9)), float(rng.uniform(1, 8))
        index, qx, qy, segment_t = nearest_segment(path, x, y)
        assert index == _brute_force(path, x, y)
        assert (qx, qy, segment_t) == point_segment_projection(x, y, *path[index], *path[index + 1])


def test_segment_range_limits_the_search_window():
    # A hairpin: the robot is closest to the return leg but tracking the outbound leg.
    path = [(0.0, 0.0), (4.0, 0.0), (4.0, 0.4), (0.0, 0.4)]
    robot = RobotState(2.0, 0.3, 0.0)
    assert path_state(path, robot, 0.65).segment_index == 2
    assert path_state(path, robot, 0.65, segment_range=(0, 0)).segment_index == 0
    assert path_state(path, robot, 0.65, segment_range=(0, 0)).cross_track_error == pytest.approx(0.3)
