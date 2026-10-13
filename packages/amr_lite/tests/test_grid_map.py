import hashlib
import math

import numpy as np
import pytest

from amr_lite.config import ROOT
from amr_lite.envs.geometry import Rectangle
from amr_lite.envs.grid_map import GridMap, feature_transform, load_ros_map, named_map, read_pgm
from amr_lite.envs.map import WarehouseMap
from amr_lite.envs.raycast import cast_rays


def _rasterize(rectangles, width=12.0, height=8.0, resolution=0.05) -> np.ndarray:
    rows, cols = int(round(height / resolution)), int(round(width / resolution))
    ys, xs = (np.arange(rows) + 0.5) * resolution, (np.arange(cols) + 0.5) * resolution
    grid = np.zeros((rows, cols), dtype=bool)
    for rect in rectangles:
        grid |= ((xs[None, :] >= rect.x_min) & (xs[None, :] <= rect.x_max)
                 & (ys[:, None] >= rect.y_min) & (ys[:, None] <= rect.y_max))
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    return grid


SCENE = [Rectangle(3.0, 2.0, 4.0, 5.0), Rectangle(6.0, 0.0, 6.5, 3.5), Rectangle(8.0, 5.0, 10.0, 5.6)]


def test_feature_transform_is_exact_below_saturation():
    occupied = np.random.default_rng(0).uniform(size=(40, 50)) < 0.03
    distance, rows, cols = feature_transform(occupied, 12)
    obstacle = np.argwhere(occupied)
    for row, col in np.ndindex(occupied.shape):
        exact = np.min(np.hypot(obstacle[:, 0] - row, obstacle[:, 1] - col))
        if exact < 12:
            assert distance[row, col] == pytest.approx(exact)
            assert math.hypot(rows[row, col] - row, cols[row, col] - col) == pytest.approx(exact)
        else:
            assert distance[row, col] == 12 and rows[row, col] == -1


def test_clearance_matches_rectangle_map_within_half_a_cell():
    world, grid = WarehouseMap(obstacles=list(SCENE)), GridMap(_rasterize(SCENE), 0.05)
    rng = np.random.default_rng(1)
    for _ in range(500):
        x, y = float(rng.uniform(0.3, 11.7)), float(rng.uniform(0.3, 7.7))
        exact = world.clearance(x, y)
        if exact > 0.02:
            assert exact - 0.051 <= grid.clearance(x, y) <= exact + 1e-9
    assert grid.collides(-0.1, 4.0, 0.0) and grid.collides(3.5, 3.0, 0.0)
    assert grid.occupied(3.5, 3.0) and not grid.occupied(1.5, 4.0, 0.4)


def test_grid_ray_cast_agrees_with_rectangle_ray_cast():
    world, grid = WarehouseMap(obstacles=list(SCENE)), GridMap(_rasterize(SCENE), 0.05)
    rng = np.random.default_rng(2)
    differences = []
    for _ in range(150):
        x, y, theta = float(rng.uniform(0.5, 11.5)), float(rng.uniform(0.5, 7.5)), float(rng.uniform(-3, 3))
        if world.clearance(x, y) < 0.1:
            continue
        angles = [theta + float(v) for v in np.linspace(-2.356, 2.356, 72)]
        differences.append(cast_rays(x, y, angles, grid, [], 8.0) - cast_rays(x, y, angles, world, [], 8.0))
    differences = np.concatenate(differences)
    assert np.median(np.abs(differences)) < 0.06
    assert np.mean(differences > 0.1) < 0.002  # rays must not leak through walls


def test_read_pgm_handles_comments_and_ascii(tmp_path):
    binary = tmp_path / "b.pgm"
    binary.write_bytes(b"P5\n# comment\n3 2\n255\n" + bytes([0, 205, 254, 254, 0, 205]))
    ascii_map = tmp_path / "a.pgm"
    ascii_map.write_text("P2\n3 2\n255\n0 205 254\n254 0 205\n")
    expected = np.array([[0, 205, 254], [254, 0, 205]], dtype=np.uint8)
    assert np.array_equal(read_pgm(binary), expected)
    assert np.array_equal(read_pgm(ascii_map), expected)


def test_ros_map_semantics_flip_rows_and_mark_unknown(tmp_path):
    (tmp_path / "m.pgm").write_bytes(b"P5\n3 2\n255\n" + bytes([0, 205, 254, 254, 254, 205]))
    (tmp_path / "m.yaml").write_text("image: m.pgm\nresolution: 0.5\norigin: [-1.0, 2.0, 0]\n"
                                     "negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    strict = load_ros_map(tmp_path / "m.yaml")
    ros_like = load_ros_map(tmp_path / "m.yaml", unknown_values=())
    # Image row 0 is the top of the map, so it becomes the last grid row.
    assert strict.occupied_cells.tolist() == [[False, False, True], [True, True, False]]
    assert ros_like.occupied_cells.tolist() == [[False, False, False], [True, False, False]]
    assert strict.to_map(0.0, 0.0) == (-1.0, 2.0) and strict.to_lite(-1.0, 2.0) == (0.0, 0.0)


def test_opticore_warehouse_asset_is_pinned():
    directory = ROOT / "assets" / "maps" / "opticore_warehouse"
    digest = hashlib.sha256((directory / "warehouse_map.pgm").read_bytes()).hexdigest()
    assert digest == "9c05eb3ca202e7d370830ef1abf9c86a8757a12e4606bb4b3096a289d5ddc7ea"
    warehouse = named_map("opticore_warehouse")
    assert (warehouse.cols, warehouse.rows) == (1207, 840)
    assert warehouse.map_origin == (-3.18, -16.9)
    # Shelf interiors (unknown 205 pixels) are solid; the aisle between rows is free.
    assert warehouse.collides(*warehouse.to_lite(8.0, 0.0), 0.0)
    assert warehouse.clearance(*warehouse.to_lite(20.0, 3.0), 0.36) > 0.5
