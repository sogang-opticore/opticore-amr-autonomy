import math
import unittest

from mission_spec import MAP, digest, load_cases, resolve


class ManifestTest(unittest.TestCase):
    def test_split_coverage_and_station_footprints(self):
        spec = load_cases()
        magic, size, maximum, pixels = MAP.read_bytes().split(b"\n", 3)
        self.assertEqual((magic, maximum, digest(MAP)), (b"P5", b"255", spec["map_sha256"]))
        cols, rows = map(int, size.split())
        for split in ("train", "selection", "test"):
            self.assertEqual({m["family"] for m in spec["missions"] if m["split"] == split},
                             {"static", "crossing", "head_on", "blockage"})
        for name, (x, y) in spec["stations"].items():
            # All cells touching a conservative 0.40 m clearance disk must be known free.
            i = math.floor((x + 3.18) / 0.05)
            j = rows - 1 - math.floor((y + 16.9) / 0.05)
            for dx in range(-9, 10):
                for dy in range(-9, 10):
                    if math.hypot(dx, dy) <= 9:
                        self.assertTrue(0 <= i+dx < cols and 0 <= j+dy < rows)
                        self.assertGreater(pixels[(j+dy)*cols+i+dx], 205, name)
        self.assertEqual(resolve("train-static")["spawn_world"], (5., 37.))
