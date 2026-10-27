import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from run_mission import ownership_error, verify_actor_position
from scene import Actor


class DriverTest(unittest.TestCase):
    def test_measured_timed_poses_and_removal(self):
        a = Actor((0.5, 0.5, 1.), ((0, 1, 0), (2, 3, 0), (4, 3, 2)), 6.)
        verify_actor_position(a, 2.2, [5.9, 15.0])
        verify_actor_position(a, 6.1, [6.0, 17.0])
        verify_actor_position(a, 6.6, None)
        for t, xy in [(3.0, None), (1.8, [4.,15.]), (2.2,[5.5,15.6]), (6.7, [6.,17.])]:
            with self.assertRaises(RuntimeError):
                verify_actor_position(a, t, xy)

    def test_competing_command_or_goal_owner(self):
        def node(name, ns):
            return SimpleNamespace(node_name=name, node_namespace=ns)
        pubs = {"/amr1/cmd_vel":[node("dwa_planner", "/amr1")],
                "/amr1/goal_pose":[node("m2_mission_driver", "/")]}
        n = SimpleNamespace(get_publishers_info_by_topic=pubs.get)
        self.assertIsNone(ownership_error(n, "b0"))
        self.assertIsNotNone(ownership_error(n, "b1"))
        pubs["/amr1/cmd_vel"] = [node("cmd_arbiter", "/amr1")]
        self.assertIsNone(ownership_error(n, "b1"))
        pubs["/amr1/goal_pose"].append(node("other", "/amr1"))
        self.assertIsNotNone(ownership_error(n, "b1"))

    def test_describe_and_no_test_outcomes(self):
        driver = str(Path(__file__).with_name("run_mission.py"))
        out = subprocess.check_output([sys.executable, driver, "train-blockage", "--describe"], text=True)
        self.assertEqual(json.loads(out)["spawn_world"], [49.5, 8.0])
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "still-closed"
            result = subprocess.run([sys.executable, driver, "test-blockage", "--out", str(target)],
                                    capture_output=True, text=True)
            self.assertIn("reserved", result.stderr)
            self.assertFalse(target.exists())
