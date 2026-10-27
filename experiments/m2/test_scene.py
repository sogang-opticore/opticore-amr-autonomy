import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from scene import Actor, GazeboActor, world_xy


class SceneTest(unittest.TestCase):
    def test_timeline_and_service_calls(self):
        a = Actor((0.5, 0.5, 1), ((0, 1, 3), (1, 1, 3), (3, 3, 3)), 5)
        self.assertEqual(a.position(2), (2, 3))
        self.assertEqual(a.position(4), (3, 3))
        xml = ElementTree.fromstring(a.sdf())
        self.assertEqual(xml.findtext(".//model/pose"), "4.0 18.0 0.5 0 0 0")
        self.assertIsNotNone(xml.find(".//collision/geometry/box"))
        with patch("scene.ignition_service") as service:
            actor = GazeboActor(a)
            for t in (0, 1, 2, 5, 6):
                actor.update(t)
            self.assertEqual([c.args[0] for c in service.call_args_list], ["create", "set_pose", "remove"])

    def test_invalid_time_missing_zero_or_dimension(self):
        for knots in [((1, 2, 3),), ((0, 1, 2), (0, 3, 4))]:
            with self.assertRaises(ValueError):
                Actor((1, 1, 1), knots, None)
        with self.assertRaises(ValueError):
            Actor((float("nan"), 1, 1), ((0, 2, 3),), None)
        self.assertEqual(world_xy((2, 22)), (5, 37))
