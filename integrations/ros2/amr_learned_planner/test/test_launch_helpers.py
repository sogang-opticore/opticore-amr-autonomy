"""Wiring of make_learned_nav_nodes with stubbed launch_ros / ament_index (skipped under ROS)."""
import importlib
import sys
import types
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def helpers(tmp_path, monkeypatch):
    try:
        import launch_ros  # noqa: F401
        pytest.skip("real launch_ros available")
    except ImportError:
        pass
    navigation = tmp_path / "amr_navigation"
    (navigation / "config").mkdir(parents=True)
    for name in ("astar_params.yaml", "dwa_params.yaml"):
        (navigation / "config" / name).write_text("{}\n")
    shares = {"amr_learned_planner": PACKAGE_ROOT, "amr_navigation": navigation}

    class Node:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "launch_ros", types.ModuleType("launch_ros"))
    monkeypatch.setitem(sys.modules, "launch_ros.actions", types.SimpleNamespace(Node=Node))
    monkeypatch.setitem(sys.modules, "ament_index_python", types.ModuleType("ament_index_python"))
    monkeypatch.setitem(sys.modules, "ament_index_python.packages",
                        types.SimpleNamespace(get_package_share_directory=lambda name: str(shares[name])))
    sys.modules.pop("amr_learned_planner.launch_helpers", None)
    return importlib.import_module("amr_learned_planner.launch_helpers")


def _by_name(nodes):
    return {node.kwargs["name"]: node.kwargs for node in nodes}


def _merged(parameters):
    merged = {}
    for item in parameters:
        if isinstance(item, dict):
            merged.update(item)
    return merged


def test_namespaced_robot_wiring(helpers):
    nodes = _by_name(helpers.make_learned_nav_nodes(mode="learned", bundle_path="/bundles/cand", namespace="amr1",
                                                    record=True))
    assert set(nodes) == {"astar_planner", "dwa_planner", "learned_planner", "cmd_arbiter", "nav_episode_recorder"}
    assert all(node["namespace"] == "amr1" for node in nodes.values())
    astar = nodes["astar_planner"]
    assert _merged(astar["parameters"])["dwa_status_topic"] == "dwa/status"
    assert ("/global_path", "/amr1/global_path") in astar["remappings"]
    dwa = nodes["dwa_planner"]
    dwa_parameters = _merged(dwa["parameters"])
    assert dwa_parameters["cmd_vel_topic"] == "dwa_fallback/cmd_vel"
    assert dwa_parameters["local_frame"] == "amr1/odom_filtered"
    assert ("/dwa/status", "dwa_fallback/status") in dwa["remappings"]
    learned = _merged(nodes["learned_planner"]["parameters"])
    assert learned["bundle_path"] == "/bundles/cand" and learned["local_frame"] == "amr1/odom_filtered"
    assert learned["odom_topic"] == "odometry/filtered" and learned["scan_topic"] == "lidar"
    assert learned["lidar_mount_x"] == 0.25  # from config/learned_planner.yaml
    arbiter = _merged(nodes["cmd_arbiter"]["parameters"])
    assert arbiter["mode"] == "learned" and arbiter["fallback_hold_sec"] == 0.3
    recorder = _merged(nodes["nav_episode_recorder"]["parameters"])
    assert recorder["status_topic"] == "dwa/status" and recorder["label"] == "learned:cand"


def test_single_robot_defaults_keep_absolute_topics(helpers):
    nodes = _by_name(helpers.make_learned_nav_nodes(mode="dwa"))
    assert "nav_episode_recorder" not in nodes
    learned = _merged(nodes["learned_planner"]["parameters"])
    assert learned["odom_topic"] == "/odometry/filtered" and learned["local_frame"] == "odom_filtered"
    assert nodes["astar_planner"]["remappings"] == []
    assert _merged(nodes["cmd_arbiter"]["parameters"])["mode"] == "dwa"
