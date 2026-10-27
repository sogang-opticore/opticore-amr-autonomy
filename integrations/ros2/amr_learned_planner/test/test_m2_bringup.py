import json
from textwrap import dedent
import xml.etree.ElementTree as ET

import pytest

from amr_learned_planner.core.arbiter import ArbiterConfig, CommandArbiter
from amr_learned_planner.m2_bringup import bag_topics, instrument_world, robot_nodes


def test_adapter_ignores_nonzero_commands_while_stopped_and_forwards_on_pass():
    stop = CommandArbiter(ArbiterConfig(mode="stop"))
    for origin in ("dwa", "learned"):
        assert stop.on_cmd(origin, 1, 0.7, -0.3).cmd is None
        assert not stop.on_status(origin, 1, "NORMAL").statuses
    assert stop.tick(1).cmd == stop.tick(2).cmd == (0.0, 0.0)
    assert stop.active == "stop"
    json.dumps(stop.snapshot(1), allow_nan=False)
    forward = CommandArbiter(ArbiterConfig(mode="dwa"))
    assert forward.tick(0).cmd == (0.0, 0.0)
    assert forward.on_cmd("dwa", 1, 0.7, -0.3).cmd == (0.7, -0.3)
    assert forward.on_status("dwa", 1, "NORMAL").statuses == ["NORMAL"]
    assert forward.tick(1.3).cmd == (0.0, 0.0)
    assert forward.tick(0.5).cmd == (0.0, 0.0)


def test_world_instrumentation_preserves_authored_world(tmp_path):
    source, target = tmp_path / "stock.sdf", tmp_path / "measured.sdf"
    contents = '<sdf><world name="warehouse"><model name="ground_plane"/><include><uri>model://standing_person</uri></include></world></sdf>'
    source.write_text(contents)
    instrument_world(source, target, "amr2")
    assert source.read_text() == contents
    world = ET.parse(target).find("world")
    assert world.findtext("include/uri") == "model://standing_person"
    assert world.findtext("plugin/robot") == "amr2"
    assert world.findtext("plugin/floor_model") == "ground_plane"


@pytest.fixture()
def factory(tmp_path):
    factory = tmp_path / "multi_robot.launch.py"
    for name in ["amcl", "dwa_planner", "astar_planner"]:
        (tmp_path / f"{name}.yaml").write_text(f"{name}:\n  ros__parameters:\n    calibration: 7\n")
    factory.write_text(dedent("""
        from pathlib import Path
        def Node(**kwargs):
            return kwargs

        def make_robot_nodes(robot, x, y):
            result = [Node(name="bridge"), Node(name="spawn", position=[x, y])]
            for name in ["amcl", "dwa_planner", "astar_planner"]:
                params = [Path(__file__).with_name(name + ".yaml"), {"use_sim_time": True, "cmd_vel_topic": "cmd_vel"}]
                result.append(Node(name=name, namespace=robot, parameters=params,
                                   remappings=[("/dwa/status", f"/{robot}/dwa/status"), ("/foo","/bar")]))
            return result
        """))
    return factory


def test_graphs_differ_only_at_dwa_output_boundary(factory):
    stock = robot_nodes(factory, "amr1", "3.0", "13.0", "b0")
    wrapped = robot_nodes(factory, "amr1", "3.0", "13.0", "b1")
    assert len(stock) == len(wrapped) == 5
    for raw, mediated in zip(stock, wrapped):
        if raw["name"] != "dwa_planner":
            assert raw == mediated
            continue
        corrected = dict(mediated["parameters"][0], cmd_vel_topic="cmd_vel")
        assert corrected == raw["parameters"][0] == {"use_sim_time": True, "cmd_vel_topic": "cmd_vel", "calibration": 7}
        assert mediated["parameters"][0]["cmd_vel_topic"] == "dwa_fallback/cmd_vel"
        assert dict(mediated["remappings"]) == {"/dwa/status": "/amr1/dwa_fallback/status", "/foo": "/bar"}
    assert "/amr1/dwa_fallback/cmd_vel" in bag_topics("amr1", "b1")
    assert "/amr1/dwa_fallback/cmd_vel" not in bag_topics("amr1", "b0")
    assert "/amr2/m2/contacts" in bag_topics("amr2", "b1")


def test_before_repair_retains_original_yaml_argument(factory):
    nodes = robot_nodes(factory, "amr1", "3", "13", "b0", tuning="upstream")
    dwa = next(node for node in nodes if node["name"] == "dwa_planner")
    assert dwa["parameters"][0].name == "dwa_planner.yaml"
    assert dwa["parameters"][1]["cmd_vel_topic"] == "cmd_vel"
    with pytest.raises(ValueError, match="B0 diagnostics"):
        robot_nodes(factory, "amr1", "3", "13", "b1", tuning="upstream")
