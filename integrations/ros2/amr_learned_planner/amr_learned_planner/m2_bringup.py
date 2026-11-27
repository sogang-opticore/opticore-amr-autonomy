"""Construct the pinned OPTICORE robot with either direct or mediated DWA outputs."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


def instrument_world(source: Path, target: Path, robot: str, *others: str) -> None:
    tree = ET.parse(source)
    world = tree.find("world")
    if world is None or world.find("model[@name='ground_plane']") is None:
        raise ValueError("expected the OPTICORE warehouse world and ground_plane")
    for name in (robot, *others):
        plugin = ET.SubElement(world, "plugin", filename="libm2_contact_monitor.so", name="opticore::M2ContactMonitor")
        ET.SubElement(plugin, "robot").text = name
        ET.SubElement(plugin, "floor_model").text = "ground_plane"
    tree.write(target, encoding="utf-8", xml_declaration=True)


def robot_nodes(factory_path: Path, robot: str, spawn_x: str, spawn_y: str, planner: str,
                *, tuning: str = "resolved", bridge_clock: bool = True):
    """Reuse the upstream per-robot factory, including timed actions and TF wiring.

    The upstream API has no command-remapping argument and builds navigation and
    hardware in one function. Adapt the three node specs in a private module
    instance, before launch_ros resolves names. Never change installed sources.
    """
    if planner not in {"b0", "b1"}:
        raise ValueError(f"unknown M2 planner {planner!r}")
    if tuning not in {"resolved", "upstream"} or (tuning == "upstream" and planner != "b0"):
        raise ValueError("upstream tuning is only available for direct B0 diagnostics")
    spec = importlib.util.spec_from_file_location("opticore_m2_upstream_bringup", factory_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuning == "upstream" and bridge_clock:
        return module.make_robot_nodes(robot, spawn_x, spawn_y)
    upstream_node = module.Node
    expected = {"astar_planner", "dwa_planner", "amcl"} if tuning == "resolved" else set()
    suppressed_bridges = []
    seen = set()

    def configure_node(**kwargs):
        name = kwargs.get("name")
        if name == "ros_gz_bridge" and not bridge_clock:
            clock = "/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock"
            if kwargs.get("namespace") != robot or kwargs["arguments"].count(clock) != 1:
                raise ValueError("upstream clock bridge contract changed")
            kwargs["arguments"] = [arg for arg in kwargs["arguments"] if arg != clock]
            suppressed_bridges.append(name)
        if name in expected:
            if name in seen or kwargs.get("namespace") != robot:
                raise ValueError(f"upstream node contract changed for {name}")
            seen.add(name)
            config_file, overrides = kwargs["parameters"]
            with open(config_file, encoding="utf-8") as handle:
                # Stock files have a bare, unnamespaced node key.
                configured = yaml.safe_load(handle)[name]["ros__parameters"]
            params = {**configured, **overrides}
            if name == "dwa_planner" and planner == "b1":
                if params.get("cmd_vel_topic") != "cmd_vel":
                    raise ValueError("upstream DWA command endpoint changed")
                remaps = dict(kwargs["remappings"])
                if remaps.get("/dwa/status") != f"/{robot}/dwa/status":
                    raise ValueError("upstream DWA status endpoint changed")
                params["cmd_vel_topic"] = "dwa_fallback/cmd_vel"
                remaps["/dwa/status"] = f"/{robot}/dwa_fallback/status"
                kwargs["remappings"] = list(remaps.items())
            kwargs["parameters"] = [params]
        return upstream_node(**kwargs)

    module.Node = configure_node
    nodes = module.make_robot_nodes(robot, spawn_x, spawn_y)
    if not bridge_clock and len(suppressed_bridges) != 1:
        raise ValueError("expected exactly one upstream bridge to suppress the duplicate clock")
    if seen != expected:
        raise ValueError(f"missing expected upstream nodes: {sorted(expected - seen)}")
    return nodes


def bag_topics(robot: str, planner: str) -> list[str]:
    channels = ["map", "goal_pose", "global_path", "odometry/filtered", "odom", "lidar", "ground_truth",
                "cmd_vel", "dwa/status", "dynamic_obstacle_layer", "m2/contacts"]
    if planner == "b1":
        channels.extend(["dwa_fallback/cmd_vel", "dwa_fallback/status", "cmd_arbiter/state"])
    return ["/clock", "/tf", "/tf_static", *(f"/{robot}/{topic}" for topic in channels)]
