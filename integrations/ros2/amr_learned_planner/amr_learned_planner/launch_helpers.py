"""Node sets for opticore-amr: A*, DWA as fallback, learned planner and arbiter.

``make_learned_nav_nodes`` mirrors how ``amr_bringup/launch/multi_robot.launch.py``
wires A* and DWA, so a robot's navigation entry there can be replaced by
these nodes (see the package README). Parameters are passed as dicts so the
same YAML files work for namespaced robots.
"""
from __future__ import annotations

import os
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

PACKAGE = "amr_learned_planner"


def node_parameters(path: str, node_name: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    for key, value in document.items():
        if key.split("/")[-1] == node_name or key == "/**":
            return dict((value or {}).get("ros__parameters", {}))
    return {}


def make_learned_nav_nodes(*, mode: str = "shadow", bundle_path: str = "", namespace: str = "",
                           use_sim_time: bool = True, include_astar: bool = True, record: bool = False,
                           record_csv: str = "nav_episodes.csv", profile_check: str = "strict",
                           learned_params_file: str | None = None, arbiter_params_file: str | None = None,
                           recorder_params_file: str | None = None, astar_params_file: str | None = None,
                           dwa_params_file: str | None = None) -> list[Node]:
    share = get_package_share_directory(PACKAGE)
    navigation = get_package_share_directory("amr_navigation")
    learned_params_file = learned_params_file or os.path.join(share, "config", "learned_planner.yaml")
    arbiter_params_file = arbiter_params_file or os.path.join(share, "config", "cmd_arbiter.yaml")
    recorder_params_file = recorder_params_file or os.path.join(share, "config", "nav_episode_recorder.yaml")
    astar_params_file = astar_params_file or os.path.join(navigation, "config", "astar_params.yaml")
    dwa_params_file = dwa_params_file or os.path.join(navigation, "config", "dwa_params.yaml")
    ns = namespace.strip("/")
    prefix = f"{ns}/" if ns else ""
    topics = ({"odom_topic": "odometry/filtered", "odom_fallback_topic": "odom", "global_path_topic": "global_path",
               "goal_pose_topic": "goal_pose", "scan_topic": "lidar"} if ns else {})
    common = {"use_sim_time": use_sim_time}
    nodes: list[Node] = []
    if include_astar:
        astar_overrides = dict(common)
        remappings = []
        if ns:
            astar_overrides.update({"base_frame": f"{prefix}base_footprint", "dwa_status_topic": "dwa/status",
                                    "dynamic_layer_topic": "dynamic_obstacle_layer"})
            remappings = [("/map", f"/{ns}/map"), ("/goal_pose", f"/{ns}/goal_pose"),
                          ("/global_path", f"/{ns}/global_path")]
        nodes.append(Node(package="amr_navigation", executable="astar_planner", name="astar_planner",
                          namespace=ns, output="screen", emulate_tty=True,
                          parameters=[astar_params_file, astar_overrides], remappings=remappings))
    # The DWA keeps running as the fallback planner; only its outputs move.
    dwa_overrides: dict[str, Any] = {**common, "cmd_vel_topic": "dwa_fallback/cmd_vel"}
    dwa_remaps = [("/dwa/status", "dwa_fallback/status")]
    if ns:
        dwa_overrides.update({**topics, "cmd_vel_topic": "dwa_fallback/cmd_vel", "map_topic": "map",
                              "dynamic_layer_topic": "dynamic_obstacle_layer",
                              "local_frame": f"{prefix}odom_filtered", "global_frame": "map",
                              "robot_frame": f"{prefix}base_footprint"})
        dwa_remaps += [("/dwa/trajectories", f"/{ns}/dwa/trajectories"),
                       ("/dwa/best_trajectory", f"/{ns}/dwa/best_trajectory")]
    nodes.append(Node(package="amr_navigation", executable="dwa_node.py", name="dwa_planner", namespace=ns,
                      output="screen", emulate_tty=True, parameters=[dwa_params_file, dwa_overrides],
                      remappings=dwa_remaps))
    learned = {**node_parameters(learned_params_file, "learned_planner"), **common, **topics,
               "bundle_path": bundle_path, "profile_check": profile_check}
    if ns:
        learned["local_frame"] = f"{prefix}odom_filtered"
    nodes.append(Node(package=PACKAGE, executable="learned_planner", name="learned_planner", namespace=ns,
                      output="screen", emulate_tty=True, parameters=[learned]))
    arbiter = {**node_parameters(arbiter_params_file, "cmd_arbiter"), **common, "mode": mode}
    nodes.append(Node(package=PACKAGE, executable="cmd_arbiter", name="cmd_arbiter", namespace=ns,
                      output="screen", emulate_tty=True, parameters=[arbiter]))
    if record:
        recorder = {**node_parameters(recorder_params_file, "nav_episode_recorder"), **common,
                    "output_csv": record_csv, "label": f"{mode}:{os.path.basename(bundle_path.rstrip('/'))}"}
        if ns:
            recorder.update({"goal_pose_topic": "goal_pose", "pose_topic": "odometry/filtered",
                             "scan_topic": "lidar", "status_topic": "dwa/status"})
        nodes.append(Node(package=PACKAGE, executable="nav_episode_recorder", name="nav_episode_recorder",
                          namespace=ns, output="screen", parameters=[recorder]))
    return nodes
