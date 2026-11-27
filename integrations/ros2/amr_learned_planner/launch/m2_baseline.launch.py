"""Launch one complete, freshly spawned M2 warehouse with a B0 or B1 controller."""
import math
from pathlib import Path
import re

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from amr_learned_planner.m2_bringup import bag_topics, robot_nodes
from amr_learned_planner.warehouse_launch import operational_nodes, warehouse_server


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    robot, planner, command = value("robot"), value("planner"), value("command")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", robot):
        raise ValueError("robot must be a single valid ROS namespace token")
    if planner == "b0" and command != "pass":
        raise ValueError(f"command:={command} requires planner:=b1, which owns the final command")
    m3 = command in {"shadow", "learned", "operational"}
    bundle, trace = value("m3_bundle"), value("m3_trace")
    if m3 and not (Path(bundle).is_absolute() and (Path(bundle) / "bundle.json").is_file()
                   and Path(trace).is_absolute() and not Path(trace).exists()):
        raise ValueError("M3 requires an absolute policy bundle directory and unused replay file")
    x, y = value("spawn_x"), value("spawn_y")
    if not all(math.isfinite(float(coord)) for coord in (x, y)):
        raise ValueError("spawn_x and spawn_y must be finite world coordinates")
    bag, csv = value("record_bag"), value("record_csv")
    if bag and (not Path(bag).is_absolute() or Path(bag).exists()):
        raise ValueError("record_bag must be an unused absolute directory path")
    if csv and (not Path(csv).is_absolute() or Path(csv).exists()):
        raise ValueError("record_csv must be an unused absolute file path")
    bringup = Path(get_package_share_directory("amr_bringup"))
    simulator = warehouse_server(bringup, [robot], value("models_dir"))
    nodes = robot_nodes(bringup / "launch" / "multi_robot.launch.py", robot, x, y, planner, tuning=value("tuning"))
    actions = [simulator, *nodes]
    actions.append(TimerAction(period=10.0, actions=[
        Node(package="amr_slam", executable="fleet_localization_init.py", name="fleet_localization_init",
             output="screen", parameters=[{"use_sim_time": True, "robots": [f"{robot}:{x}:{y}"],
                                           "map_origin_x": 3.0, "map_origin_y": 15.0}])
    ]))
    actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", name="m2_contact_bridge",
                        namespace=robot, output="screen",
                        arguments=[f"/{robot}/m2/contacts@std_msgs/msg/String[ignition.msgs.StringMsg"]))
    if command == "operational":
        actions.extend(operational_nodes(robot, bundle, trace))
    elif planner == "b1":
        mode = "dwa" if command == "pass" else command
        settings = {"use_sim_time": True, "mode": mode}
        if m3:
            settings.update(fallback_enabled=False, recover_after_sec=0.25, state_period=0.2)
            actions.append(Node(package="amr_learned_planner", executable="m3_policy", name="m3_policy",
                                namespace=robot, output="screen",
                                parameters=[{"use_sim_time": True, "bundle_path": bundle, "replay_path": trace}],
                                on_exit=Shutdown(reason="M3 policy exited")))
        actions.append(Node(package="amr_learned_planner", executable="cmd_arbiter", name="cmd_arbiter",
                            namespace=robot, output="screen", parameters=[settings],
                            on_exit=Shutdown(reason="M2 command arbiter exited")))
    if csv:
        actions.append(Node(package="amr_learned_planner", executable="nav_episode_recorder",
                            name="nav_episode_recorder", namespace=robot, output="screen",
                            parameters=[{"use_sim_time": True, "output_csv": csv, "label": command if m3 else planner,
                                         "goal_pose_topic": "goal_pose", "pose_topic": "odometry/filtered",
                                         "path_topic": "global_path", "scan_topic": "lidar", "status_topic": "dwa/status",
                                         "scenario_id": ParameterValue(value("scenario_id"), value_type=str),
                                         "seed": int(value("seed")),
                                         "run_id": ParameterValue(value("run_id"), value_type=str),
                                         "goal_tolerance": 0.4, "contact_topic": "m2/contacts",
                                         "timeout_sec": float(value("timeout_sec"))}]))
    if bag:
        topics = bag_topics(robot, planner)
        if m3:
            topics.extend(f"/{robot}/learned_planner/{name}" for name in ("cmd_vel", "status", "health"))
        if command == "operational":
            topics.append(f"/{robot}/cancel_goal")
        actions.append(ExecuteProcess(cmd=["ros2", "bag", "record", "-o", bag, *topics],
                                      output="screen"))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("planner", default_value="b0", choices=["b0", "b1"]),
        DeclareLaunchArgument("command", default_value="pass", choices=["pass", "stop", "shadow", "learned", "operational"]),
        DeclareLaunchArgument("m3_bundle", default_value=""),
        DeclareLaunchArgument("m3_trace", default_value=""),
        DeclareLaunchArgument("tuning", default_value="resolved", choices=["upstream", "resolved"]),
        DeclareLaunchArgument("robot", default_value="amr1"),
        DeclareLaunchArgument("spawn_x", default_value="3.0"),
        DeclareLaunchArgument("spawn_y", default_value="13.0"),
        DeclareLaunchArgument("record_csv", default_value=""),
        DeclareLaunchArgument("record_bag", default_value=""),
        DeclareLaunchArgument("models_dir", default_value=""),
        DeclareLaunchArgument("scenario_id", default_value="manual"),
        DeclareLaunchArgument("run_id", default_value=""),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("timeout_sec", default_value="120.0"),
    ]
    return LaunchDescription([*args, OpaqueFunction(function=start)])
