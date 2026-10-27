"""Launch one complete, freshly spawned M2 warehouse with a B0 or B1 controller."""
import atexit
import math
import os
from pathlib import Path
import re
import tempfile

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from amr_learned_planner.m2_bringup import bag_topics, instrument_world, robot_nodes


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    robot, planner, command = value("robot"), value("planner"), value("command")
    if not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*", robot):
        raise ValueError("robot must be a single valid ROS namespace token")
    if planner == "b0" and command == "stop":
        raise ValueError("command:=stop requires planner:=b1, which owns the final command")
    x, y = value("spawn_x"), value("spawn_y")
    if not all(math.isfinite(float(coord)) for coord in (x, y)):
        raise ValueError("spawn_x and spawn_y must be finite world coordinates")
    bag, csv = value("record_bag"), value("record_csv")
    if bag and (not Path(bag).is_absolute() or Path(bag).exists()):
        raise ValueError("record_bag must be an unused absolute directory path")
    if csv and (not Path(csv).is_absolute() or Path(csv).exists()):
        raise ValueError("record_csv must be an unused absolute file path")
    bringup = Path(get_package_share_directory("amr_bringup"))
    plugin_dir = Path(get_package_prefix("amr_m2_contact_monitor")) / "lib"
    if not (plugin_dir / "libm2_contact_monitor.so").is_file():
        raise RuntimeError(f"M2 contact monitor is missing from {plugin_dir}")

    # Gazebo must reopen the SDF by name after launch evaluates this function.
    scratch = tempfile.TemporaryDirectory(prefix="opticore-m2-world-")
    atexit.register(scratch.cleanup)
    world = Path(scratch.name) / "warehouse.world"
    instrument_world(bringup / "worlds" / "warehouse.world", world, robot)
    # The pinned amr_bringup installs its worlds but omits its models directory.
    candidates = [Path(value("models_dir"))] if value("models_dir") else [
        bringup / "models", *[p / "src" / "amr_bringup" / "models" for p in bringup.parents],
        *[Path(p) for p in os.environ.get("IGN_GAZEBO_RESOURCE_PATH", "").split(os.pathsep) if p]]
    model_dir = next((d for d in candidates if all((d / name / "model.config").is_file()
                      for name in ("pickup", "standing_person"))), None)
    if model_dir is None:
        raise ValueError("set models_dir to the amr_bringup/models directory of the pinned checkout")
    environment = {}
    for key, directory in (("IGN_GAZEBO_RESOURCE_PATH", model_dir),
                           ("IGN_GAZEBO_SYSTEM_PLUGIN_PATH", plugin_dir)):
        environment[key] = os.pathsep.join(filter(None, [str(directory), os.getenv(key)]))
    # The stock world contains the fixtures. Only the per-mission driver moves its challenge actor.
    simulator = ExecuteProcess(cmd=["ign", "gazebo", "-s", "-r", str(world)], additional_env=environment,
                               output="screen", on_exit=Shutdown(reason="M2 Gazebo exited"))
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
    if planner == "b1":
        mode = "stop" if command == "stop" else "dwa"
        actions.append(Node(package="amr_learned_planner", executable="cmd_arbiter", name="cmd_arbiter",
                            namespace=robot, output="screen", parameters=[{"use_sim_time": True, "mode": mode}],
                            on_exit=Shutdown(reason="M2 command arbiter exited")))
    if csv:
        actions.append(Node(package="amr_learned_planner", executable="nav_episode_recorder",
                            name="nav_episode_recorder", namespace=robot, output="screen",
                            parameters=[{"use_sim_time": True, "output_csv": csv, "label": planner,
                                         "goal_pose_topic": "goal_pose", "pose_topic": "odometry/filtered",
                                         "path_topic": "global_path", "scan_topic": "lidar", "status_topic": "dwa/status",
                                         "scenario_id": ParameterValue(value("scenario_id"), value_type=str),
                                         "seed": int(value("seed")),
                                         "run_id": ParameterValue(value("run_id"), value_type=str),
                                         "goal_tolerance": 0.4, "contact_topic": "m2/contacts",
                                         "timeout_sec": float(value("timeout_sec"))}]))
    if bag:
        actions.append(ExecuteProcess(cmd=["ros2", "bag", "record", "-o", bag, *bag_topics(robot, planner)],
                                      output="screen"))
    return actions


def generate_launch_description():
    args = [
        DeclareLaunchArgument("planner", default_value="b0", choices=["b0", "b1"]),
        DeclareLaunchArgument("command", default_value="pass", choices=["pass", "stop"]),
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
