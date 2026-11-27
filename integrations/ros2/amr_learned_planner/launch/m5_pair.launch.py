"""One instrumented warehouse, two operational AMRs and one simulation clock."""
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from amr_learned_planner.m2_bringup import robot_nodes
from amr_learned_planner.warehouse_launch import operational_nodes, warehouse_server


def start(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    manager = value("fleet_manager")
    if manager not in {"true", "false"}:
        raise ValueError("fleet_manager must be true or false")
    output, bundle = Path(value("output")), Path(value("bundle"))
    if not (output.is_absolute() and bundle.is_absolute() and (bundle / "bundle.json").is_file()):
        raise ValueError("expected absolute output directory and a policy bundle")
    output.mkdir(parents=True, exist_ok=False)
    bringup = Path(get_package_share_directory("amr_bringup"))
    # World coordinates use map origin (3,15). Both goals in the isolation probe
    # belong to the development warehouse, on separate parallel aisles.
    if manager == "true":
        spawns = {"amr1": ("6.6", "37.0"), "amr2": ("5.0", "37.0")}
    else:
        spawns = {"amr1": ("5.0", "37.0"), "amr2": ("5.0", "31.0")}
    robots = [f"{r}:{x}:{y}" for r, (x, y) in spawns.items()]
    actions = [warehouse_server(bringup, spawns, value("models_dir"))]
    for robot, (x, y) in spawns.items():
        actions.extend(robot_nodes(bringup / "launch/multi_robot.launch.py", robot, x, y, "b1",
                                   bridge_clock=robot == "amr1"))
        actions.extend(operational_nodes(robot, bundle, output / f"{robot}.jsonl"))
        actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", name="m2_contact_bridge",
                            namespace=robot, output="screen",
                            arguments=[f"/{robot}/m2/contacts@std_msgs/msg/String[ignition.msgs.StringMsg"]))
    actions.append(TimerAction(period=10.0, actions=[
        Node(package="amr_slam", executable="fleet_localization_init.py", name="fleet_localization_init",
             output="screen", parameters=[{"use_sim_time": True, "map_origin_x": 3.0, "map_origin_y": 15.0,
                                           "robots": robots}])]))
    if manager == "true":
        actions.append(Node(package="amr_fleet", executable="deadlock_manager.py", name="deadlock_manager",
                            output="screen", parameters=[{"use_sim_time": True, "robots": robots,
                                                         "x_hold": 20.0}],
                            on_exit=Shutdown(reason="upstream fleet manager exited")))
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("bundle"), DeclareLaunchArgument("output"),
                              DeclareLaunchArgument("models_dir", default_value=""),
                              DeclareLaunchArgument("fleet_manager", default_value="false"),
                              OpaqueFunction(function=start)])
