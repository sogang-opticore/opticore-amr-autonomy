"""Send a paired scenario set as goals and record it (ROADMAP M6).

    ros2 launch amr_learned_planner paired_runner.launch.py paired_set:=warehouse_v1 runs:=3 \
        label:=learned output_csv:=/workspace/results/gazebo_learned.csv
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    names = {"paired_set": "warehouse_v1", "runs": "1", "label": "ros", "output_csv": "paired_episodes.csv",
             "namespace": "", "use_sim_time": "true"}
    arguments = [DeclareLaunchArgument(name, default_value=value) for name, value in names.items()]
    runner = Node(
        package="amr_learned_planner", executable="goal_sequence_runner", name="goal_sequence_runner",
        namespace=LaunchConfiguration("namespace"), output="screen",
        parameters=[{"paired_set": LaunchConfiguration("paired_set"), "runs": LaunchConfiguration("runs"),
                     "label": LaunchConfiguration("label"), "output_csv": LaunchConfiguration("output_csv"),
                     "use_sim_time": LaunchConfiguration("use_sim_time")}])
    return LaunchDescription([*arguments, runner])
