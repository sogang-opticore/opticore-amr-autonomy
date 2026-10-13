"""A* + DWA (fallback) + learned planner + command arbiter for opticore-amr.

Start the simulation and localization first (as for astar_dwa.launch.py), then:

    ros2 launch amr_learned_planner learned_nav.launch.py mode:=dwa
    ros2 launch amr_learned_planner learned_nav.launch.py mode:=shadow  bundle:=/path/to/bundle
    ros2 launch amr_learned_planner learned_nav.launch.py mode:=learned bundle:=/path/to/bundle record:=true

mode:=dwa is the rollback: the arbiter forwards the DWA unchanged.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from amr_learned_planner.launch_helpers import make_learned_nav_nodes


def _nodes(context, *args, **kwargs):
    value = lambda name: LaunchConfiguration(name).perform(context)
    optional = lambda name: value(name) or None
    return make_learned_nav_nodes(
        mode=value("mode"), bundle_path=value("bundle"), namespace=value("namespace"),
        use_sim_time=value("use_sim_time").lower() == "true",
        include_astar=value("include_astar").lower() == "true",
        record=value("record").lower() == "true", record_csv=value("record_csv"),
        profile_check=value("profile_check"),
        learned_params_file=optional("learned_params_file"), arbiter_params_file=optional("arbiter_params_file"),
        astar_params_file=optional("astar_params_file"), dwa_params_file=optional("dwa_params_file"))


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("mode", default_value="shadow", choices=["dwa", "shadow", "learned"]),
        DeclareLaunchArgument("bundle", default_value="", description="policy bundle directory"),
        DeclareLaunchArgument("namespace", default_value="", description="robot namespace, e.g. amr1"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("include_astar", default_value="true"),
        DeclareLaunchArgument("record", default_value="false"),
        DeclareLaunchArgument("record_csv", default_value="nav_episodes.csv"),
        DeclareLaunchArgument("profile_check", default_value="strict", choices=["strict", "warn"]),
        DeclareLaunchArgument("learned_params_file", default_value=""),
        DeclareLaunchArgument("arbiter_params_file", default_value=""),
        DeclareLaunchArgument("astar_params_file", default_value=""),
        DeclareLaunchArgument("dwa_params_file", default_value=""),
    ]
    return LaunchDescription([*arguments, OpaqueFunction(function=_nodes)])
