"""Headless Fortress world and isolated operational controllers for warehouse launches."""
import atexit
import os
from pathlib import Path
import tempfile

from ament_index_python.packages import get_package_prefix
from launch.actions import ExecuteProcess, Shutdown
from launch_ros.actions import Node

from amr_learned_planner.m2_bringup import instrument_world


def warehouse_server(bringup, robots, models_dir):
    plugin_dir = Path(get_package_prefix("amr_m2_contact_monitor")) / "lib"
    if not (plugin_dir / "libm2_contact_monitor.so").is_file():
        raise RuntimeError(f"M2 contact monitor is missing from {plugin_dir}")

    # Gazebo must reopen the SDF by name after launch evaluates this function.
    scratch = tempfile.TemporaryDirectory(prefix="opticore-m2-world-")
    atexit.register(scratch.cleanup)
    world = Path(scratch.name) / "warehouse.world"
    instrument_world(bringup / "worlds" / "warehouse.world", world, *robots)
    # The pinned amr_bringup installs its worlds but omits its models directory.
    candidates = [Path(models_dir)] if models_dir else [
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
    return simulator


def operational_nodes(robot, bundle, trace):
    return [
        Node(package="amr_learned_planner", executable="m3_policy", name="m3_policy", namespace=robot,
             output="screen", parameters=[{"use_sim_time": True, "operational": True,
                                            "bundle_path": str(bundle), "replay_path": str(trace)}],
             on_exit=Shutdown(reason=f"{robot} operational policy exited")),
        Node(package="amr_learned_planner", executable="operational_arbiter", name="cmd_arbiter",
             namespace=robot, output="screen", parameters=[{"use_sim_time": True}],
             on_exit=Shutdown(reason=f"{robot} operational arbiter exited")),
    ]
