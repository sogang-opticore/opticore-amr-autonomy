#!/usr/bin/env bash
# Build the pinned single-robot stack in a fresh workspace; calibrate before sending a goal.
set -eo pipefail

if [[ $# != 2 ]]; then
    echo "Usage: $0 OPTICORE_AMR_CHECKOUT NEW_RESULTS_DIR" >&2
    exit 2
fi
upstream=$(cd "$1" && pwd -P)
autonomy=$(cd "$(dirname "$0")/../.." && pwd -P)
mkdir -p "$(dirname "$2")"
mkdir "$2"
out=$(cd "$2" && pwd -P)
(
    phase=preflight
    trap 'result=$?; echo "$phase $result" > "$out/exit-status"; exit "$result"' EXIT
    echo "UTC start $(date -u +%FT%TZ)"
    if [[ $(uname -s) != Linux || ! -r /opt/ros/humble/setup.bash ]]; then
        echo "This pilot requires a Linux host with ROS 2 Humble and Gazebo Fortress." >&2
        exit 2
    fi
    # Container images may put a separate PyTorch Python ahead of the Jammy interpreter.
    export PATH="/usr/bin:$PATH"
    source /opt/ros/humble/setup.bash
    for executable in ros2 ign colcon cmake ctest git python3 stdbuf; do
        command -v "$executable" >/dev/null || { echo "Missing $executable" >&2; exit 2; }
    done
    export PYTHONPATH="$autonomy/packages/amr_lite${PYTHONPATH:+:$PYTHONPATH}"
    python3 -c 'import amr_lite, numpy, yaml, rclpy, xacro, tf2_ros; from scipy.ndimage import distance_transform_edt'
    for dependency in ros_gz_bridge ros_gz_sim robot_state_publisher robot_localization nav2_map_server nav2_amcl nav2_lifecycle_manager rosbag2_transport; do
        ros2 pkg prefix "$dependency" >/dev/null
    done
    models="$upstream/ros2_ws/src/amr_bringup/models"
    export IGN_GAZEBO_RESOURCE_PATH="$models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
    git -C "$autonomy" rev-parse HEAD > "$out/autonomy.sha"
    phase=build-underlay
    mkdir "$out/underlay" "$out/overlay"
    cd "$out/underlay"
    colcon build --base-paths "$upstream/ros2_ws/src" --packages-select amr_slam amr_navigation amr_bringup --cmake-args -DBUILD_TESTING=OFF
    source install/setup.bash
    phase=build-overlay
    cd "$out/overlay"
    colcon build --base-paths "$autonomy/integrations/ros2" --packages-up-to amr_learned_planner --cmake-args -DBUILD_TESTING=ON
    source install/setup.bash
    phase=contact-calibration
    suite="$out/overlay/build/amr_m2_contact_monitor"
    ctest --test-dir "$suite" --show-only=json-v1 > "$out/registered-contact-tests.json"
    python3 - "$out/registered-contact-tests.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    registered = {test["name"] for test in json.load(handle)["tests"]}
required = {"m2_physics_contact_samples", "m2_fortress_contact_calibration"}
if not required <= registered:
    raise SystemExit(f"Missing calibration tests: {sorted(required - registered)}")
PY
    ctest --test-dir "$suite" --output-on-failure --no-tests=error -R '^m2_(physics_contact_samples|fortress_contact_calibration)$' | tee "$out/contact-calibration.log"
    phase=pilot
    cd "$autonomy"
    python3 scripts/m2/run_pilot.py --out "$out/pilot"
    phase=complete
) 2>&1 | tee "$out/build-and-pilot.log"
