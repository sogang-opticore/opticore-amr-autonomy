#!/usr/bin/env bash
# Build and calibrate once, then run serially in isolated ROS domains and Gazebo partitions.
set -eo pipefail
if [[ $# != 4 ]]; then
  echo "Usage: $0 UPSTREAM CANDIDATES NEW_RESULTS {all|paired|baseline|blockage|faults|pair|fleet|transport|sequence}" >&2
  exit 2
fi
repo=$(cd "$(dirname "$0")/../.." && pwd -P)
upstream=$(cd "$1" && pwd -P)
candidates=$(cd "$2" && pwd -P)
suite=$4
case "$suite" in all|paired|baseline|blockage|faults|pair|fleet|transport|sequence) ;; *) exit 2;; esac
mkdir -p "$(dirname "$3")"
mkdir "$3"
out=$(cd "$3" && pwd -P)
phase=preflight
trap 'rc=$?; printf "%s %s\n" "$phase" "$rc" > "$out/exit-status"' EXIT
if [[ -f "$upstream/UPSTREAM_COMMIT" ]]; then
  upstream_revision=$(cat "$upstream/UPSTREAM_COMMIT")
else
  upstream_revision=$(git -C "$upstream" rev-parse HEAD)
fi
test "$upstream_revision" = e8c9197ac2801c2770c83a7ee696b2b992d473b8
echo "$upstream_revision" > "$out/upstream.sha"
git -C "$repo" rev-parse HEAD > "$out/autonomy.sha"
date -u +%FT%TZ > "$out/started-at-utc"
export PATH="/usr/bin:$PATH"
source /opt/ros/humble/setup.bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export ROS_LOCALHOST_ONLY=1
export IGN_GAZEBO_RESOURCE_PATH="$upstream/ros2_ws/src/amr_bringup/models${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
python3 -c 'import rclpy, numpy, yaml, xacro, tf2_ros; from scipy.ndimage import distance_transform_edt'
phase=build-underlay
build=$(mktemp -d /tmp/opticore-m5-build-XXXXXXXX)
mkdir "$build/underlay" "$build/overlay"
cd "$build/underlay"
packages=(amr_slam amr_navigation amr_bringup)
if [[ $suite == all || $suite == fleet ]]; then
  test "$(sha256sum "$upstream/ros2_ws/src/amr_fleet/src/deadlock_manager.py" | cut -d' ' -f1)" = 62ad7e3142c8469a806e6ef9d17d6c855c5c9390e4cd600ede5f41a085893692
  packages+=(amr_fleet)
fi
colcon build --base-paths "$upstream/ros2_ws/src" --packages-select "${packages[@]}" --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
phase=build-overlay
cd "$build/overlay"
colcon build --base-paths "$repo/integrations/ros2" --packages-up-to amr_learned_planner --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
export PYTHONPATH="$repo/packages/amr_lite${PYTHONPATH:+:$PYTHONPATH}"
phase=contact-calibration
monitor="$build/overlay/build/amr_m2_contact_monitor"
ctest --test-dir "$monitor" --show-only=json-v1 > "$out/registered-contact-tests.json"
python3 - "$out/registered-contact-tests.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    names = {t["name"] for t in json.load(f)["tests"]}
if not {"m2_physics_contact_samples", "m2_fortress_contact_calibration"} <= names:
    raise SystemExit("Both physical contact tests must be registered.")
PY
ctest --test-dir "$monitor" --output-on-failure --no-tests=error -R '^m2_(physics_contact_samples|fortress_contact_calibration)$' | tee "$out/contact-calibration.log"
phase=trials
cd "$repo"
candidate_args=()
if [[ -n ${M5_SEED:-} ]]; then candidate_args+=(--seed "$M5_SEED"); fi
python3 experiments/m5/run_batch.py --candidates "$candidates" --out "$out/development" --suite "$suite" "${candidate_args[@]}"
phase=complete
