# Single-robot baseline launch

## Build on the simulator host

Use Ubuntu 22.04 with ROS 2 Humble, the upstream repository's `docs/environment-setup.md` runtime dependencies and the Fortress development libraries for C++17 (`ignition-gazebo6`, `ignition-plugin1` with `register`). The ROS tools needed for M2 include `colcon`, `ros_gz_bridge`, `ros2 bag`, TF2, NumPy and PyYAML. The upstream ROS workspace supplies the robot, localization and navigation packages. For two checkouts under `/workspace`, build and source them in order:

```bash
export OPTICORE_AMR=/workspace/opticore-amr
export OPTICORE_AUTONOMY=/workspace/opticore-amr-autonomy
source /opt/ros/humble/setup.bash
cd "$OPTICORE_AMR/ros2_ws"
colcon build --base-paths src
source install/setup.bash
python3 -m pip install --no-deps -e "$OPTICORE_AUTONOMY/packages/amr_lite"
python3 -c 'import rclpy, yaml, numpy, amr_lite'
mkdir -p /workspace/m2-overlay
cd /workspace/m2-overlay
colcon build --base-paths "$OPTICORE_AUTONOMY/integrations/ros2" --packages-up-to amr_learned_planner
source install/setup.bash
colcon test --base-paths "$OPTICORE_AUTONOMY/integrations/ros2" --packages-select amr_m2_contact_monitor --event-handlers console_direct+
colcon test-result --verbose
```

Use the Humble Python interpreter for all commands. The contact monitor calibration must pass before scoring trials. To run the static development mission, use a fresh output directory for each invocation:

```bash
python3 "$OPTICORE_AUTONOMY/experiments/m2/run_mission.py" train-static --planner b0 --out /tmp/m2-train-static-b0
python3 "$OPTICORE_AUTONOMY/experiments/m2/run_mission.py" train-static --planner b1 --out /tmp/m2-train-static-b1
```

The driver starts the launch and recorder itself and assigns the run ID, scenario, timeout, CSV and bag paths. It is the only goal writer. The following manual launch is for separate diagnostics.

## Manual verification

Use a fresh ROS domain, Gazebo partition and output directory per concurrent trial. Launch one instance per trial. The launch owns the simulator, robot spawn and localization. The source for `amr_bringup/models` must be available because the upstream installer omits model resources; use `models_dir:=/absolute/path/to/models` if the source is outside the build workspace. The two stock model directories are `pickup` and `standing_person`.

From a sourced shell (example output parent must already exist):

```sh
ros2 launch amr_learned_planner m2_baseline.launch.py planner:=b0 \
  robot:=amr1 spawn_x:=3.0 spawn_y:=13.0 \
  record_bag:=/tmp/m2-trial-001-b0-bag record_csv:=/tmp/m2-trial-001-b0.csv
```

Use `planner:=b1` with fresh output paths to forward the same baseline through the command arbiter. Use `planner:=b1 command:=stop` to hold a zero Twist while DWA publishes nonzero inputs; run that check separately from scored missions. For the B1 command-loss check stop the DWA process while the simulator clock advances. After its 0.25 s command timeout the arbiter publishes zero on the next 20 Hz tick. This check concerns commanded velocity; inspect ground truth to measure the robot's physical stopping distance.

1. Wait for active AMCL and map server and a stamped transform from `map` to `amr1/base_footprint`. Check `ros2 topic info -v /amr1/cmd_vel` and the effective status topic. The single command owner must be `/amr1/dwa_planner` (B0) or `/amr1/cmd_arbiter` (B1).
2. Inspect `ros2 param get /amr1/dwa_planner robot_radius`, `ros2 param get /amr1/astar_planner preferred_clearance` and `ros2 param get /amr1/amcl update_min_d`. The pinned YAML specifies 0.36, 0.50 and 0.05 respectively. The M2 launcher resolves bare YAML selectors for namespaced nodes in both configurations. Save the three complete parameter dumps for comparisons.
3. Inspect `/amr1/m2/contacts` and `/amr1/ground_truth`. The first is a periodic evaluation stream from Gazebo physics. Perform positive-contact and free-space calibration before accepting collision-free results. Confirm timestamps increase and the spawn pose maps to the expected pose.
4. For a manual trial, use one goal writer. Set `run_id:=...`, `scenario_id:=...`, `seed:=...`, `timeout_sec:=...` with `record_csv`. Only submit a goal when localization, DWA, A*, contact telemetry and the recorder are ready. A zero-return launch command or recorded bag alone does not establish mission success.
5. In B1 inspect raw DWA command/status, final command/status, and `cmd_arbiter/state`. In both configurations the bag contains clock, ground truth, contacts, command, status, goal, path, map, dynamic layer, scans, odometry and TF. Record shutdown cleanly and check topic message counts and reports.

Stock warehouse fixtures remain in the world. The stock periodic pedestrian mover and fleet goal writer do not run in this single-robot harness. Scenario inputs and the ground-truth monitor remain the same for B0 and B1.

**Measuring the namespaced parameter repair.** `planner:=b0 tuning:=upstream` calls the pinned upstream robot factory without adapting any node specifications. It passes the original bare-key YAML paths through ROS. The normal B0 and B1 launches use `tuning:=resolved` and explicitly bind the upstream YAML settings to each named node, followed by the original per-robot overrides. The shipped tuning values in the files are unchanged. Actual parameter handling by the ROS runtime and the effect on navigation are still unmeasured here. Run an upstream-mode diagnostic first, dump the live A*, DWA and AMCL parameters, and save the corresponding trajectory, contacts and status. Repeat with resolved B0 on the same development mission and record both versions. Attribute any difference between these runs to the tuning repair; the B0/B1 adapter comparison uses resolved tuning in both. Keep diagnostic files separate from scored mission outputs.
