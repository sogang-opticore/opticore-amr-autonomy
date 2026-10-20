# M2 single-robot ROS and fleet contract

The reference is opticore-amr commit `e8c9197ac2801c2770c83a7ee696b2b992d473b8`. Its `amr_bringup/launch/multi_robot.launch.py::make_robot_nodes` constructs a prefixed robot, sensor bridges, map server, AMCL, EKF, A* and the hybrid local controller. M2 uses one instance. Each run gets a fresh world and one goal writer. The fleet deadlock manager and the stock pedestrian mover stay off during isolated M2 missions.

| Interface (robot = amr1) | Producer → consumer | Payload and frame |
|---|---|---|
| `/amr1/goal_pose` | mission → A*, DWA, recorder | `PoseStamped` in `map`; the fleet manager can replace this goal during multi-robot operation |
| `/amr1/map`, `/amr1/global_path` | map server → A* and DWA; A* → DWA | `OccupancyGrid` and `Path` in `map` |
| `/amr1/lidar`, `/amr1/odometry/filtered`, `/amr1/odom` | Gazebo bridge / EKF → DWA | `LaserScan`, filtered `Odometry` and fallback odometry. Keep message stamps and TF frames |
| `/tf`, `/tf_static` | AMCL / EKF / state publisher → navigation | `map → amr1/odom_filtered → amr1/base_footprint`; resolve pose and goal in the same frame at the observation time |
| `/amr1/dynamic_obstacle_layer` | DWA → A* | `OccupancyGrid`. The original DWA still owns obstacle estimation and publication when its command output is routed via an adapter |
| `/amr1/cmd_vel` | B0 DWA or B1 arbiter → Gazebo bridge | `Twist`. Exactly one final publisher. Forward linear.x and angular.z in m/s and rad/s |
| `/amr1/dwa/status` | B0 DWA or B1 arbiter → A*, fleet, recorder | `String`. DWA can repeat `REACHED` after its initial `GOAL_REACHED`; status contains no goal ID |
| `/amr1/dwa_fallback/{cmd_vel,status}` | B1 DWA → B1 arbiter | Private `Twist` / `String`. Other consumers continue to see the effective outputs above |
| `/amr1/cmd_arbiter/state` | B1 arbiter → telemetry | JSON in a `String`, including mode, active source and freshness. B0 has no arbiter |
| `/amr1/ground_truth`, `/amr1/m2/contacts` | simulator → evaluation | World-frame `Odometry`, periodic stamped `String` contact reports. Goal alignment uses the map/world transform. No ground-truth input to the controller |
| `/clock` | Gazebo bridge → ROS nodes | Simulation clock. Keep wall time separately for throughput |

The upstream fleet manager consumes effective status, goal, filtered odometry, TF and `/fleet/priorities`. It publishes HOLD, retreat and resume by replacing `/amrN/goal_pose`. It publishes diagnostics on `/fleet/deadlock_status`. Fleet actions never require a second publisher of the wheel command. Two-robot execution and coordination are M5 checks.

**B0/B1 comparison.** B0 is the upstream A* + Pure Pursuit/adaptive-speed DWA node with its normal final command and status. B1 runs the same controller, sensors, obstacle layer, global planner and localization. The adapter alone owns final command/status; B1 adds forwarding latency and stops on a stale command stream. Both configurations explicitly apply the same upstream tuning in the namespace. Record loaded parameters, versions, TF, world, initial pose, command rate and contact history for both before comparing them. B1 is an adapter configuration at M2; obstacle estimation still resides in the DWA process.

**Frame convention.** With the pinned localization initializer, world spawn (3,13) becomes map (0,−2) at zero yaw. The map-frame goal and the world-frame ground-truth pose differ by (3,15) metres. The map image's YAML origin locates the grid inside the map; it does not convert world poses to map poses.
