# M2 warehouse geometry and configured dynamics

Upstream baseline is OPTICORE/opticore-amr @ `e8c9197ac2801c2770c83a7ee696b2b992d473b8`. The table pins inputs to a single-robot run in the upstream 60 × 40 m warehouse. No empirical delay, deceleration or real-time factor has been inferred from configuration.

| File under upstream ros2_ws/src | SHA-256 |
|---|---|
| amr_bringup/worlds/warehouse.world | 13cc2f4a829f1a54a4e4aa23db8bff2ac333407a411425d8435c60aaf520de9a |
| amr_bringup/maps/warehouse_map.pgm | 9c05eb3ca202e7d370830ef1abf9c86a8757a12e4606bb4b3096a289d5ddc7ea |
| amr_bringup/maps/warehouse_map.yaml | 6c8c41a2579f6c84284d5556d6ff1ff195cfd97e81a02f55b2ff67667682c363 |
| amr_bringup/urdf/amr_robot.urdf.xacro | c4d33d6b0c3ae4ae1437aa516016052679ba9d10ba28b431ec0d2696af362cfe |
| amr_navigation/config/dwa_params.yaml | 1592e249502ba842ce17bf7243a3ec4fb0b20b31c9ff9c06f7795ddaea1c1344 |

## Coordinates, masks and landmarks

The raster has 1207 × 840 cells at 0.05 m/cell. Both map files match the copies in `packages/amr_lite/assets/maps/opticore_warehouse` byte for byte. The ROS map origin of the lower-left cell is (−3.18, −16.9), yaw 0. The upstream fleet initializer uses a different offset: `map_xy = world_xy − (3, 15)`. Thus a Gazebo spawn at (5, 37) has a ROS map pose of (2, 22) and a lower-left-based Lite coordinate of (5.18, 38.9). PGM row zero is at the top. Direct ground truth odometry is in world, while the goal topic accepts map.

West/east north dock floor centers are world (5, 37) and (55, 37). Two 11 m long pinch passages are centered on world x = 14.5 and 49.5, y = 15. The vertical walls have centers 2.25 m apart and thickness 0.25 m, leaving 2.0 m wall-to-wall. Navigation coordinates may also name approach points; these are explicitly identified as diagnostic stations in the mission manifest.

The raster uses unknown gray 205 for parts of the exterior and shelf interiors. Nav2 map_server trinary thresholds 0.25/0.65 interpret 205 as free; the Lite loader conservatively masks it. Use the actual Gazebo collision engine for outcomes. Free endpoints on a raster alone do not demonstrate that a route or bypass can be driven.

## Robot, controls and sensor timing

| Component | Upstream configuration |
|---|---|
| Chassis | Box 0.60 × 0.40 × 0.20 m, mass 40 kg. Planner radius 0.36 m approximates the chassis half diagonal. Wheel outer lateral extent is 0.24 m, camera reaches x = 0.315 m. The model includes separate collision links. |
| Differential drive | Wheel radius 0.08 m, track 0.44 m; odom 50 Hz. DiffDrive maximum linear acceleration 2.0 m/s². |
| Local control | 20 Hz; forward limit 1.5 m/s, turn limit 1.5 rad/s, acceleration 1.0 m/s², angular acceleration 1.5 rad/s². DWA command-braking caps are 5.0 m/s² and 6.0 rad/s². |
| LiDAR | Forward offset 0.25 m, 720 samples over ±π at 10 Hz, 0.12–25 m range, configured Gaussian standard deviation 0.03 m. |
| Gazebo | Physics step 0.004 s; requested real-time factor 1.0. Measure achieved factor, sensor intervals, command→motion delay and stopping distance from timestamped runs. |

Retain both the requested commands and the executed trajectory. A velocity limit or physics setting is insufficient to estimate actual stopping distance on its own.

## Measured static-transfer timing

The four train-static bags from the local Fortress pilot at `661ebc815d3a` contain median stamp intervals of 100 ms for LiDAR, 20 ms for world ground truth and 4 ms for the simulation clock in every trial. The table uses simulation time after goal dispatch.

| Control | Repeat | Forward response at 0.05 m/s | Distance during final approach deceleration |
|---|---:|---:|---:|
| B0 | 0 | 88 ms | 0.709 m |
| B0 | 1 | 112 ms | Sustained stop window unavailable |
| B1 | 0 | 64 ms | 1.418 m |
| B1 | 1 | 128 ms | 0.425 m |

For forward response, timestamp the first final `cmd_vel.linear.x ≥ 0.05` m/s with the most recently recorded `/clock`. Subtract that time from the first world ground-truth sample at or above 0.05 m/s planar speed. This estimate includes message ordering and ground-truth sampling. For final deceleration, start with the first command below 1 m/s after the last command at or above 1 m/s. Sum world-pose displacements until ground-truth speed stays below 0.02 m/s for 0.25 s. These distances describe the normal controller's approach to the dock. Emergency braking would require a separately triggered stop.
