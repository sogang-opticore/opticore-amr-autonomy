# M2-L1 · Local policy input and action contract

`packages/amr_lite/configs/m2/policy_io.yaml` defines the starting interface for M3. Both Lite and ROS must reproduce the same observation and proposed target from the same stamped inputs. A replacement observation design needs a new identifier and new replay evidence. M1 bundles keep their original interpretation.

## Inputs and clock

| Input | Source and unit | Processing |
|---|---|---|
| LaserScan | Robot's `lidar` topic, simulated acquisition time in `header.stamp` | The Gazebo xacro specifies 720 rays across 360°, 10 Hz, 0.12–25 m, scanner 0.25 m ahead of the base. Use the angle increment and valid range carried by the message. |
| Motion | Robot's `odometry/filtered`, linear x in m/s and angular z in rad/s | Resolve the active robot's base and local odometry frames. A fallback odometry stream requires a valid transform into the same frame. |
| Route | Active goal and `global_path` in map coordinates | Transform the active path into the local pose frame. Compute the nearest path projection, then sample 0.65 m ahead along the polyline, capped at its endpoint. Stop on an empty path or unavailable transform. |
| Attribution | Mission, goal, route revision and robot namespace | Keep these alongside replay inputs and commands. Supply only the tensor below to the policy. |

Resample valid hits in the base frame using the common `resample_scan` function. The nearest assigned hit fills each angular bin, clipped to 8 m. Each scan contributes 72 bins from right to left around the forward axis, over 270°. Positive infinity denotes a valid no-return beam. Drop individual NaN, negative infinity and below-minimum readings. An empty scan, invalid angular metadata or a scan with no usable beam and no legitimate no-return beam must make the input unavailable. This check precedes bin filling.

Use a deque of four *distinct sensor acquisitions*, oldest first. Duplicates and out-of-order messages do not advance time or refresh freshness. At 20 Hz control and 10 Hz LiDAR, the intermediate tick reuses the current stack. For stamps 0.0, 0.1, 0.2 and 0.3 s, preserve those four at 0.35 s. A scan at 0.4 s yields [0.1, 0.2, 0.3, 0.4]. Use simulation time and bound both odometry and scan age by the configured 0.5 s development timeout. A future stamp or backward clock jump invalidates timing. Clear history on simulation reset, namespace/mission change or loss of freshness. Keep the physical command at zero until input validity and four fresh frames have been restored. The 0.5 s threshold still needs review against measured sensor and braking latency.

## Tensor and action

Concatenate 288 range values divided by 8 and the following eight float32 values in order, using the existing `build_observation` and `path_state` primitives.

| Indices | Features |
|---|---|
| 288–290 | Distance to the local lookahead target divided by 8 m, clipped to [0,1], then sine and cosine of its bearing in the base frame |
| 291–292 | Measured forward speed divided by 1.5 m/s, clipped to [0,1], and angular speed divided by 1.5 rad/s, clipped to [-1,1] |
| 293–295 | Signed cross-track error divided by 2 m, clipped to [-1,1] (left of path is positive), then sine and cosine of path heading minus robot heading |

The direct policy proposes two normalized scalars. After clipping, use `v_target=(a_v+1)*1.5/2` m/s and `w_target=a_w*1.5` rad/s. The domain supports forward motion, turn-in-place and stop. Reverse recovery by other controllers is outside the policy's input/command domain. Re-arm learned control from standstill after recovery. A [0,0] action requests 0.75 m/s. A fault or deadline miss bypasses the policy and emits a physical zero Twist through the common command authority. Record the proposed target, limited command and measured motion separately. Simulation obstacle identities, contact signals and ground-truth pose are available to reward/evaluation only.

## Existing assets and M3 acceptance

The versioned source profile comes from `opticore-amr/ros2_ws/src/amr_bringup/urdf/amr_robot.urdf.xacro` and `amr_navigation/config/dwa_params.yaml`. The Lite `opticore_gazebo` profile matches the nominal scan and speed limits. Actual rates and maximum achievable deceleration require a simulator trace. The DWA command brake parameter (5 m/s²), the simulated DiffDrive maximum linear acceleration (2 m/s²) and measured stopping distance describe different quantities.

Existing spike code supplies resampling, tensor math and action scaling. Lite appends held scans on control ticks. The ROS spike timestamps scan callbacks at receipt and appends frames during control. Both can fill the initial stack with one scan. The Lite invalid-action branch replaces a malformed action with normalized zeros before advancing dynamics. These paths require changes in M3 before granting learned driving authority. Regress reset/warmup, a duplicate at a later receipt time, shuffled stamps, a physically changing obstacle, stale input, invalid action and path replacement on the same replay. Retain the four source stamps, goal/path revisions, tensor and physical command for each tick.

Run the M2 primitive/profile checks with `python -m pytest packages/amr_lite/tests/test_m2_policy_io.py` from the repository root.
