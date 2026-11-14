# Gazebo and Lite timing

The four static-train Fortress bags from M2 provide the nominal sensor profile and a measured response envelope. All bags use the same warehouse, simulated AMR and upstream drive controller. [Measurement and source digests](../../experiments/m3/fortress-motion.json)

| Quantity | Gazebo | M3 Lite |
|---|---|---|
| Scan acquisitions | 720 beams every 100 ms | 720 beams every 100 ms; 20 Hz control uses held scans between acquisitions |
| Ground truth and simulation clock | Pose every 20 ms, clock every 4 ms | State integrated every 50 ms; motion is sampled at the control tick |
| Stationary LiDAR noise | 0.02972–0.03015 m estimated standard deviation | Gaussian 0.03 m nominal |
| First forward response | 64–128 ms from the first final command ≥0.05 m/s to first world speed ≥0.05 m/s | Zero added transport ticks nominal; sensitivity adds one or two 50 ms ticks |
| Speed increase | 0.221–0.233 m/s² average between first samples at 0.05 and 1.0 m/s | Shared command slew followed by the simulated drive cap |
| Approach braking | 1.02–5.37 m/s² peak over 100 ms during final approach | Nominal drive cap from the simulator configuration |

The command header has no stamp. The analyzer places it on the most recent recorded Gazebo clock and compares with stamped ground truth. LiDAR noise is estimated from adjacent finite readings of the same beam in the second before dispatch, while the robot is stationary. The ramps and braking include upstream command shaping, measured path motion and sampling. They do not identify a drive transfer function or warrant replacing the configured physical limits.

Run from the repository root in a sourced Humble environment with NumPy and PyYAML:

```sh
python3 scripts/m3/measure_fortress.py /path/to/m2/pilot/train-static-b?-?
```

For sensitivity runs, `amr_lite.envs.m3_variation.measured_variation` draws one or two whole command-delay ticks and a noise standard deviation uniformly over 0.0297–0.0302 m from a caller-supplied generator. The bounds round the observed sensor range outward. The delay settings approximate the measured onset window at Lite's 50 ms resolution. Use recorded seeds and effective profiles when comparing against nominal training. Geometry, scan frequency and nominal drive caps are fixed.
