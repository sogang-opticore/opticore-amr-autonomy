# M2-L2 · Reference PPO, reward and episode boundaries

`packages/amr_lite/configs/m2/ppo_start.yaml` specifies the first training configuration. Instantiate `stable_baselines3.PPO(env=..., seed=7, **config["ppo"])` in M3 with four training environments, each implementing the M2-L1 observation and direct action contract. Pin Stable-Baselines3 2.4.1 and Gymnasium 1.0.0 for that run and archive the resolved Python, Torch, NumPy and device versions. The checked-in spike's custom residual PPO and its older evaluation curves remain separate experiments.

One rollout collects 1,024 transitions (256 per environment), followed by ten epochs of 64-sample minibatches. Start with two 64-unit layers for each MLP head. Fixed feature scales are already applied to the observation. Use the environment's four-frame stack exactly once. Use the raw reward and record its components. Overall training steps, wall-clock budget and final sample count depend on measured throughput; they must be recorded before the corresponding comparison. Seed 7 exercises the training path. Reserve the multiple training seeds and ablations for the later policy milestone.

## Reward

At each 0.05 s control tick accumulate `2*(previous_remaining_m-current_remaining_m) - 0.01`. Compute remaining arc length on the same directed reference route for both measurements. Rebase the previous length when the route changes to avoid rewarding or penalizing an A* replan. Add +20 on goal attainment or −20 on collision once; collision wins if both occur in the same interval. Start without a separate idle penalty: slowing or waiting for an obstacle costs time, and can avoid a collision. Ground-truth collision and progress are permissible training signals. Inputs for inference remain the stamped observation contract.

| Event | Gymnasium boundary and accounting |
|---|---|
| Active goal reached within 0.4 m without contact | `terminated=True`, `truncated=False`, success. Match this tolerance to the upstream namespaced launch. Use the mission evaluator's full completion rule for the operational result. |
| Collision, including a hit while stopped | `terminated=True`, `truncated=False`, collision. Capture the terminal observation. |
| Manifest time limit, neither terminal event | `terminated=False`, `truncated=True`, timeout. Preserve the terminal observation for the vector environment's time-limit bootstrap. |
| Invalid action, upstream reset, missing initial route, invalid sensor stream | Stop the command. Mark the reason and affected episode as invalid for training statistics. A collector must reject corrupt transitions and report their counts; they cannot be counted as successful completions. |
| Goal replacement or cancellation during a single-target training episode | End collection for the old goal and reset per-goal progress. Record the interruption separately. Operational multi-goal runs use the mission recorder. |

Training uses the train split. Use selection episodes to select a checkpoint, then lock the model identifier and normalization settings before opening any test outcomes. Select on timeout-bounded collision-free completion with interventions/efficiency reported alongside it. A reward curve does not measure that outcome.

## Verification boundary

`python -m pytest packages/amr_lite/tests/test_m2_ppo_start.py` exercises reward decomposition and conflicting terminal/time-limit cases against the Lite environment without optimization. The existing Gymnasium wrapper delegates reset and step to that environment. M3 must wire stamped sensor history, invalid-transition handling and the manifest deadline, run the SB3/Gymnasium environment check, and save/reload model plus preprocessing configuration. The M2 tests use a synthetic short development episode; they establish neither B0 performance nor convergence.
