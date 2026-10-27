# M2 diagnostic scenarios

Each case starts one amr1 in a fresh copy of the warehouse with the stock random obstacle mover disabled. Existing warehouse fixtures remain. Goals and actor positions below use the ROS map frame. An actor is a model with the same visible geometry and solid collision box. The driver checks the Ignition boolean reply for creation, relocation and removal. These are scenario controls, not collision observations.

| Family | Train-case condition | What to inspect |
|---|---|---|
| Static | Dock-1 (2, 22) to Dock-A (52, 22), no new actor | Localization, path, final goal identity and motion on the nominal station transfer |
| Crossing | Same transfer; 0.5 × 0.5 m actor holds at (14, 19) for 5 s, crosses the path by moving to (14, 24.25) at t=12 s | Actual closest approach and timing, yield/continue, delay or contact |
| Head-on with bypass room | Same transfer in the open north strip; the actor holds (24, 22) for 2 s then moves west to (7.5, 22) by t=24 s | Opposing trajectories with lateral space; check whether the robot passes, waits or makes contact |
| Blockage | Right passage staging points (46.5, −7) to (46.5, 7.5); a 1.8 m wide block closes the 2.0 m passage at (46.5, 0) until removal at t=20 s | Waiting, detour or post-release recovery; inspect the actual global path to identify which occurred |

All challenge actors are 1.0–1.3 m tall and visible to the horizontal LiDAR. The scenario's clock starts at dispatch of its stamped goal. Actor positions interpolate linearly between simulation-time knots, with a 0.1 s target update interval. Removal is also scheduled in simulation time. A late driver tick exceeding its due time by 0.5 s invalidates the acquisition; commands and acknowledgements go to the run's actor log. The driven contact oracle and raw observations determine outcomes. A scheduled path intersection alone does not demonstrate that the robot encountered an actor.

The same four diagnostic families have separate station poses and schedules in the train, selection and final-test allocations. These are cases on the same map. Independent topology, randomized sensor/dynamics and seeds for the full M6 protocol remain additional evaluation dimensions.
