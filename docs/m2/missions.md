# Running a single M2 baseline trial

Run from this repository's root. Source the ROS Humble / Fortress workspace with upstream OPTICORE packages, the amr_learned_planner integration and amr_m2_contact_monitor installed. Use the same map, world and controller configuration versions for both arms. The local driver checks the installed map content and records the installed configuration, package, relevant executable and launch hashes. Before dispatch it dumps effective A*, controller, AMCL, recorder and (B1) arbiter parameters to parameters/. The upstream mesh/model directory must be discoverable by the launch file or through IGN_GAZEBO_RESOURCE_PATH.

```sh
python3 experiments/m2/run_mission.py train-crossing --describe
python3 experiments/m2/run_mission.py train-crossing --planner b0 --seed 0 --out results/m2/b0-crossing-0
python3 experiments/m2/run_mission.py train-crossing --planner b1 --seed 0 --out results/m2/b1-crossing-0
python3 -m unittest discover -s experiments/m2 -p 'test_*.py'
```

`experiments/m2/missions.json` defines named stations, a 120-s simulation deadline, frame, case IDs, exact schedules and train/selection/test assignment. Dock IDs identify floor centers or specified within-dock waypoints. Pinch IDs are diagnostic route endpoints. Every variant uses its own start/end positions across partitions. M2 uses the four train cases; selection is available for later candidate selection, and execution of the reserved test cases is disabled by this runner. Seed is a repeat index to pair corresponding trials, not the state of the upstream Gazebo sensor RNG. Alternate the order of B0/B1 across repeats.

Each invocation claims an unused results directory, a private Ignition partition and a local ROS domain (default 81; override with `--domain` for concurrent trials). It starts a new simulator and localization stack, verifies the world spawn against ground truth, checks map localization, scan and contact heartbeat, and waits for both planners and the recorder to subscribe. It checks exclusive goal and command publishers before dispatch and every 0.5 simulation seconds in flight. It publishes one stamped goal, controls the actor schedule, and shuts down its own process group. A retry requires a new output directory. Check the initial pose and incoming clocks in the bag if startup times out.

| Output | Contents |
|---|---|
| run.json | Full input condition, planner, run and trial IDs, installed/input hashes, exact launch command, dispatch time, wall and sim durations from dispatch to collected result, achieved real-time factor, effective-parameter hashes, and acquisition status. Invalid runs store the error. |
| bag/ | ROS bag with clock, raw sensors, transforms, contacts and baseline topics. Required goal/clock/contact/truth/command counts and bag data files are checked after shutdown. |
| actor.jsonl | Acknowledged actor commands and timestamped Gazebo world poses read by the contact plugin. Placement must be confirmed before dispatch. A missing observation for more than 0.75 s or a pose outside the allowed 0.5-s lag window and 0.08-m spatial tolerance invalidates the run. Empty for the nominal scene. |
| episodes.csv, episodes.events.jsonl | Recorder outcome and underlying events. Read collision coverage and termination along with success. |
| launch.log | Startup, upstream navigation, recorder and shutdown diagnostics. |

`recorded` describes completed acquisition with evidence files. It does not imply task success. A run that lacks a valid terminal row, stalls, or loses mandatory raw evidence is marked `invalid`; preserve its data for diagnosis and exclude it from the measured paired baseline.
