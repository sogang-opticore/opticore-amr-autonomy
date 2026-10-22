# M2 navigation measurement contract

One episode is one dispatched station goal for one robot. The predeclared mission manifest fixes the world, route, start, target, obstacle timeline, reset, timeout and ordered legs. A mission succeeds only when every prescribed leg succeeds in order within its time limit. Keep abandoned and skipped legs in the mission denominator. A fallback-assisted arrival counts as operational completion; report it separately from autonomous policy completion.

## Evidence and identity

Each invocation owns one fresh CSV and a sibling `<stem>.events.jsonl`. Use a different output path for every invocation. Carry the manifest identifier, split, robot namespace, planner/versions, launch parameters and obstacle/reset evidence alongside these files. A CSV alone cannot establish simulator provenance. The recorder uses the ROS simulation clock for chronology and monotonic wall time for throughput. A process restart gets a fresh run ID unless supplied explicitly.

| Fields | Definition |
| --- | --- |
| `version,run_id,episode_id,goal_id` | Schema 2, run identifier, per-run episode key, goal dispatch key. Passive goals use frame and ROS header stamp. The paired runner uses leg and repetition. |
| `policy,scenario_id,seed,frame` | Configured controller label, manifest scenario and seed/repetition, measurement frame. Compare like-for-like manifest conditions. |
| `started,ended,episode_time,wall_elapsed` | Start and end in ROS seconds, nonnegative simulated duration, wall duration in seconds. |
| `termination_reason,arrived,truncated,failure_type` | ARRIVED, COLLISION, TIMEOUT, PREEMPTED, INTERRUPTED or CLOCK_RESET. A deadline ends the trial even if a late arrival subsequently appears. |
| `success,collision,collision_known,contact_source` | Success = verified arrival before timeout with complete contact-free coverage. A blank collision cell means unknown. A measured positive collision proves collision even if other intervals lack coverage. |
| `goal_x/y,start_x/y,final_x/y,final_goal_distance,path_length` | Metres in the declared common frame, computed from valid stamped poses. |
| `minimum_clearance,proximity_warning` | LiDAR estimate relative to the footprint circle and configured proximity threshold. Use only as a diagnostic. |
| `fallback_count,fallback_time,shield_intervention_rate,max_inference_ms,deadline_misses,faults` | Runtime intervention, timing and faults from arbiter and health reports; a missing health stream gives no shield or inference measurement. |

The upstream DWA status is an unkeyed String. Associate it with the current goal only after a fresh matching path and subsequent active status, then require a fresh goal-frame pose within arrival tolerance. Ignore a held REACHED from the previous goal. If TF at the odometry timestamp is missing, discard that sample. Never subtract map and odom coordinates. Zero valid pose samples provide no final distance.

## Collision observation

Subscribe to the robot-scoped relative topic `m2/contacts` (std_msgs/String JSON). Each periodic sample has numeric simulation `stamp`, boolean `collision`, and `source` equal to `gazebo_contacts` or `ground_truth_geometry`. Optional `pairs` identifies bodies. Gazebo samples must include cumulative `collision_events` and `coverage_epoch` integers. The first detects lost positive packets. The second advances whenever the physics observer loses coverage, including gaps shorter than a publication period. A counter reset or increase spanning a trial boundary leaves the interval uncertain. Use the Gazebo physics probe or an independently validated geometry observer. The producer must include contact with moving obstacles while the robot is stationary, omit normal wheel–floor support, and report a positive observation even when contact ends between periodic publications. The default maximum gap is 0.25 s; physics monitoring is continuous and reports nominally at 20 Hz. A continuous source with samples bracketing dispatch and arrival is required to establish zero. A startup gap, interruption, source change or missing end coverage yields unknown rather than zero. No LiDAR range establishes collision.

Keep all rows. For an arrival with unknown contact evidence the recorder writes `arrived=1, success=0, collision_known=0, failure_type=CONTACT_UNKNOWN`. Expose unknown/invalid trials when reporting rates; a mixed set cannot establish a complete collision rate. Experimental or fake-ROS regression logs establish recorder behavior only.

## Raw decision log

JSONL events have `version,run_id,episode_id,goal_id,time,wall_monotonic,kind`. Events track goal dispatch, frame-aligned pose, path association, status, contact observations, LiDAR minimum, runtime diagnostics and terminal decision. The end event stores the reason, verified success and measured or unknown collision. Keep the original ROS bag when exact sensor rays, TF history or publisher timing are required. Inspect partial JSONL if a process was killed before its CSV snapshot. Separate actual simulation results from offline test fixtures.
