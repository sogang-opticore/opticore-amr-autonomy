# Deploy와 ROS2 통합 — policy bundle과 `amr_learned_planner`의 실제 동작

이 문서는 학습 checkpoint를 torch 없는 policy bundle로 바꾸는 코드(`amr_lite/deploy/`)와, 그 bundle을
`opticore-amr`에서 돌리는 ROS2 패키지(`integrations/ros2/amr_learned_planner/`)가 **실제로 어떻게
동작하는지** 설명한다. 계약(bundle 형식 §7, 토픽 §8, arbiter 모드 §9, health fault §10)은
[ARCHITECTURE.md](../ARCHITECTURE.md), 단계와 gate는 [ROADMAP.md](../ROADMAP.md) M5·M6, 설치·실행 명령은
[패키지 README](../../integrations/ros2/amr_learned_planner/README.md)를 따른다. 기준 commit `4e83d29`, 2026-10-14.

## 1. 역할과 범위

| 구분 | 내용 |
|---|---|
| 담당 | R4 ROS2(bundle·노드·launch), R5 Gazebo(recorder·paired runner) ([internals 색인](README.md)) |
| 포함 | checkpoint → bundle export, NumPy runtime, `LearnedPlannerCore`, `CommandArbiter`, `HealthMonitor`, `EpisodeRecorder`, `GoalSequence`, rclpy 노드 4개, launch helper, `fake_ros` 테스트 하네스 |
| 제외 | 관측·재표본화·shield 알고리즘 자체([SENSORS_AND_SAFETY](SENSORS_AND_SAFETY.md)), 정책 학습([LEARNING](LEARNING.md)), Lite paired 실행과 gap report([EVALUATION_AND_EXPERIMENTS](EVALUATION_AND_EXPERIMENTS.md)) |
| 상태 | ROS-free 테스트(ROS 패키지 26개, `test_deploy.py` 7개)만 통과했다. ROS2/Gazebo 실기동은 한 번도 하지 않았다 |

ROS 쪽 판단 로직은 모두 `core/`(rclpy import 없음)에 있고, 노드는 메시지 변환·timer·파라미터만 한다.
관측·행동 변환·shield는 `amr_lite`의 학습용 함수를 그대로 import하므로 복사본이 없다.

## 2. 파일 지도

| 파일 | 책임 |
|---|---|
| `packages/amr_lite/amr_lite/deploy/export.py` | `export_bundle`: checkpoint → `bundle.json`/`weights.npz`/`golden.npz`. torch 필요 |
| `packages/amr_lite/amr_lite/deploy/runtime.py` | `PolicyRuntime`: load 시 검사, NumPy 추론, residual 합성. `BundleError(code)` |
| `packages/amr_lite/amr_lite/cli/export_policy.py` | export 후 즉시 `PolicyRuntime.load`·golden 재검증, JSON 요약 출력 |
| `packages/amr_lite/amr_lite/cli/verify_bundle.py` | torch 없이 bundle 검사 + 지정 robot profile과 대조(exit 0/1/2) |
| `packages/amr_lite/tests/test_deploy.py` | export·runtime parity·변조 거부·torch 미import 테스트 |
| `integrations/ros2/amr_learned_planner/package.xml`, `setup.py`, `setup.cfg` | ament_python, console script 4개, `launch/*.launch.py`·`config/*.yaml` 설치 |
| `.../amr_learned_planner/core/planner_core.py` | `PlannerConfig`, `PathTracker`, `LearnedPlannerCore`(tick 상태 머신), `coerce` |
| `.../amr_learned_planner/core/health.py` | `HealthMonitor`: permanent/transient fault, deadline 집계 |
| `.../amr_learned_planner/core/arbiter.py` | `ArbiterConfig`, `CommandArbiter`: 모드·fallback 상태 머신 |
| `.../amr_learned_planner/core/recorder.py` | `EpisodeRecorder`: goal 단위 episode → Lite CSV 열 |
| `.../amr_learned_planner/core/sequence.py` | `GoalSequence`: paired set leg 순서, 실패 시 abort |
| `.../amr_learned_planner/learned_planner_node.py` | `LearnedPlannerNode`: 입력 변환, path TF, 20 Hz 제어 timer, 런타임 파라미터 |
| `.../amr_learned_planner/cmd_arbiter_node.py` | `CmdArbiterNode`: `cmd_vel`·`dwa/status`·`cmd_arbiter/state` 단일 발행자 |
| `.../amr_learned_planner/nav_episode_recorder_node.py` | `/goal_pose`마다 episode 1행(수동 주행용) |
| `.../amr_learned_planner/goal_sequence_runner_node.py` | paired set goal 발행 + 기록(M6) |
| `.../amr_learned_planner/ros_common.py` | QoS 상수, 파라미터 선언, `RECORDER_TOPICS`, `subscribe_recorder` |
| `.../amr_learned_planner/launch_helpers.py` | `make_learned_nav_nodes`: A*·DWA(fallback)·learned·arbiter·recorder 노드 묶음 |
| `.../launch/learned_nav.launch.py`, `paired_runner.launch.py` | launch 인자 → helper / runner 노드 |
| `.../config/learned_planner.yaml`, `cmd_arbiter.yaml`, `nav_episode_recorder.yaml` | 기본 파라미터 |
| `.../test/fake_ros.py`, `conftest.py`, `test_*.py` | 가짜 rclpy·msg·tf2, smoke bundle fixture, 테스트 26개 |

`...` = `integrations/ros2/amr_learned_planner`.

## 3. 동작 흐름

### 3.1 checkpoint → bundle → runtime

```text
[학습 장비, torch]
best.pt ─torch.load─► payload{kind, env_config?, policy{mode,residual_scale}, baseline_config, config, ...}
   │                       env_config 없음(BC, M1 PPO) → resolve_env_config(--robot-profile | lite_default), source=assumed
   ├─load_policy──► _actor_graph(kind) ─► network spec(op 목록) + weights{w0,w1,...} ──► weights.npz
   └─────────────► golden_observations(env_config, 64, seed=5) ─► policy.predict_batch ──► golden.npz
                    (S3, D0, S1, P-PINCH rule baseline + N(0,0.4) 잡음 rollout)
   bundle.json(마지막에 기록): 계약, profile, policy, safety, provenance, files{weights,golden SHA-256}

[로봇, NumPy+PyYAML]
PolicyRuntime.load: bundle.json 읽기 → format/version → files SHA-256 → obs 계약 → act 계약
                    → weights.npz → (residual이면 RuleBasedPlanner 생성) → golden parity ≤ 1e-5
   실패: BundleError(BUNDLE_LOAD_FAILED | CONTRACT_MISMATCH | GOLDEN_PARITY_FAILED)
LearnedPlannerCore.from_bundle: 실패 → runtime=None + permanent fault
                                성공 → _configure_from_bundle: profile 검사(strict→PROFILE_MISMATCH | warn→로그만)
```

### 3.2 `LearnedPlannerCore.step(now)` 한 번 (20 Hz timer)

```text
odom 없음 ─────────────────────────────► WAITING_ODOM, (0,0), frames 비움
odom 나이 > odom_timeout ──────────────► STOPPED + ODOM_STALE
경로 없음 ─────────────────────────────► STOPPED          reached ─► REACHED
goal까지 ≤ goal_tolerance (또는 +ε & |v| 작음) ─► REACHED + event GOAL_REACHED
PathTracker.offset > path_lost_offset ──► PATH_LOST (+event, path_lost_repeat_sec 간격)
scan 없음/나이 > scan_timeout/stale_scan 주입 ─► (0,0) + SCAN_STALE, status 유지
runtime 없음 ──────────────────────────► STOPPED
build_observation: resample_scan(scan stamp별 cache) → frames(deque 4) → PathTracker.state → build_observation
비유한 관측 ───────────────────────────► (0,0) + NON_FINITE_OBSERVATION
runtime.predict (+residual base) ← perf_counter로 inference_ms 측정, fault 주입, record_inference
비유한 행동 ───────────────────────────► (0,0) + NON_FINITE_ACTION
scale_action(bundle v_max,w_max) → scan_points + tracker → ScanSafetyShield.apply(가감속 제한 포함)
STOP 연속 ≥ emergency_after_sec ───────► EMERGENCY(+event) 아니면 NORMAL → PlannerOutput(v,w,status,events,health,diag)
```

### 3.3 arbiter 상태도 (`learned` 모드)

```text
 start ─► [FALLBACK] ── allow_recovery && ok && (now−state_since) ≥ min_fallback_sec
   (active=dwa) ▲         && ok 연속 ≥ recover_after_sec ───────────────► [LEARNED] (active=learned)
                │                                                            │
                │ (now−state_since) ≥ fallback_hold_sec                      │ !ok: health stale | unhealthy
                └──────── [FALLBACK_HOLD] (active=stop, 매 tick (0,0)) ◄─────┘      | learned cmd stale
 mode=dwa|shadow: [DWA] 고정(active=dwa), 전이 없음
```

계약 표는 [ARCHITECTURE §9](../ARCHITECTURE.md), 구현 세부는 §4.5.

## 4. 핵심 구성요소 상세

### 4.1 Export — `packages/amr_lite/amr_lite/deploy/export.py`

`_Graph.ops`가 torch 모듈을 재귀로 걸으며 op dict를 만들고, 텐서를 `w0, w1, ...` 순서로 float32 저장한다.
지원 모듈은 `nn.Sequential`, `nn.Linear`(`linear`), `nn.Conv1d`(`conv1d`, padding 0·dilation 1·groups 1만,
`stride` 기록), `nn.Tanh`, `nn.ReLU`뿐이고 나머지는 `ValueError`다.

| `kind` | 걸어가는 모듈 (`_actor_graph`) | `network.type` | `output` |
|---|---|---|---|
| `bc` | `ActorMLP.model`(Linear-ReLU-Linear-ReLU-Linear-Tanh) | `mlp` | `identity`(Tanh가 층에 포함) |
| `ppo`(M1) | `ActorCritic.body` + `actor_mean` | `mlp` | `tanh` |
| `ppo_stable` | `StableActorCritic.actor_body` + `actor_mean` | `mlp` | `tanh` |
| `ppo_v2` MLP | `PolicyNetwork.actor_body` + `actor_mean` | `mlp` | `tanh` |
| `ppo_v2` CNN | `actor_body[0]`(`ScanCNNEncoder`)의 `conv`·`state`, `actor_body[1]` + `actor_mean` | `scan_cnn`(`frames` 4, `rays` 72 고정) | `tanh` |

- critic·`log_std`는 export하지 않는다. 결정적(deterministic) 평균 행동만 재현한다.
- direct/residual: `payload["policy"]["mode"]`(기본 `direct`), `residual_scale`(기본 0.5). residual이면
  `policy.base_policy = {name: lite_rule_baseline, env_config: BASE_POLICY_ENV_KEYS 6개, config: payload["baseline_config"] 또는 현재 configs/baseline.yaml}`.
  `BASE_POLICY_ENV_KEYS`(`lidar_max_range`, `goal_distance_scale`, `max_cross_track_error`, `v_max`, `w_max`, `decel_max`)는
  `RuleBasedPlanner`가 읽는 env 키 전부다.
- `robot_profile.values`는 export의 `PROFILE_KEYS` 15개 중 env_config에 있는 것, `safety`는 `scan_shield`,
  `scan_tracker`, `shield_strategy`다.
- provenance: checkpoint 절대 경로와 SHA-256, `payload["config"]`를 `json.dumps(sort_keys=True)`한 SHA-256,
  `git_snapshot()`의 `commit`·`branch`·`dirty`, `training_seed`, `total_timesteps`, `selection`(`--selection` JSON 또는 `payload["evaluation"]`).
- `files`에는 `weights.npz`, `golden.npz`의 SHA-256만 들어간다. `bundle.json` 자체는 해시되지 않는다(§8).
- 쓰기 순서는 weights → golden → `bundle.json`이므로 중간에 죽으면 `bundle.json`이 없어 load가 실패한다.

### 4.2 Runtime — `packages/amr_lite/amr_lite/deploy/runtime.py`

`PolicyRuntime.load(directory, verify=True)`의 검사 순서와 fault code:

| 순서 | 검사 | 실패 code |
|---|---|---|
| 1 | `bundle.json` 읽기·JSON 파싱 | `BUNDLE_LOAD_FAILED` |
| 2 | `(format, format_version) == ("opticore-learned-planner-bundle", 1)` | `BUNDLE_LOAD_FAILED` |
| 3 | `files`의 각 파일 존재 + SHA-256 일치 | `BUNDLE_LOAD_FAILED` |
| 4 | `observation_contract.version == "amr-lite-obs-v1"` | `CONTRACT_MISMATCH` |
| 5 | `action_contract.version == "amr-lite-act-v1"` | `CONTRACT_MISMATCH` |
| 6 | `weights.npz` → float32 dict, 생성자(residual이면 `RuleBasedPlanner(base env_config, base config)`) | — |
| 7 | `verify_golden`: `max|predict_batch(golden.observations) − golden.actions| ≤ 1e-5` | `GOLDEN_PARITY_FAILED` |

- **profile 검사는 runtime에 없다.** ROS에서는 `LearnedPlannerCore._configure_from_bundle`, 오프라인에서는 `verify_bundle`이 한다.
- 추론(`policy_actions`): 입력을 `(N, 296)` float32로 바꾼다. `mlp`는 op를 차례로 적용한다. `scan_cnn`은
  `obs[:, :288]`를 `(N, 4, 72)`(frame=channel)로 바꿔 `conv` op(`sliding_window_view` + `einsum`)를 적용·flatten하고,
  `obs[:, 288:]`에 `state` op를 적용해 이어 붙인 뒤 `layers`를 적용한다. 마지막에 `output` 활성화.
- `predict_batch`: residual이면 행마다 `RuleBasedPlanner.predict`로 `a_base`를 구해 `clip(a_base + s·a_policy, −1, 1)`.
- `observation_overrides = {"history_mode": ...}`는 Lite 평가가 쓰고, ROS core는 `observation_contract`를 직접 읽는다.
- 2026-10-14 로컬 측정(Apple M2, 256×256 MLP residual smoke bundle, `step()` 200회, shield CLAMP 상황):
  `inference_ms` 중앙값 약 0.06 ms, `step()` 전체 중앙값 약 0.54 ms. deadline 25 ms 대비 여유가 크다.

### 4.3 `LearnedPlannerCore` — `.../core/planner_core.py`

**bundle 설정.** `_configure_from_bundle`은 bundle `robot_profile.values`와 노드 파라미터를 비교한다.

| 비교 대상 | 규칙 |
|---|---|
| `v_max`, `w_max`, `accel_max`, `decel_max`, `angular_accel_max`, `robot_radius` | bundle에 키가 있을 때만, 차이 > 1e-6이면 불일치 |
| `lidar_model.type` | `raw`가 아니면 무조건 불일치(Phase 0 `contract` bundle은 항상 걸림) |
| `lidar_model.mount_x/y/yaw` | `raw`일 때 `lidar_mount_*`와 차이 > 1e-6이면 불일치 |

`profile_check == "strict"`일 때만 permanent `PROFILE_MISMATCH`(details는 JSON)이고, 그 외 값(`warn` 등)은
`profile_mismatches` 속성과 노드 warn 로그만 남긴다. shield 설정은 **명령 한계는 노드 파라미터**,
`scan_shield`는 bundle `safety`, `policy_dt = 1/control_rate`로 만든다. tracker는 bundle `safety.scan_tracker`를 쓴다.

**입력.** `on_odometry`는 최신값만 보관한다. `on_scan`은 `pose`가 없으면 그 순간의 odom pose를 붙인다(shield의
motion compensation용, 관측에는 쓰지 않음). `on_path(points)`는 `None`이면 무시, `[]`면 경로 삭제(`STOPPED`),
`reached` 상태에서 새 경로 끝점이 로봇에서 `reached_new_path_rearm_dist` 이내면 같은 goal의 주기 재계획으로 보고 무시한다.
그 외에는 `PathTracker.set_path`(index 초기화)다.

**경로 투영(`PathTracker`).** 첫 tick은 전체 경로를, 이후는 직전 segment index 기준 `[index − path_search_back, index + path_search_ahead]`
segment 창만 탐색한다(`nearest_segment`/`path_state(segment_range=...)`). 뒤로 2 segment까지 허용하므로 엄밀한
단조는 아니고, 창 크기는 거리가 아니라 **segment 개수**다. `offset()`은 창을 쓰지만 index를 갱신하지 않고 `state()`만 갱신한다.
새 경로가 올 때마다 index가 초기화되어 그 tick은 전체 탐색이다.

**frame stack.** `deque(maxlen=4)`. 비어 있으면 현재 contract scan 4개로 채우고, 이후 tick마다 최신 contract scan을
append한다(새 scan이 없으면 같은 scan 반복 = zero-order hold). contract scan은 `scan.stamp`별로 cache한다.
`history_mode == "newest"`면 4칸 모두 최신 scan. `WAITING_ODOM`/`STOPPED`/`REACHED`로 정지할 때 frames와
`stop_streak`를 비운다. `PATH_LOST`·`SCAN_STALE`·비유한 값으로 인한 정지는 frames를 유지한다.

**명령 처리.** `scale_action`은 bundle `action_contract`의 `v_max`, `w_max`로 단위 변환하고, 이후 가감속 제한과
shield는 노드 파라미터로 동작한다. 가감속 제한(`limit_velocity`)은 `ScanSafetyShield.apply` 안에서 **odom 측정 속도 기준**으로
적용되며 `shield_enabled: false`여도 적용된다. shield 후보 순서(PASS → SLOW → EVADE → STOP)는
[SENSORS_AND_SAFETY](SENSORS_AND_SAFETY.md)를 본다. 정지 분기(`_stop`)는 가감속 제한 없이 즉시 `(0, 0)`을 낸다.

**상태 어휘와 A\* 반응** (A\* 반응은 `opticore-amr`의 `astar_node.py` 기본 파라미터 기준):

| status | 발생 | 발행 방식 | A\* 반응 |
|---|---|---|---|
| `WAITING_ODOM` | odom 미수신 | `status_period`(1.0 s) 주기 | 없음 |
| `STOPPED` | 경로 없음/빈 경로, `ODOM_STALE`, runtime 없음 | 주기 | re-arm(`status_replan_reset_states`) |
| `NORMAL` | 정상 추종, 비유한 관측·행동으로 정지 중, `SCAN_STALE` 정지 중(직전 상태가 정지 계열일 때. 아니면 직전 상태 유지) | 주기 | re-arm, path hysteresis 허용 |
| `EMERGENCY` | shield STOP 연속 ≥ `emergency_after_sec` | event(`emergency_repeat_sec` 간격) + 주기 | 재계획(`status_replan_states`, cooldown 2.0 s, armed일 때) |
| `PATH_LOST` | offset > `path_lost_offset` | event(`path_lost_repeat_sec` 간격) + 주기 | 재계획 |
| `GOAL_REACHED` | 도착 tick | event 1회 | re-arm |
| `REACHED` | 도착 후 | 주기 | re-arm |

event는 제어 tick에서 즉시, 현재 status는 1 Hz timer로 `learned_planner/status`에 나간다. DWA의 동적 상태(`DYNAMIC_*`,
`ALIGN`, `RECOVERY` 등)는 내지 않는다.

**health·deadline.** `inference_ms`는 `runtime.predict` 호출(residual base 포함)만 `perf_counter`로 잰 **벽시계** 값이다.
재표본화·관측·shield 시간은 포함하지 않는다. `record_inference(ms, ms > deadline_ms)`가 연속 miss를 세고
`deadline_miss_limit`(3)회 연속이면 transient `DEADLINE_MISS`다. transient fault는 해당 검사 단계에 다시 도달해야
지워진다(예: `DEADLINE_MISS`는 다음 추론이 deadline 안이어야 해제).

**fault injection** (`fault_injection`, 실행 중 변경 가능):

| 값 | 구현 | 결과 |
|---|---|---|
| `nan_action` | 추론 후 행동을 `[nan, 0]`으로 교체 | 같은 tick `(0,0)` + `NON_FINITE_ACTION` |
| `slow_inference` | `inference_ms += deadline_ms + 1` | 3 tick째 `DEADLINE_MISS`(명령은 정상 계산) |
| `stale_scan` | scan stale 조건을 강제로 참 | 같은 tick `(0,0)` + `SCAN_STALE` |
| 그 외 문자열 | 검증 없음 | `none`과 같다 |

### 4.4 `HealthMonitor` — `.../core/health.py`

`PERMANENT_FAULTS` = `BUNDLE_LOAD_FAILED`, `CONTRACT_MISMATCH`, `PROFILE_MISMATCH`, `GOLDEN_PARITY_FAILED`이며 노드를 재시작해야
사라진다. 나머지는 `set_transient`/`clear`. `healthy`는 두 dict가 모두 비었을 때만 참이다. `report()` 필드는 `healthy`,
`faults`(정렬), `details`, `inference_ms`(마지막), `max_inference_ms`, `inferences`, `deadline_misses`(누적), `deadline_miss_rate`이고,
노드가 여기에 `status`, `shield_mode`, `shield_action`, `risk`를 더해 매 제어 tick `learned_planner/health`로 낸다.
정지 분기에서는 diagnostics가 없어 `shield_mode`가 `PASS`로 채워진다.

### 4.5 `CommandArbiter` — `.../core/arbiter.py`

| 상태 | `active` | `/cmd_vel` | `dwa/status` |
|---|---|---|---|
| `DWA` | `dwa` | DWA 명령 수신 즉시 전달 | DWA status 전달 |
| `LEARNED` | `learned` | 학습 명령 수신 즉시 전달 | 학습 status 전달 |
| `FALLBACK` | `dwa` | DWA | DWA |
| `FALLBACK_HOLD` | `stop` | 매 tick `(0,0)` | **아무것도 전달 안 함** |

- 초기 상태: `dwa`/`shadow`면 `DWA`, `learned`면 `FALLBACK`(`state_since = 0.0`). `dwa`/`shadow`에는 전이가 없다.
- `_learned_ok`: health 수신 나이 ≤ `health_timeout`, 마지막 health가 healthy, 학습 cmd 나이 ≤ `cmd_timeout`. 셋 중 처음 실패한 이유가 전환 reason이 된다.
- `tick(now)`(20 Hz): ok면 `healthy_since`를 유지, 아니면 `None`. 전이는 tick당 최대 1개다.
  `LEARNED → FALLBACK_HOLD`(ok 아님), `FALLBACK_HOLD → FALLBACK`(hold 경과, 건강 여부 무관), `FALLBACK → LEARNED`
  (`allow_recovery` && ok && `min_fallback_sec` && `recover_after_sec`). HOLD에서 LEARNED로 바로 돌아가는 길은 없다.
- **시작 시 LEARNED로 가는 것도 같은 recovery 규칙**이다. `allow_recovery: false`면 `learned` 모드가 영원히 DWA다(`test_recovery_can_be_disabled`).
- handover(`_enter`): 새 active source의 마지막 status를 즉시 내고, 그 source cmd가 fresh면 다음 메시지를 기다리지 않고 바로 전달한다. `FALLBACK_HOLD` 진입 시 `fallbacks += 1`.
- stale: active source cmd가 `cmd_timeout`보다 오래되면 매 tick `(0,0)`, `STOPPED`는 한 번만(`stopped_published`). 비유한 cmd는 저장하지 않으므로 그 source가 stale이 된다.
- 노드는 전이마다 warn 로그와 `cmd_arbiter/state`를 즉시 내고, 그 외에는 `state_period`(1.0 s)마다 `snapshot()`(mode, state, active, reason, since, transitions, fallbacks, learned_healthy, learned_faults, cmd 나이 2개)을 낸다. health JSON 파싱 실패는 `HEALTH_PARSE_ERROR`로 unhealthy 처리한다.

### 4.6 `EpisodeRecorder`·`GoalSequence` — `.../core/recorder.py`, `sequence.py`

- episode 1개 = goal 1개. `start`(이전 episode는 `PREEMPTED` 실패로 종료) → `on_status`가 `GOAL_REACHED`/`REACHED`면 성공 종료,
  `tick`에서 `timeout_sec` 초과면 `TIMEOUT`.
- 열: `policy, scenario_id, seed, success, collision, truncated, failure_type, episode_time, path_length, minimum_clearance,
  shield_intervention_rate, fallback_count, fallback_time, max_inference_ms, deadline_misses, faults, goal_x/y, start_x/y, final_x/y,
  final_goal_distance`. 앞 11개가 Lite `PAIRED_COLUMNS`(`amr_lite/experiments/paired.py`)와 같다.
- collision proxy(`on_scan`): 유효 return을 `x = mount_x + r·cos(θ+mount_yaw)`, `y = mount_y + r·sin(...)`로 옮겨
  `min(hypot) − robot_radius < collision_clearance`(기본 0.0)면 충돌. 도착했으면 `success=0, collision=1, failure_type=""`, 도착 못 했으면 `COLLISION_PROXY`.
- `path_length`는 pose 누적, `shield_intervention_rate` = health의 `shield_mode ∉ {PASS}` tick 비율, `fallback_*`은 `mode == learned`이면서 state가 `FALLBACK_HOLD`/`FALLBACK`인 구간, `deadline_misses`는 episode 시작 대비 증가분.
- `GoalSequence`: leg 순서대로 1개씩 대기 → `on_result` 후 `settle_sec` 뒤 다음 leg. `abort_run`(기본)이면 실패 leg 뒤 나머지를
  `skipped`(`NOT_RUN_AFTER_FAILURE`)로 두고 다음 run으로 간다. timeout은 spec `goal_timeout_sec`(warehouse_v1: 120.0). scenario id는 `WH-<set>-<NN>`, CSV `seed` 열에는 run 번호가 들어간다.

### 4.7 rclpy 노드

| 노드 | 구독 (QoS) | 발행 (depth 10) | timer | 비고 |
|---|---|---|---|---|
| `learned_planner` | `odom_topic`·`odom_fallback_topic`·`global_path_topic`·`goal_pose_topic`(RELIABLE), `scan_topic`(BEST_EFFORT) | `cmd_topic`, `status_topic`, `health_topic` | `1/control_rate`, `status_period` | `/odom`은 EKF odom이 `odom_timeout`보다 오래됐을 때만 사용. goal은 로그만 |
| `cmd_arbiter` | learned/dwa cmd·status, learned health (RELIABLE) | `cmd_vel`, `dwa/status`, `cmd_arbiter/state` | `1/rate`, `state_period` | |
| `nav_episode_recorder` | `RECORDER_TOPICS` 6개 | — | 0.5 s | 행이 늘 때마다·종료 시 CSV 전체 재작성 |
| `goal_sequence_runner` | `RECORDER_TOPICS`(goal 제외) | `goal_pose_topic`(PoseStamped, `map`, yaw quaternion) | 0.2 s | `start_delay_sec` 후 시작, `spin_once` 루프, 완료 시 종료 |

- **시간**: 모든 stamp는 header stamp가 아니라 노드 clock의 **수신 시각**이다(`now_seconds`). launch 기본 `use_sim_time: true`라
  timer·timeout은 sim time, `inference_ms`만 벽시계다.
- **path 변환**(`_on_path`): `header.frame_id`가 비면 `global_frame`. `local_frame`이면 그대로, `global_frame`이면
  `lookup_transform(local, global, Time(), timeout=path_tf_timeout)`으로 한 번 2D 변환, 그 외 frame은 무시(warn). TF 실패 시 이전 경로 유지.
- **런타임 파라미터**: `_on_parameters`는 `fault_injection`, `shield_enabled`만 반영하고 다른 이름도 `successful=True`로 받아 조용히 무시한다.
- 파라미터 선언: `declare_dataclass_parameters`가 `PlannerConfig`/`ArbiterConfig` 필드마다 기본값으로 선언하고 `coerce`로 형 변환한다(bool은 `"1"/"true"/"yes"/"on"`).

### 4.8 launch wiring — `.../launch_helpers.py`의 `make_learned_nav_nodes`

| 노드 | 파라미터 전달 | 단일 로봇(`namespace=""`) | namespace `rn` |
|---|---|---|---|
| `astar_planner` | `[astar_params.yaml 경로, overrides dict]` | remap 없음 | `base_frame=rn/base_footprint`, `dwa_status_topic=dwa/status`, `dynamic_layer_topic`; remap `/map`·`/goal_pose`·`/global_path` → `/rn/...` |
| `dwa_planner` | `[dwa_params.yaml 경로, overrides dict]` | `cmd_vel_topic=dwa_fallback/cmd_vel`, remap `/dwa/status → dwa_fallback/status` | 위 + 입력 토픽 상대화, `map_topic`, `local_frame=rn/odom_filtered`, `robot_frame=rn/base_footprint`, `/dwa/trajectories`·`/dwa/best_trajectory` remap |
| `learned_planner` | 이 패키지 YAML에서 뽑은 **dict** + `use_sim_time`, `bundle_path`, `profile_check` | YAML의 절대 토픽 | 입력 토픽 상대화, `local_frame=rn/odom_filtered` |
| `cmd_arbiter` | YAML dict + `mode` | 상대 토픽 | 같음 |
| `nav_episode_recorder`(`record`) | YAML dict + `output_csv`, `label=<mode>:<bundle 폴더명>` | 절대 토픽 | goal/pose/scan/status 상대화 |

- `node_parameters(path, node_name)`은 YAML 최상위 키의 마지막 `/` 조각이 노드 이름이거나 `/**`인 첫 항목의
  `ros__parameters`를 dict로 꺼낸다. dict 파라미터는 launch_ros가 `/**`로 넘기므로 **namespace와 무관하게 적용**된다.
- A*·DWA는 sibling YAML **파일 경로**를 그대로 넘긴다. sibling YAML 최상위 키는 `astar_planner:`, `dwa_planner:`라
  namespace 아래(`/rn/dwa_planner`)에서는 매칭되지 않는다. 이는 `opticore-amr/docs/server-run-prescription.md`(2026-10-13) D1과 같은
  문제이며, 이 패키지의 learned/arbiter/recorder는 dict 전달이라 영향이 없다(§8).
- `learned_nav.launch.py`는 `OpaqueFunction`으로 인자를 문자열로 평가해 helper를 부른다. `recorder_params_file`은 helper 인자에만 있고 launch 인자로는 노출되지 않는다.

### 4.9 `opticore-amr` 인터페이스 대조 (`ros2_ws/src/amr_navigation/`)

| 항목 | 이 패키지 | `opticore-amr` 코드 | 판정 |
|---|---|---|---|
| DWA 명령 토픽 파라미터 | `cmd_vel_topic` override | `dwa_node.py` `declare_parameter("cmd_vel_topic", "/cmd_vel")` | 일치 |
| DWA status 토픽 | remap `/dwa/status` | `create_publisher(String, "/dwa/status")` 절대 이름 고정 | 일치(절대 이름이라 remap 규칙이 맞음) |
| DWA 입력·frame 파라미터 | `odom_topic`, `odom_fallback_topic`, `global_path_topic`, `goal_pose_topic`, `scan_topic`, `map_topic`, `dynamic_layer_topic`, `local_frame`, `global_frame`, `robot_frame` | 모두 선언됨 | 일치 |
| A* status 구독 | `dwa_status_topic` | `astar_node.py` `declare_parameter('dwa_status_topic', '/dwa/status')` | 일치 |
| 재계획 트리거 | `EMERGENCY`, `PATH_LOST` | `status_replan_states` 기본값에 포함 | 일치 |
| re-arm | `NORMAL`, `STOPPED`, `REACHED`, `GOAL_REACHED` | `status_replan_reset_states` = NORMAL, ALIGN, STOPPED, REACHED, GOAL_REACHED | 일치 |
| A* 고정 토픽 | ns에서 `/map`, `/goal_pose`, `/global_path` remap | `'/map'`, `'/goal_pose'`, `'/global_path'` 하드코딩 | 일치 |
| 주기 status | 1.0 s, `REACHED`/`STOPPED`/`WAITING_ODOM` | DWA `_publish_status` 1 Hz, 같은 문자열 | 일치 |
| `path_lost_offset`, `reached_new_path_rearm_dist` | 1.8, 0.75 | DWA 기본 1.8, 0.75 | 일치 |
| `goal_tolerance` | 0.25 | DWA YAML 0.20, `multi_robot.launch.py` 0.4, A* 주기 재계획 skip 반경 0.30 | 의도된 차이. 0.25 < 0.30이라 도착 후 A* 주기 재계획이 멈춘다. helper는 DWA에 0.4를 넘기지 않는다 |
| 빈 경로 처리 | 빈 path면 즉시 `STOPPED` | DWA는 새 goal이 없으면 빈 path 무시(`_should_ignore_empty_path`) | 차이(학습 planner가 더 민감) |

토픽 이름·타입·status 문자열 불일치는 찾지 못했다. 다만 sibling의 현재 bring-up은 `robot.launch.py`(기본 `robot_name: amr1`,
모든 노드 namespace)와 `multi_robot.launch.py`뿐이고 둘 다 A*·DWA를 무조건 띄운다. `warehouse.launch.py`는 로봇을 spawn하지 않는다.
따라서 README의 단일 로봇(namespace 없음) 절차를 그대로 쓸 수 있는 bring-up이 sibling에 없고, `namespace:=amr1`로 붙이면
`/amr1/astar_planner`·`/amr1/dwa_planner`가 중복된다(추정, launch 코드 기준). M5 전에 sibling launch에 내비게이션 노드를 끄는 옵션이 필요하다.

### 4.10 `fake_ros` 하네스 — `.../test/fake_ros.py`

`install()`은 `rclpy`가 import되지 않을 때만 `rclpy`, `rclpy.node/qos/duration/time`, `geometry_msgs/nav_msgs/sensor_msgs/std_msgs.msg`,
`rcl_interfaces.msg`, `tf2_ros`를 `sys.modules`에 가짜로 넣는다. 가짜 `Node`는 `parameter_overrides`(이름만으로 매칭)로 파라미터를 주고,
publisher는 메시지를 리스트에 쌓고, subscription·timer는 테스트가 `deliver(topic, msg)`, `fire(period)`로 직접 부른다. `Clock.time`과
`Buffer.transform`은 테스트가 설정한다. `test_launch_helpers.py`는 별도로 `launch_ros.actions.Node`와 `get_package_share_directory`를
stub하고 sibling YAML을 `{}`로 만든다.

## 5. 설정

### 5.1 `learned_planner` (`config/learned_planner.yaml` = `PlannerConfig`·`NODE_PARAMETERS` 기본값과 같음)

| 파라미터 | 기본 | 의미 |
|---|---|---|
| `bundle_path` / `profile_check` | `""` / `strict` | launch `bundle:=`, `profile_check:=`가 덮어씀 |
| `control_rate` | 20.0 | 제어 timer, shield `policy_dt = 1/control_rate`(bundle `policy_dt`와 비교하지 않음) |
| `v_max`, `w_max`, `accel_max`, `decel_max`, `angular_accel_max`, `robot_radius` | 1.5, 1.5, 1.0, 2.0, 1.5, 0.36 | 로봇 한계, profile 검사 대상 |
| `lidar_mount_x/y/yaw` | 0.25 / 0.0 / 0.0 | 재표본화·shield 장착값(TF에서 읽지 않음) |
| `goal_tolerance`, `goal_reached_epsilon`, `goal_reached_stopped_speed` | 0.25, 0.03, 0.03 | 도착 판정 |
| `reached_new_path_rearm_dist` | 0.75 | 도착 후 같은 goal 재계획 무시 반경 |
| `path_lost_offset`, `path_search_back`, `path_search_ahead` | 1.8, 2, 40 | 경로 이탈·투영 창(segment) |
| `odom_timeout`, `scan_timeout` | 0.5, 0.5 | stale 판정(s) |
| `deadline_ms`, `deadline_miss_limit` | 25.0, 3 | 추론 deadline |
| `shield_enabled`, `fault_on_saturation` | true, false | 런타임 변경 가능(`shield_enabled`) / STOP 지속 시 fault |
| `emergency_after_sec`, `emergency_repeat_sec`, `path_lost_repeat_sec` | 1.0, 2.0, 2.0 | event 조건·간격 |
| `fault_injection` | `none` | §4.3 표 |
| `odom_topic`, `odom_fallback_topic`, `global_path_topic`, `goal_pose_topic`, `scan_topic` | `/odometry/filtered`, `/odom`, `/global_path`, `/goal_pose`, `/lidar` | ns면 helper가 상대화 |
| `cmd_topic`, `status_topic`, `health_topic` | `learned_planner/{cmd_vel,status,health}` | |
| `local_frame`, `global_frame`, `path_tf_timeout`, `status_period` | `odom_filtered`, `map`, 1.0, 1.0 | |

### 5.2 `cmd_arbiter` (`config/cmd_arbiter.yaml` = `ArbiterConfig`·`NODE_PARAMETERS`)

| 파라미터 | 기본 | | 파라미터 | 기본 |
|---|---|---|---|---|
| `mode` | `shadow` | | `min_fallback_sec` | 2.0 |
| `cmd_timeout` | 0.25 | | `recover_after_sec` | 2.0 |
| `health_timeout` | 0.5 | | `allow_recovery` | true |
| `fallback_hold_sec` | 0.3 | | `rate` / `state_period` | 20.0 / 1.0 |

토픽 파라미터: `learned_cmd_topic`, `learned_status_topic`, `learned_health_topic`, `dwa_cmd_topic`(`dwa_fallback/cmd_vel`),
`dwa_status_topic`(`dwa_fallback/status`), `cmd_topic`(`cmd_vel`), `status_topic`(`dwa/status`), `state_topic`(`cmd_arbiter/state`).

### 5.3 recorder·runner

| 노드 | 파라미터(기본) |
|---|---|
| 공통 `RECORDER_TOPICS` | `goal_pose_topic` `/goal_pose`, `pose_topic` `/odometry/filtered`, `scan_topic` `/lidar`, `status_topic` `/dwa/status`, `arbiter_state_topic` `cmd_arbiter/state`, `learned_health_topic` `learned_planner/health` |
| `nav_episode_recorder` | `label` `ros`, `output_csv` `nav_episodes.csv`, `robot_radius` 0.36, `collision_clearance` 0.0, `timeout_sec` 120.0, `lidar_mount_*` 0.25/0.0/0.0 |
| `goal_sequence_runner` | `paired_set` `warehouse_v1`, `runs` 1, `label` `ros`, `output_csv` `paired_episodes.csv`, `start_delay_sec` 5.0, `settle_sec` 2.0, `on_failure` `abort_run`, `global_frame` `map`, `robot_radius`·`collision_clearance`·`lidar_mount_*` 위와 같음 |

### 5.4 launch 인자

| launch | 인자(기본) |
|---|---|
| `learned_nav.launch.py` | `mode`(`shadow`, choices dwa/shadow/learned), `bundle`(`""`), `namespace`(`""`), `use_sim_time`(`true`), `include_astar`(`true`), `record`(`false`), `record_csv`(`nav_episodes.csv`), `profile_check`(`strict`, choices strict/warn), `learned_params_file`, `arbiter_params_file`, `astar_params_file`, `dwa_params_file`(모두 `""` → 기본 파일) |
| `paired_runner.launch.py` | `paired_set`(`warehouse_v1`), `runs`(`1`), `label`(`ros`), `output_csv`(`paired_episodes.csv`), `namespace`(`""`), `use_sim_time`(`true`) |

### 5.5 CLI

| 명령 | 인자 | 동작·종료 코드 |
|---|---|---|
| `python -m amr_lite.cli.export_policy` | `--checkpoint`(필수), `--output`(필수), `--robot-profile`(profile 없는 checkpoint용), `--selection`(JSON 파일) | export 후 `PolicyRuntime.load` + `verify_golden`, 요약 JSON 출력. `golden_count`는 64 고정 |
| `python -m amr_lite.cli.verify_bundle <bundle>` | `--robot-profile`(기본 `opticore_gazebo`) | load 실패 exit 1(`fault` 출력), `v_max`·`w_max`·`accel_max`·`decel_max`·`angular_accel_max`·`lidar_model` 중 불일치 exit 2, 모두 같으면 exit 0 |

`verify_bundle`은 bundle을 **profile YAML**과 비교하고, 노드는 bundle을 **노드 파라미터**와 비교한다. 두 기준이 같아야
(`learned_planner.yaml` = `opticore_gazebo`) exit 0이 노드 수락을 뜻한다. `verify_bundle`은 `robot_radius`·mount를 직접 보지 않는다.
실행 예는 [RUNBOOK](../RUNBOOK.md) §11 Gate 8.

## 6. 불변식과 검증

| 불변식 | 보호하는 테스트 |
|---|---|
| NumPy runtime = torch(≤ 1e-5): `bc`, `ppo_stable`, PPO v2 residual MLP, PPO v2 direct CNN(`newest`) | `packages/amr_lite/tests/test_deploy.py::test_numpy_runtime_reproduces_torch_policy` |
| bundle에 mode·profile·history 기록, CNN은 `scan_cnn` | `test_deploy.py::test_bundle_records_mode_profile_and_history` |
| 계약 버전 변경 → `CONTRACT_MISMATCH`, 파일 변조 → `BUNDLE_LOAD_FAILED`, 해시까지 고친 변조 → `GOLDEN_PARITY_FAILED` | `test_deploy.py::test_tampered_or_incompatible_bundles_are_refused` |
| runtime이 torch를 import하지 않음 | `test_deploy.py::test_runtime_does_not_import_torch` |
| ROS core 관측 = Lite 관측(≤ 1e-6, S3·D0·P-PINCH), golden 관측 재현 | `test_planner_core.py::test_core_observation_matches_lite_environment`, `::test_core_reproduces_golden_observation_vectors` |
| Lite 폐루프 도착, `GOAL_REACHED` 1회, 도착 후 `(0,0)` | `test_planner_core.py::test_closed_loop_core_drives_the_lite_robot_to_the_goal` |
| status 어휘, `SCAN_STALE`·`ODOM_STALE`, `PATH_LOST` event 간격 | `test_planner_core.py::test_status_vocabulary_and_input_faults` |
| `nan_action`·`slow_inference` → health fault | `test_planner_core.py::test_fault_injection_reports_health_faults` |
| shield STOP 지속 → `EMERGENCY` event 2회(4 s) | `test_planner_core.py::test_shield_saturation_raises_emergency_for_a_replan` |
| 같은 goal 재계획에 `REACHED` 유지, 새 goal에 재가동 | `test_planner_core.py::test_reached_holds_against_replans_of_the_same_goal` |
| bundle 없음 → `BUNDLE_LOAD_FAILED`, strict/warn profile | `test_planner_core.py::test_bundle_and_profile_faults` |
| `dwa`/`shadow`는 DWA만 전달 | `test_arbiter_recorder.py::test_dwa_and_shadow_modes_forward_only_the_dwa` |
| `learned`는 DWA로 시작해 `min_fallback_sec` 후 handover | `test_arbiter_recorder.py::test_learned_mode_starts_on_dwa_and_hands_over_when_healthy` |
| fault → HOLD 정지 → DWA → 복귀, handover 즉시 전달 | `test_arbiter_recorder.py::test_fault_holds_a_stop_then_falls_back_and_recovers` |
| stale → `(0,0)` + `STOPPED` 1회, NaN = 미수신, 잘못된 mode 거부 | `test_arbiter_recorder.py::test_stale_active_source_stops_once_and_nan_commands_count_as_missing` |
| `allow_recovery: false` → `FALLBACK` 유지 | `test_arbiter_recorder.py::test_recovery_can_be_disabled` |
| recorder 성공·timeout·collision proxy·fallback·CSV, goal sequence abort·run 반복 | `test_arbiter_recorder.py::test_recorder_success_timeout_collision_and_fallbacks`, `::test_goal_sequence_orders_legs_aborts_on_failure_and_repeats_runs` |
| 노드 배선(토픽 이름, map→odom 변환, `/odom` 무시, fault 파라미터), bundle 없는 노드 | `test_nodes_fake_ros.py::test_learned_planner_node_publishes_commands_status_and_health`, `::test_learned_planner_node_without_bundle_has_no_authority` |
| arbiter·runner·recorder 노드 배선 | `test_nodes_fake_ros.py::test_cmd_arbiter_node_hands_over_and_forwards_status`, `::test_goal_sequence_runner_sends_legs_and_records`, `::test_nav_episode_recorder_node_writes_rows` |
| helper의 namespace remap·파라미터 dict | `test_launch_helpers.py::test_namespaced_robot_wiring`, `::test_single_robot_defaults_keep_absolute_topics` |

모든 bundle 기반 ROS 테스트는 `conftest.py`의 `bundle_dir`(PPO v2 residual smoke, `opticore_gazebo`)을 쓴다. residual 초기 행동은
rule baseline과 같으므로 폐루프 테스트는 **배선과 계약**을 증명하지 학습 정책의 주행 능력을 증명하지 않는다.

**테스트되지 않는 것 (코드 경로)**

- M1 `ppo` kind export, PPO v2 direct MLP, residual + CNN 조합. `export_policy`·`verify_bundle` CLI(종료 코드 포함).
- `profile_check: warn` + `mode:=learned` 조합(막는 코드도 없다, §8). `stale_scan` 주입, `/odom` fallback 실제 사용, `health_timeout` 경로, `HEALTH_PARSE_ERROR`.
- recorder의 `REACHED` 경합, 좌표계, 건너뛴 leg 처리(§8). `learned_nav.launch.py`·`paired_runner.launch.py` 자체와 runner의 namespace 실행.

**ROS-free 테스트가 증명할 수 없는 것 (실제 ROS2/Gazebo 필요)**

- QoS 호환, 이름 해석·remap(가짜 Node는 토픽 문자열을 그대로 키로 쓴다), YAML 파라미터 파일 매칭(가짜는 이름만 본다 → D1 재현 불가).
- 실제 timer 주기·executor 경합·sim time·RTF, TF 지연·외삽, `ros2 param set` 형 검사, 설치된 entry point·share 파일.
- DWA·A*와의 상호작용(status 기반 재계획, path hysteresis, dynamic layer), Gazebo 동역학·odom 지연, 실제 LiDAR 잡음, 다중 로봇.
- ROS 컨테이너에서 `colcon test`를 돌리면 rclpy·launch_ros가 진짜라 `test_nodes_fake_ros.py`·`test_launch_helpers.py`가 skip되고,
  `--no-deps` 설치라 torch가 없으면 `bundle_dir` 테스트도 skip된다. 로컬에서 torch import를 막고 실행하면 14 passed, 12 skipped였고,
  ROS 컨테이너에서는 `test_arbiter_recorder.py`의 8개만 실행될 것으로 본다(추정).

## 7. 확장 방법

### 7.1 bundle에 새 network op 추가

1. `export.py`의 `_Graph.ops`에 `isinstance` 분기를 추가해 op dict(`{"op": ..., 가중치 이름, 하이퍼파라미터}`)와 `_store`로 가중치를 남긴다.
2. `runtime.py`의 `PolicyRuntime._run`(가중치 있는 op) 또는 `_activation`(무가중치)에 같은 이름을 구현한다. 모르는 op는 `_activation`에서
   `BUNDLE_LOAD_FAILED`가 나는데, 이는 load 중 `verify_golden`이 추론을 돌릴 때 드러난다.
3. 옛 runtime이 새 bundle을 잘못 해석할 수 있으면 `BUNDLE_FORMAT_VERSION`과 `SUPPORTED_FORMAT`을 함께 올리고 ARCHITECTURE §7을 고친다.
4. `test_deploy.py`의 `checkpoints` fixture에 그 op를 쓰는 checkpoint를 추가해 `test_numpy_runtime_reproduces_torch_policy`에 넣는다.

### 7.2 recurrent 상태(GRU/LSTM) — [NEXT_STEPS](../NEXT_STEPS.md)의 후보

- export: gate 순서까지 포함한 GRU/LSTM 가중치 op와 초기 hidden 크기를 기록한다. golden은 단일 행이 아니라 **시퀀스**(초기 상태 포함)여야 한다.
- runtime: `predict(observation, state) -> (action, state)` 형태의 상태 있는 API가 필요하다. 지금 `predict_batch`는 행 독립이다.
- core: hidden state를 `LearnedPlannerCore`가 보관하고 frames를 비우는 지점(`_stop`의 `STOPPING_STATES`)과 새 경로 수신에서 초기화 규칙을
  정해야 한다. core는 arbiter 모드와 무관하게 계속 돌기 때문에 FALLBACK 동안에도 상태가 갱신된다(추정, 현재 구조 기준).
- 비유한 hidden state를 새 fault로 다룰지 결정하고(§7.3), 계약 버전과 ARCHITECTURE §4·§7을 함께 올린다.

### 7.3 새 health fault

1. 코드 이름을 정하고 영구 fault면 `health.py`의 `PERMANENT_FAULTS`에 넣는다.
2. `LearnedPlannerCore.step`의 알맞은 단계에서 `set_transient`/`clear`를 짝으로 호출한다. 해제는 그 단계에 다시 도달해야 일어나므로 위치를 신중히 고른다.
3. arbiter는 `healthy`만 보므로 수정이 필요 없다. recorder는 `faults` 열에 자동 기록한다.
4. 주입이 필요하면 `PlannerConfig.fault_injection` 주석, `step`, README fault 목록을 함께 고친다. ARCHITECTURE §10 표와 `test_planner_core.py`에 추가한다.

### 7.4 다중 로봇 launch 통합

README §4대로 `multi_robot.launch.py`의 로봇별 A*·DWA `TimerAction`을 `make_learned_nav_nodes(namespace=rn, ...)`로 바꾼다. 이때 다음을 함께 처리한다.

- D1: A*·DWA에 넘기는 sibling YAML 키를 `/**/astar_planner:`, `/**/dwa_planner:`로 바꾸거나 helper에서 dict로 넘긴다. 그전에는 fallback DWA가 코드 기본값(예: `robot_radius` 0.20)으로 돈다.
- `multi_robot.launch.py`가 DWA에 넘기던 `goal_tolerance: 0.4`를 helper는 넘기지 않는다.
- 로봇마다 `record_csv`를 다르게 준다. `paired_runner.launch.py`는 namespace에서 토픽을 상대화하지 않는다(§8).
- 로봇당 프로세스가 2개(learned, arbiter) 늘어난다. RTF 약 0.3 환경에서는 sim time 기준 timer라 제어 주기는 유지되지만 CPU 경합으로 벽시계 `inference_ms`가 커질 수 있다(추정).

## 8. 주의할 점과 알려진 한계

| # | 내용 | 근거 |
|---|---|---|
| 1 | **recorder가 이전 goal의 `REACHED`로 새 episode를 성공 처리할 수 있다.** learned planner와 DWA는 도착 후 새 경로가 올 때까지 1 Hz로 `REACHED`를 내고 arbiter가 이를 `dwa/status`로 전달한다. 새 goal로 episode를 시작한 뒤 A* 경로가 도착하기 전에 이 메시지가 오면 `success=1`, `episode_time`≈0으로 끝나고 runner는 다음 leg를 보낸다. A* 계획이 1 s를 넘으면 거의 확실하다(추정) | `EpisodeRecorder.on_status`(`SUCCESS_STATES`), `LearnedPlannerNode._on_status_timer`·`_on_goal`(로그만), DWA `_publish_status`·`_on_goal_pose`(reached 미해제), `CommandArbiter.on_status` |
| 2 | recorder의 위치는 `/odometry/filtered`(`odom_filtered` frame)인데 goal은 `map` frame이다. `start_x/y`, `final_x/y`, `final_goal_distance`는 map과 odom이 어긋난 만큼 틀린다. warehouse_v1 시작점은 map (0.0, −2.0), EKF odom은 spawn에서 0으로 시작한다 | `RECORDER_TOPICS["pose_topic"]`, `configs/paired/warehouse_v1.yaml` 주석, sibling `ekf.yaml`(`world_frame: odom_filtered`) |
| 3 | `abort_run`으로 건너뛴 leg는 CSV에 행이 없다(`skipped`는 로그만). gap report의 leg별 성공률 분모에서 빠져 실패가 과소 집계된다 | `GoalSequence.on_result`, `GoalSequenceRunnerNode._on_timer`, `experiments/paired.py::gap_report` |
| 4 | `paired_runner.launch.py`에 `namespace`를 주어도 `RECORDER_TOPICS`의 절대 이름(`/goal_pose`, `/odometry/filtered`, `/lidar`, `/dwa/status`)이 그대로라 goal이 `/rn/goal_pose`로 가지 않는다 | `paired_runner.launch.py` 파라미터 dict, `ros_common.RECORDER_TOPICS` |
| 5 | `profile_check: warn`은 "shadow 시험 전용"이라고 YAML·ARCHITECTURE §6에 적혀 있지만 강제하지 않는다. warn이면 health가 healthy라 `mode:=learned`에서 그대로 주행 권한을 받는다 | `_configure_from_bundle`, `CommandArbiter._learned_ok`, `learned_nav.launch.py` |
| 6 | ARCHITECTURE §10은 `DEADLINE_MISS`를 "추론+후처리"로 정의하지만 코드는 `runtime.predict`만 잰다. G5의 deadline miss ≤ 0.1%도 이 값 기준이다 | `LearnedPlannerCore.step`의 `started`/`inference_ms` |
| 7 | `allow_recovery: false`면 시작 시 handover도 막혀 `learned` 모드가 DWA와 같아진다 | `CommandArbiter.tick`, `test_recovery_can_be_disabled` |
| 8 | namespace 실행 시 A*·DWA YAML이 적용되지 않는다(sibling D1과 같은 원인). learned·arbiter·recorder는 dict 전달이라 안전하다. 실기동 시 `ros2 param get /rn/dwa_planner robot_radius`로 확인한다 | `make_learned_nav_nodes`, sibling `dwa_params.yaml`·`astar_params.yaml` 최상위 키 |
| 9 | sibling에 namespace 없는 로봇 bring-up이 없고 `robot.launch.py`는 A*·DWA를 항상 띄운다. README §3 절차는 sibling launch 수정 없이 실행되지 않을 가능성이 높다(추정) | §4.9 |
| 10 | G5 "shadow에서 DWA 명령 차이 기록"을 남기는 노드가 없다. recorder는 shield 개입률만 기록하므로 `learned_planner/cmd_vel`·`dwa_fallback/cmd_vel`을 rosbag으로 따로 남겨야 한다 | `EpisodeRecorder` 열, ROADMAP M5 G5 |
| 11 | `bundle.json`은 해시되지 않는다. `safety.scan_shield`(shield 설정)나 `robot_profile`을 고쳐도 golden parity에 걸리지 않고, `files` 키를 지우면 해시 검사 자체가 생략된다. provenance `dirty`는 저장소 어디든 untracked 파일이 있으면 참이고 `scope_dirty`는 기록하지 않는다 | `PolicyRuntime.load`, `export_bundle`, `experiments/manifest.py::git_snapshot` |
| 12 | 실행 중 `shield_enabled`를 끌 수 있고 막는 장치가 없다(가감속 제한은 남음). `_on_parameters`는 다른 파라미터 변경도 성공으로 응답하지만 반영하지 않는다. `fault_injection` 값은 검증하지 않는다 | `LearnedPlannerNode._on_parameters` |
| 13 | stamp는 수신 시각이고 header stamp를 쓰지 않는다. scan 지연·odom 지연이 그대로 관측과 shield 입력에 섞인다. bundle 학습의 `policy_dt`(0.05)와 `control_rate`를 비교하지 않는다 | `_odometry`, `_on_scan`, `_configure_from_bundle` |
| 14 | `/global_path` 콜백의 TF 조회는 단일 스레드 executor에서 최대 `path_tf_timeout`(1.0 s) 막을 수 있다. TF가 없을 때 그동안 제어 timer가 멈추면 학습 cmd가 stale이 되어 fallback한다(추정, sibling 처방전 D3와 같은 구조) | `LearnedPlannerNode._on_path` |
| 15 | transient fault는 해당 검사에 다시 도달해야 지워진다. 예: 정지 상태에서는 추론이 없어 `DEADLINE_MISS`가 남고 `learned` 모드 복귀가 늦어진다 | `HealthMonitor.record_inference`, `LearnedPlannerCore.step` |
| 16 | collision proxy는 0.36 m 원 기준이라 0.60 × 0.40 m 차체 옆 여유를 충돌로 셀 수 있다. `shield_intervention_rate` 분모에는 정지 tick도 들어간다 | README §7, `EpisodeRecorder.on_scan`·`on_health` |
| 17 | sibling 서버 런의 RTF 약 0.3: timer·timeout은 sim time이라 계약 주기는 유지되지만 벽시계 `inference_ms`와 CPU 경합은 커진다. learned 노드 tick은 로컬 측정 약 0.54 ms라 DWA(틱당 50–75 ms, 처방전 D2)보다 가볍다 | §4.2 측정, `server-run-prescription.md` §0·§3.2 |
