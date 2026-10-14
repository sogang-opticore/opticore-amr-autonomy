# Sensor, Planning and Safety Internals

이 문서는 AMR-Lite의 센서 모델(`sensors/`), 경로 계획(`planning/astar.py`, `path_utils.py`,
`rule_baseline.py`), 안전 계층(`planning/safety_shield.py`, `scan_shield.py`)이 코드에서 실제로
어떻게 동작하는지 설명한다. 관측·행동·shield 계약 자체는 [ARCHITECTURE.md](../ARCHITECTURE.md),
마일스톤과 gate는 [ROADMAP.md](../ROADMAP.md)를 따른다. 경로는 `packages/amr_lite/` 기준이다.

## 1. 역할과 범위

| 계층 | 구성요소 | 도입 | 기본 설정에서 |
|---|---|---|---|
| 센서 | contract LiDAR(`envs/raycast.py`의 `lidar_scan`, 로봇 중심 72 ray) | Phase 0 | 사용 |
| 센서 | raw LiDAR 시뮬레이션 + `resample_scan` | M2 | `opticore_gazebo`에서 사용 |
| 센서 | `SensorDynamicsModel`, domain randomization | M2 | 모두 no-op, randomization off |
| 계획 | grid A* + line-of-sight 단순화 | Phase 0 | 사용 |
| 계획 | A* 벽 거리 비용(`astar_clearance`) | M2 | `opticore_gazebo`에서 on |
| 계획 | 경로 투영·local goal(`path_utils`) | Phase 0, M2에서 vectorized 검색 추가 | 사용 |
| 계획 | `lite_rule_baseline`(`RuleBasedPlanner`) | Phase 0 | 비교 기준선, BC 교사, residual base |
| 안전 | `legacy` shield(정답 지도·장애물) | Phase 0 | `shield_strategy: legacy` |
| 안전 | `predictive` shield(정답 위치 기반 tracker, CPA) | M1(`c44806c`) | off |
| 안전 | `scan` shield + `ScanObstacleTracker`(LiDAR만 사용) | M2 | protocol v2, PPO v2, DAgger에서 사용 |

ROS core(`integrations/ros2/amr_learned_planner/amr_learned_planner/core/planner_core.py`)는
`resample_scan`, `scan_points`, `ScanSafetyShield`, `ScanObstacleTracker`, `transform_points`를 그대로
import한다. `legacy`와 `predictive`는 시뮬레이터 정답을 읽으므로 Lite 전용이다.

## 2. 파일 지도

| 파일 | 책임 |
|---|---|
| `amr_lite/sensors/lidar.py` | `ContractScanSpec`, `RawLidarSpec`, `simulate_raw_scan`, `resample_scan`(`resample-v1`), `scan_points`, `contract_scan_points` |
| `amr_lite/sensors/randomization.py` | `SensorDynamicsModel`, `episode_sensor_model`(seed·randomization), `noisy_raw_ranges`, `noisy_contract_scan`, `perturb_path` |
| `amr_lite/envs/raycast.py` | `cast_rays`(정적 지도 + 원형 장애물), `lidar_scan`(contract 모델) |
| `amr_lite/envs/dynamics.py` | `RobotState`, `integrate`(unicycle 정확 적분), `limit_velocity`(가감속 제한) |
| `amr_lite/learning/action.py` | `scale_action`(행동 → 명령), `normalize_command`(명령 → 행동) |
| `amr_lite/planning/astar.py` | `astar`, `clearance_preference_cost`, `simplify_path`, `clearance_key` |
| `amr_lite/planning/path_utils.py` | `PathState`, `nearest_segment`, `path_state`, `sample_ahead`, `remaining_path_distance` |
| `amr_lite/planning/rule_baseline.py` | `RuleBasedPlanner`(`lite_rule_baseline`) |
| `amr_lite/planning/safety_shield.py` | `ShieldResult`, `RiskAssessment`, `SafetyShield`(legacy/predictive) |
| `amr_lite/planning/scan_shield.py` | `ScanSafetyShield`, `ScanObstacleTracker`, `transform_points`, 기본값 dict |
| `amr_lite/envs/warehouse_env.py` | 호출 지점: `reset`, `step`, `_sensor_scan`, `_fresh_scan`, `_update_scan_obstacles`, `_apply_scan_shield`, `_actuate`, `_perceived_robot`, `_failure_type` |
| `configs/env.yaml`, `configs/baseline.yaml`, `configs/robot_profiles/*.yaml` | 기본값(§5) |

## 3. 동작 흐름

### 3.1 `WarehouseEnv.reset`

```text
episode_sensor_model(config, seed) ─► sensor_model, _noise_rng
astar(map, start, goal, grid_resolution, robot_radius + obstacle_inflation, astar_clearance)
   └─ _PATH_CACHE(LRU 4096) 또는 scenario.reference_path 재사용
perturb_path(...)            # path_resample_spacing 또는 path_lateral_noise > 0일 때만
_action_queue  = [STOP 행동 [-1, 0]] × action_delay_steps
_next_scan_time = scan_phase(>0이면) 또는 scan_period
_sensor_scan(force=True) → scan_history 4칸을 같은 scan으로 채움
_observation() → _observation_queue = [첫 관측] × observation_delay_steps
```

A*는 reset에서 한 번만 실행된다. Lite에는 episode 중 재계획이 없다.

### 3.2 `WarehouseEnv.step`

```text
action ─(shape≠(2,) 또는 NaN → [0,0])─► _action_queue ─► scale_action ─► (target_v, target_w)
                                                                         │
  shield_strategy == "scan"                                              │ 그 외(legacy/predictive/오타)
  _apply_scan_shield:                                                    │ SafetyShield.apply(정답 map,
    points = transform_points(_scan_points, last_scan_pose → 현재 pose)  │   dynamic_obstacles, elapsed)
    tracks = scan_tracker.relative_tracks(pose, elapsed, track_min_speed)│
    ScanSafetyShield.apply(robot, target, points, tracks, shield_enabled)│
                                   └──────────────┬──────────────────────┘
                     ShieldResult(v, w, mode, action_type, risk_source …)   ← limit_velocity는 shield 안에서 적용
                                                  │
  physics_substeps(5) × physics_dt(0.01): obstacle.update → _actuate(scale·lag) → integrate → _collides
                                                  │
  _sensor_scan(): _scan_due()이면 _fresh_scan()
     raw:      simulate_raw_scan(+angular_bias) → noisy_raw_ranges → [scan] scan_points → tracker.update → resample_scan
     contract: lidar_scan(heading+bias) → noisy_contract_scan → [scan] contract_scan_points → tracker.update
                                                  │
  _observation(): _perceived_robot()(localization 오차) → path_state → build_observation → _observation_queue
```

정책(예: `RuleBasedPlanner.predict`)은 env 밖 evaluator·trainer loop에서 관측을 받아 행동을 낸다.

## 4. 핵심 구성요소 상세

### 4.1 Contract LiDAR와 raw LiDAR

계약은 ARCHITECTURE §4.2를 따른다. 구현 차이는 다음과 같다.

| 항목 | `contract`(`lidar_model.type`) | `raw` |
|---|---|---|
| 생성 | `lidar_scan`: 로봇 중심에서 `linspace(−135°, 135°, 72)` ray | `simulate_raw_scan`: 장착 pose에서 720 beam, `−3.14159…3.14159` |
| 거리 | 중심 ray의 첫 교차, `[0, 8]` | `cast_range`까지, 미반사는 `+inf` → `resample_scan` |
| 잡음 | `noisy_contract_scan` | `noisy_raw_ranges` |
| shield 입력 점 | `contract_scan_points`(72점 이하) | `scan_points`(360° raw 점 전체) |

- `RawLidarSpec.angle_increment`는 `(angle_max − angle_min) / (samples − 1)`이다. 기본값에서 약
  0.00874 rad이다.
- env의 `_raw_cast_range`는 `lidar_max_range + hypot(mount_x, mount_y) + 0.25` = 8.5 m다. 그 너머는
  계약 범위에 들어올 수 없으므로 계산하지 않는다. `angular_bias`는 실제 beam만 돌리고 보고 각도는
  그대로 둔다(로봇이 모르는 장착 yaw 오차).

`resample_scan` 구현:

1. `valid = isfinite(r) & (r >= range_min) & (r <= range_max)`. 계약의 "무효"(NaN, −inf, `< range_min`)와
   "반사 없음"(+inf, `> range_max`)은 구현상 같은 처리(점 없음)로 끝난다.
2. 점 = `(mount_x, mount_y) + r·(cos, sin)(beam + mount_yaw)`.
3. `index = rint((atan2(py, px) + fov/2) / step)`, `step = fov / (rays − 1)` = 3.803°. `0 ≤ index < 72`만
   남기므로 `|방위각| ≤ 135° + step/2`다.
4. `np.minimum.at`으로 bin별 최소 `hypot(px, py)`(로봇 중심 거리), 초기값 8.0, `clip(0, 8)`, float32.

bin은 wedge 안 최소값이라 같은 장면에서 raw→resample 결과는 중심 ray보다 짧거나 같다
(`test_dense_centered_raw_scan_agrees_with_contract_ray_cast`). `contract`로 학습한 정책은 `raw`
관측 분포를 본 적이 없다.

### 4.2 `SensorDynamicsModel`과 randomization

`episode_sensor_model(config, seed)`가 episode마다 모델을 만든다.

- 기본값은 dataclass no-op, `scan_period`는 `sensor_model.scan_period`(없으면 `policy_dt`),
  나머지는 `sensor_model`의 같은 이름 키로 덮어쓴다.
- seed: `SeedSequence([seed % 2**32, crc32(b"sensor-dynamics-model")]).spawn(2)` →
  `parameter_rng`(값 추첨, scan phase, bias 방향)와 `noise_rng`(scan 잡음·dropout, localization 잡음,
  경로 흔들림). env의 `np_random`과 분리돼 시나리오 생성을 바꾸지 않는다.
- `domain_randomization.enabled`이면 `parameter_rng.uniform() < probability`일 때 `ranges`의 **모든**
  키를 이름 정렬 순서로 함께 추첨한다(정수 키는 `[low, high]` 포함 정수). `randomized = True`.
- `randomize_scan_phase`는 `enabled` 안이지만 probability 판정 밖이다. `probability: 0.0`이어도
  scan phase는 무작위다.
- `ranges`에 모르는 키가 있으면 `KeyError`다.

| 키 | `env.yaml` | `opticore_gazebo` | DR 범위 | 적용 위치와 효과 |
|---|---:|---:|---|---|
| `scan_period` | 0.05 | 0.1 | — | `_scan_due`: 주기 사이 scan hold |
| `scan_phase` | 0.0 | — | `U(0, scan_period)` | 첫 갱신 시각. 어느 tick에 갱신할지만 바꾸고 scan 나이는 모델링하지 않는다 |
| `range_noise_std` | 0.0 | 0.03 | [0.0, 0.05] | 반사점에 가우시안, `≥ 0`으로 자름 |
| `dropout_probability` | 0.0 | — | [0.0, 0.03] | raw는 `+inf`, contract는 `max_range` |
| `angular_bias_deg` | 0.0 | — | [−1.0, 1.0] | beam(또는 contract heading)을 bias만큼 돌림 |
| `observation_delay_steps` | 0 | — | [0, 1] | 관측 FIFO |
| `action_delay_steps` | 0 | — | [0, 1] | 행동 FIFO, 첫 값은 정지 행동 |
| `v_scale`, `w_scale` | 1.0, 1.0 | — | [0.92, 1.05], [0.88, 1.05] | `_actuate`: shield 출력에 곱함 |
| `actuator_time_constant` | 0.0 | — | [0.0, 0.10] | `_actuate`: substep마다 `1 − exp(−dt/τ)` 1차 지연 |
| `collision_radius_margin` | 0.0 | — | [0.0, 0.03] | `_collides`, 충돌 분류만. shield와 reward clearance는 `robot_radius` 사용 |
| `localization_bias_xy` | 0.0 | — | [0.0, 0.06] | 크기. 방향은 `U(−π, π)` → `bias_x`, `bias_y` |
| `localization_bias_theta` | 0.0 | — | [0.0, 0.03] | 크기. 부호는 ±1 추첨 |
| `localization_noise_xy`, `_theta` | 0.0 | — | [0.0, 0.01], [0.0, 0.005] | `_perceived_robot`: 관측마다 가우시안 |
| `path_resample_spacing` | 0.0 | — | [0.05, 0.20] | `perturb_path`: 선분을 간격 이하로 조밀화 |
| `path_lateral_noise` | 0.0 | — | [0.0, 0.02] | `perturb_path`: 법선 방향 흔들림 |

- localization 오차는 `_perceived_robot`을 통해 관측의 local goal·경로 특징(index 288–290, 293–295)에만
  들어간다. 속도(291, 292), scan, shield, tracker는 참 상태를 쓴다.
- `perturb_path`: 각 선분을 `ceil(길이/spacing)` 조각으로 나누고, `N(0, σ√5)`를 길이 5 이동평균
  (`mode="same"`)으로 다듬어 내부 σ가 대략 `path_lateral_noise`가 되게 한 뒤 `np.gradient` 법선으로
  민다. 양 끝 offset은 0이고 끝점은 원래 좌표로 되돌린다.
- `noise_rng`의 추첨 횟수는 반사점 개수에 따라 달라진다. 같은 seed라도 궤적이 갈리면 이후 잡음
  수열이 달라진다(paired 평가에서 정책 간 잡음 실현값은 같지 않다).

### 4.3 A* (`astar`)

- grid는 `ceil(width/resolution) × ceil(height/resolution)`, cell 중심 좌표를 쓴다. env는
  `grid_resolution` 0.20, inflation `robot_radius + obstacle_inflation` = 0.40으로 호출한다.
- start 또는 goal이 `occupied(inflation)`이면 `[]`(→ `NO_PATH`)다.
- 8-이웃, 이동 비용 1/√2(cell 단위), heuristic은 cell 단위 유클리드 거리다. closed set이 없어 오래된
  heap 항목도 다시 확장한다(결과는 같고 비용만 든다).
- `astar_clearance.enabled`이면 차단 판정이 `planning_clearance(x, y) <= inflation`(사각형 지도는
  5 cm raster bilinear, grid 지도는 cell 거리장)으로 바뀌고 이동 비용이 `step × (1 + cost)`가 된다.
  `clearance_preference_cost`는 opticore-amr `astar_node.py`와 같은 식이다:
  `c < preferred_clearance`이면 `weight × ((pc − c)/pc)²`, `inflation < wall_avoid_clearance`이고
  `c < wall_avoid_clearance`이면 `wall_weight × max(0, desired/free − 1)²`
  (`free = max(c − inflation, min_margin)`, `desired = max(wac − inflation, min_margin)`).
- 복원한 cell 열의 첫·끝 점을 정확한 start/goal로 바꾼 뒤 `simplify_path`로 줄인다. 곡선 smoothing은
  없다. `simplify_path`는 anchor에서 보이는 가장 먼 점으로 건너뛰는 greedy line-of-sight이고 0.08 m
  이하 간격으로 `occupied`를 검사한다. 비용 모드에서는 `_RasterView`와
  `max(inflation, simplify_min_clearance)` = 0.45를 써서 단축이 벽 거리 비용을 무르지 않게 한다.
- 결과는 꼭짓점 몇 개의 sparse polyline이다. ROS A*의 조밀한 경로를 흉내 내려면 `perturb_path`를 켠다.
- 동적 장애물(D0–D4, `PD-*`, `WHD-*`)은 지도에 없으므로 경로는 그 위를 지난다.

### 4.4 경로 투영과 local goal (`path_utils`)

- `nearest_segment`: 검색 구간이 16개 이하이면 scalar loop, 초과하면 NumPy 일괄 계산 후 최종값만
  `point_segment_projection`으로 다시 계산해 scalar 결과와 비트 단위로 맞춘다. `segment_range`로 검색
  창을 제한할 수 있다.
- `path_state`: 최근접 선분 투영점, 부호 있는 cross-track(경로 왼쪽 +), heading 오차
  `wrap(경로 접선 − θ)`, `sample_ahead`로 투영점에서 경로를 따라 `lookahead_distance`(0.65 m) 앞의
  local goal(끝을 넘으면 `path[-1]`), local goal까지 거리 ρ·방위 α를 낸다. 경로 점이 2개 미만이면
  goal(또는 로봇 위치)을 local goal로 둔다.
- `remaining_path_distance`는 progress reward용 잔여 호 길이다.
- Lite env는 `segment_range` 없이 전역 검색한다. ROS core는 이전 index 주변만 검색한다
  (ARCHITECTURE §4.3). 접히는 조밀한 경로에서 두 쪽 투영이 다를 수 있다.

### 4.5 `lite_rule_baseline` (`RuleBasedPlanner.predict`)

입력은 관측 벡터뿐이라 BC 교사, residual base, ROS NumPy runtime에서 같은 코드가 돈다.
`env_config`를 생략하면 `load_config("env")`(`lite_default`)를 쓴다.

| 단계 | 계산 (`baseline.yaml` 값) |
|---|---|
| 1 | `scan = obs[216:288] × lidar_max_range` (최신 frame만, 로봇 중심 거리) |
| 2 | `ρ = obs[288] × goal_distance_scale(8.0)`, `α = atan2(obs[289], obs[290])`, `cte = obs[293] × 2.0`, `e_ψ = atan2(obs[294], obs[295])` |
| 3 | Pure Pursuit: `L = max(0.2, ρ)`, `κ = 2 sin α / L` |
| 4 | `curve_limit = sqrt(curvature_accel_limit(1.0) / max(abs(κ), 1e-5))` |
| 5 | `heading_factor = max(0.12, 1 − abs(e_ψ)/1.2)`, `path_factor = max(0.2, 1 − abs(cte)/0.8)` |
| 6 | `clearance = min(scan[24:48])`(전방 ±43.73°), `clearance_factor = clip((clearance − 0.12)/(0.75 − 0.12), 0, 1)` |
| 7 | `braking = sqrt(2·decel_max·max(0, ρ − 0.1))` |
| 8 | `v = min(v_max, curve_limit, braking) × heading_factor × path_factor × clearance_factor` |
| 9 | `w = κ·v + 0.65·e_ψ − 0.8·cte` |
| 10 | `clearance < 0.12`이면 `v = 0`, `w = 0.7 × (α < 0이면 −1, 아니면 +1)` |
| 11 | `normalize_command`: `v ∈ [0, v_max]`, `w ∈ [−w_max, w_max]`로 자르고 행동으로 변환 |

- 7단계의 ρ는 최종 goal이 아니라 local goal 거리라서, 경로 위 주행 중(ρ ≈ 0.65)에도 속도 상한이
  `lite_default` 1.285 m/s, `opticore_gazebo` 1.483 m/s다(`v_max` 1.5보다 낮다).
- opticore-amr `dwa_node.py`(5,875줄)는 dynamic window 샘플링, 동적 장애물 track·회피 목표, rejoin,
  14개 상태의 `NavState`(`DYNAMIC_BLOCKED`, `AVOIDING_DYNAMIC`, `SPIN`, `EMERGENCY` 등)를 가진다.
  `lite_rule_baseline`(44줄)은 그중 어느 것도 없으므로 그 성능을 DWA 성능으로 읽으면 안 된다.

**미지도 장애물을 피하지 못하는 이유 (코드 근거)**

1. A* 경로는 정적 지도로 한 번만 만들어지고 동적 장애물은 지도에 없다(§4.3).
2. `w`에는 장애물 항이 없다. scan은 `v`만 줄인다.
3. scan 거리는 로봇 중심 기준인데 `clearance_stop` 0.12가 `robot_radius` 0.36보다 작다. 접촉 순간
   (전방 거리 0.36)에도 `clearance_factor` = 0.381이라 목표 속도가 0.489(`lite_default`)/0.565
   (`opticore_gazebo`) m/s다. 정지·회전 분기는 장애물이 footprint 안에 들어온 뒤에만 켜진다.
4. 전방 ±43.73° 밖(측면 횡단)은 보지 않고, 최신 frame 하나만 써서 장애물 속도를 모른다.

측정(2026-10-14, protocol v2 selection split 85 episode, `opticore_gazebo`, scan shield, 비공식 1회):

| 조건 | 성공 | 동적 충돌 | TIMEOUT | SHIELD_SATURATION |
|---|---:|---:|---:|---:|
| Shield OFF | 55 | 30 | 0 | 0 |
| Shield ON | 59 | 6 | 9 | 11 |

- OFF 충돌: D1–D4·`PD-MIXED` 전부(25), `WHD-RANDOM` 4, D0 1. 충돌 시 로봇 속도 0.41–1.38 m/s,
  30건 중 29건은 직전 관측의 전방 최소 거리가 0.32–0.44 m였다(위 3번과 일치).
- ON: D1 5건 모두 충돌(로봇 속도 0–0.22 m/s, 후진 후보 없음), D2 5건 TIMEOUT(느린 선행 장애물을
  추월할 조향이 없음), D3·D4 각 5건 SHIELD_SATURATION(경로 위 정지 장애물 앞 STOP 반복).

### 4.6 공통: `ShieldResult`, 모드, 가감속 제한

- `mode`: `PASS`(원래 목표), `CLAMP`(STOP 외 후보 선택), `STOP`(STOP 후보 선택). predictive·scan의
  STOP은 감속 제한을 받으므로 `mode == "STOP"`이어도 `v > 0`일 수 있다.
- `action_type`: `PASS`, `SLOW`, `EVADE`, `REVERSE`, `STOP`. env의 `shield_action_counts` 키와 같다.
- `risk_source`: legacy는 항상 `LEGACY`, predictive는 `NONE`/`DYNAMIC_CPA`/`STATIC_ROLLOUT`/
  `DYNAMIC_ROLLOUT`, scan은 `NONE`/`SCAN_STATIC`/`SCAN_TRACK`, 비유한 목표는 `INVALID_ACTION`.
- `limit_velocity`(`amr_lite/envs/dynamics.py`): 목표를 `[v_min, v_max]`, `±w_max`로 자르고,
  `dv ≥ 0`이면 `accel_max·dt`, 아니면 `decel_max·dt`, `dw`는 `angular_accel_max·dt`로 제한한다.
  `v_min`은 REVERSE 후보에서만 `−reverse_max`다.
- **위치**: ARCHITECTURE §5.2의 "4. 가감속 제한 → 5. Shield"는 별도 단계가 아니다. 각 shield가
  `_limit`으로 최종 명령을 제한하고, predictive·scan은 rollout의 매 step에도 적용한다. Lite 동역학
  (`integrate`, `_actuate`)에는 가속 제한이 없으므로 shield의 `_limit`이 유일한 제한기다. Shield OFF
  (`enabled=False`)도 `apply`를 거쳐 PASS 명령을 `_limit`으로 제한한다.

### 4.7 세 전략 비교

| 항목 | `legacy`(`_apply_legacy`) | `predictive`(`_apply_predictive`) | `scan`(`ScanSafetyShield.apply`) |
|---|---|---|---|
| 입력 | 정답 지도·장애물 위치·속도 | 정답 지도, 정답 위치에서 추정한 속도 | LiDAR 점(로봇 좌표), scan track |
| rollout | 상수 `(v, w)`, dt 0.05 고정, `int(0.7/0.05)` = 13 step | 50 step(`ceil(2.5/0.05)`), 매 step `_limit` | 14 step(0.7 s), track 있으면 30 step(1.5 s), 매 step `_limit` |
| 정적 검사 | `map.clearance(probe, r) < 0.03` | 0.45 s 이내 `clearance ≤ 0.01` | 0.7 s 이내 점까지 거리 − r `≤ 0.03` |
| 동적 검사 | 정답 궤적(`start_delay`, `stop_time` 반영), 여유 0.03 | CPA(TTC ≤ 2.5 s) + 등속 track 외삽, 여유 0.08 | 등속 track 외삽 1.5 s, 여유 0.08 |
| SLOW | `v×0.5`, `v×0.2`(제한된 v 기준, 재제한 없음) | `shield_slow_scales` × target_v | `slow_scales` × target_v |
| EVADE | `_limit(min(v, 0.45·v_max), ±0.65·w_max)` | `min(target_v, 0.55)`, `±1.45` | `min(target_v, 0.45·v_max)`, `±0.65·w_max` |
| REVERSE | 없음 | `shield_reverse_enabled`(기본 true): −0.35, w ∈ {0, 0.8, −0.8} | `reverse_enabled`(기본 false) |
| STOP | `(0, 0)` 즉시 | `_limit(0, 0)` | `_limit(0, 0)` |
| 전부 위험 | STOP | 최대 `(최소 clearance, TTC, STOP 아님)` | 최대 `(최소 clearance, 위반 시각, STOP임)` |
| FP/FN | 계산 안 함 | 정답 audit로 계산 | 계산 안 함 |
| ROS | 불가 | 불가 | 같은 클래스 사용 |

후보 순서는 모두 SLOW → EVADE(좌 `+1` 먼저, 우 `−1`) → (REVERSE) → STOP이고, 처음으로 안전한 후보를
쓴다.

### 4.8 `SafetyShield` 세부 (legacy, predictive)

- `apply`는 먼저 `_update_tracks`를 돈다. 장애물 index별로 정답 위치 차분 속도를
  `shield_tracker_alpha`(0.65) EMA로 추정하고 `tracking_speed_error`(추정−정답 속도 차 평균)를 남긴다.
- predictive `_assess_target`: `_closest_approach`는 로봇이 현재 heading으로 `target_v` 등속 직진한다고
  두고 상대 운동의 이차식 근으로 TTC(반경합+여유 도달 시각)와 CPA 거리를 낸다. `use_cpa`는 원래
  목표·SLOW·STOP에서만 켜진다. rollout 위반 중 가장 이른 시각의 원인이 `risk_source`가 된다.
- audit: 같은 평가를 정답 속도, 여유 0으로 다시 한다. PASS인데 audit 위험이면 `false_negative`
  (enabled일 때만), 개입했는데 audit 안전이면 `false_positive`다.

### 4.9 `ScanSafetyShield` 세부

- `assess(state, target_v, target_w, points, tracks, allow_reverse)`: 로봇 좌표 `(0, 0, 0)`에서 시작해
  측정 속도 `state.v, state.w`로부터 `_limit` → `integrate`를 반복한다. 현재 pose(t=0)는 검사하지 않는다.
- 정적: 정적 horizon 안 rollout이 닿을 수 있는 거리(`max |pose| + r + static_margin + 0.05`와
  `max_point_range` 4.0 중 작은 값) 안의 점만 남기고, 각 step에서 최근접 점 거리 − r을 계산한다.
  동적 장애물의 반사점도 여기서 정지물로 취급된다.
- 동적: 각 track `(x, y, vx, vy, radius)`을 등속 외삽해 `거리 − r − track_radius ≤ dynamic_margin`을 본다.
- 반환 `(unsafe, source, 첫 위반 시각, 최소 clearance)`. `ShieldResult.predicted_ttc`와
  `predicted_minimum_clearance`는 **원래 목표**의 평가값이고, `cpa_distance`는 항상 `inf`다.
- 정책이 계속 전진을 요구하고 SLOW·EVADE가 위반이면 매 tick STOP이 선택돼 saturation이 된다. 점이
  `r + static_margin`(0.39 m) 안에 있으면 STOP과 제자리 회전까지 위반이라 최대 clearance 규칙으로
  고르고(동률은 STOP), 후진이 꺼져 있으면 앞쪽 점에서 빠져나올 후보가 없다. 뒤쪽 점은 전진 후보가
  멀어지므로 빠져나온다.

### 4.10 `ScanObstacleTracker`

- `update(robot_points, pose, time, static_filter)`: 점을 `pose`로 odom 좌표로 옮긴 뒤 `clusters`를
  만든다. 입력 순서(beam 순)대로 인접 점 간격이 `cluster_gap`(0.35) 초과인 곳에서 끊고, 점이
  `min_points`(3) 미만이거나 반경(중심에서 최대 거리)이 `max_radius`(0.85) 초과인 cluster(벽)는 버린다.
  `static_filter(cx, cy)`가 참인 cluster도 버린다.
- 연관: 모든 (detection, track) 쌍을 거리순 정렬해 `association_distance`(0.9) 이내에서 greedy로 짝짓는다.
  속도 = `velocity_alpha`(0.5) × 중심 변위/dt + (1 − 0.5) × 이전 속도, `max_speed`(2.0)로 자른다.
  짝 없는 track은 `timeout`(1.0 s)까지 유지하고, 짝 없는 detection은 속도 0, age 1인 새 track이 된다.
- `relative_tracks(pose, time, min_speed, min_age=2)`: age ≥ 2, 속도 ≥ `track_min_speed`(0.15)인
  track만 `time − last_seen`만큼 외삽해 로봇 좌표로 돌려준다.
- 새 track 속도는 0에서 시작하므로 k번째 연관 후 추정 속도는 `(1 − 0.5^k) × 실제 속도`다. 10 Hz
  scan에서 첫 사용 시점(age 2) 추정은 실제의 절반이다.
- env는 tracker를 새 scan이 들어올 때만(`_fresh_scan`) `self.elapsed`로 갱신한다. `static_filter`는
  `scan_tracker.static_filter_distance > 0`일 때 env가 `clearance(x, y, 0) < distance`로 만든다. ROS
  core는 `static_filter`를 넘기지 않는다.

### 4.11 env 계측

| info 키 | 산출 |
|---|---|
| `shield_counts`, `shield_action_counts` | step마다 `mode`, `action_type` 집계 |
| `unsafe_without_shield` | 원래 목표가 위험으로 평가된 step 수(Shield OFF에서도 계산) |
| `shield_false_positive`, `shield_false_negative` | predictive만 값이 생긴다. legacy·scan은 항상 0 |
| `shield_actual_false_negative` | shield 켜짐 + 이번 step 충돌 + `mode == PASS` |
| `shield_ineffective_intervention` | shield 켜짐 + 이번 step 충돌 + `mode != PASS` |
| `shield_risk_counts` | step별 `risk_source` 집계 |
| `shield_sustained_time` | 연속 비PASS step 최대값 × `policy_dt` |
| `failure_type == SHIELD_SATURATION` | `STOP` mode step 수 > `max(20, step_count // 2)` |

`_failure_type`은 NO_PATH → INVALID_ACTION → 충돌 → SHIELD_SATURATION → OSCILLATION → CIRCLING →
STUCK → TIMEOUT 순서로 판정하고 `success`를 보지 않는다. 시간 초과로 끝난 episode도 STOP이 많으면
SHIELD_SATURATION으로 분류된다(§4.5 측정의 11건은 모두 25.7–26.5 s 또는 18.5 s에 시간 초과로 끝났다).

## 5. 설정

`configs/env.yaml`의 최상위 shield 키(`SafetyShield.config`, env가 읽음):

| 사용처 | 키와 값 |
|---|---|
| 공통·env | `shield_enabled` true, `shield_strategy` legacy, `record_shield_trace` false |
| legacy | `shield_horizon` 0.7 (SLOW 배율·EVADE 비율·여유 0.03·dt 0.05는 코드 하드코딩) |
| predictive | `shield_static_horizon` 0.45, `shield_dynamic_horizon` 2.5, `shield_static_margin` 0.01, `shield_dynamic_margin` 0.08, `shield_audit_static_horizon` 0.45, `shield_audit_dynamic_horizon` 2.5, `shield_sim_dt` 0.05, `shield_slow_scales` [0.5, 0.2], `shield_evade_v` 0.55, `shield_evade_w` 1.45, `shield_reverse_enabled` true, `shield_reverse_max` 0.35, `shield_reverse_w` 0.8 |
| legacy·predictive track | `shield_tracker_alpha` 0.65 |

`scan_shield`(`ScanSafetyShield.settings` = `SCAN_SHIELD_DEFAULTS` + 이 dict): `horizon` 0.7,
`sim_dt` 0.05, `static_margin` 0.03, `dynamic_horizon` 1.5, `dynamic_margin` 0.08, `slow_scales`
[0.5, 0.2], `evade_v_fraction` 0.45, `evade_w_fraction` 0.65, `reverse_enabled` false, `reverse_max`
0.35, `reverse_w` 0.8, `max_point_range` 4.0, `use_tracks` true, `track_min_speed` 0.15.

`scan_tracker`(`ScanObstacleTracker.config` = `SCAN_TRACKER_DEFAULTS` + 이 dict): `cluster_gap` 0.35,
`min_points` 3, `max_radius` 0.85, `association_distance` 0.9, `timeout` 1.0, `velocity_alpha` 0.5,
`max_speed` 2.0, `static_filter_distance` 0.0(env `_update_scan_obstacles`만 읽음).

코드 기본값 dict와 `env.yaml` 값은 현재 같다. ROS core는 bundle `safety.scan_shield`·`scan_tracker`를
쓰고, 명령 한계·`robot_radius`·`policy_dt`는 노드 파라미터에서 가져온다.

`lidar_model`(`RawLidarSpec.from_env_config`): `type` contract, `raw_samples` 720, `raw_angle_min`
−3.14159, `raw_angle_max` 3.14159, `raw_range_min` 0.12, `raw_range_max` 25.0, `mount_x` 0.25,
`mount_y` 0.0, `mount_yaw` 0.0. contract 사양은 `lidar_rays` 72, `lidar_fov_deg` 270.0,
`lidar_max_range` 8.0(`ContractScanSpec.from_env_config`). `sensor_model`과 `domain_randomization`은 §4.2 표.
`domain_randomization`: `enabled` false, `probability` 1.0, `randomize_scan_phase` true.

`astar_clearance`: `enabled` false, `preferred_clearance` 0.50, `clearance_cost_weight` 9.0,
`wall_avoid_clearance` 0.55, `wall_avoid_cost_weight` 3.0, `wall_avoid_min_margin` 0.05,
`simplify_min_clearance` 0.45. `grid_resolution` 0.20, `obstacle_inflation` 0.04, `robot_radius` 0.36.

`configs/baseline.yaml`(`RuleBasedPlanner.config`): `curvature_accel_limit` 1.0, `heading_slowdown` 1.2,
`path_error_slowdown` 0.8, `clearance_stop` 0.12, `clearance_slow` 0.75, `recovery_turn_rate` 0.7.
rule baseline은 env config의 `lidar_max_range`, `goal_distance_scale`, `max_cross_track_error`, `v_max`,
`w_max`, `decel_max`도 읽는다.

Robot profile에서 이 계층에 닿는 값: `opticore_gazebo`는 `w_max` 1.5, `decel_max` 2.0,
`angular_accel_max` 1.5, `lidar_model.type: raw`, `sensor_model.scan_period` 0.1, `range_noise_std`
0.03, `astar_clearance.enabled: true`, `episode_timeout_per_meter` 2.0. `lite_default`는 override가 없다.
`shield_strategy: scan`은 profile이 아니라 `evaluation_selection_v2.yaml`, `evaluation_test_v2.yaml`,
`ppo_v2.yaml`, `dagger.yaml`의 `environment_overrides`가 정한다. PPO v2 curriculum은
`domain_randomization.probability`를 단계별 0.0 → 0.5 → 1.0으로 올린다.

## 6. 불변식과 검증

| 불변식 | 보호 테스트 (`packages/amr_lite/tests/` 기준) |
|---|---|
| raw 반사점이 가장 가까운 bin에 들어간다 | `test_lidar_resample.py::test_return_lands_in_nearest_contract_bin` |
| 장착 offset·yaw를 binning 전에 적용한다 | `::test_mount_offset_is_applied_before_binning`, `::test_mount_yaw_rotates_returns` |
| bin 최소값, NaN·−inf·`<range_min`·`>range_max`·FOV 밖 제외 | `::test_bin_keeps_nearest_return_and_ignores_invalid_or_rear_points` |
| 조밀 raw → resample ≤ contract ray(+1e-3), 중앙값 차 < 0.06 | `::test_dense_centered_raw_scan_agrees_with_contract_ray_cast` |
| raw env가 공유 resampler를 쓴다 | `::test_raw_lidar_environment_uses_the_shared_resampler` |
| `scan_points`·`contract_scan_points` 유효성 | `::test_scan_points_respect_mount_and_validity` |
| raw + σ 0.03 잡음 + 10 Hz hold + resample 관측 고정 | `test_observation_golden.py::test_lite_reproduces_golden_observations`, ROS `test_planner_core.py::test_core_reproduces_golden_observation_vectors`, `::test_core_observation_matches_lite_environment` |
| vectorized ray cast = 참조 구현 | `test_raycast.py::test_vectorized_cast_is_bit_identical_to_reference` |
| 기본 센서 모델은 identity | `test_sensor_model.py::test_default_environment_has_identity_sensor_model` |
| profile은 명시 키만 덮어쓴다 | `::test_robot_profile_overrides_only_listed_keys` |
| randomization은 seed 재현·범위 내 | `::test_domain_randomization_is_seeded_and_bounded`, `::test_zero_probability_keeps_profile_values`, `::test_unknown_randomization_key_is_rejected` |
| scan hold, 행동·관측 지연, actuation, localization bias, 경로 흔들림 | `::test_scan_hold_repeats_frames_between_ten_hertz_updates`, `::test_action_delay_executes_the_stop_action_first`, `::test_observation_delay_returns_the_previous_observation`, `::test_actuation_scale_and_lag_slow_the_robot`, `::test_localization_bias_moves_only_the_perceived_pose`, `::test_perturb_path_keeps_endpoints_and_stays_close` |
| Phase 0/M1 기본 rollout 불변(legacy·predictive·rule) | `test_legacy_parity.py::test_default_environment_reproduces_pre_change_rollouts` |
| A* 회피, 투영·local goal, no-path | `test_astar.py::test_astar_avoids_obstacle`, `::test_projection_and_local_goal`, `::test_no_path_is_classified` |
| 벽 거리 비용 = ROS 식, 경로 중앙화 | `test_warehouse_scenarios.py::test_clearance_cost_matches_ros_formula_and_centres_paths` |
| vectorized 투영 = brute force, 검색 창 | `test_path_utils.py::test_vectorized_search_on_dense_paths_matches_brute_force`, `::test_segment_range_limits_the_search_window` |
| rule baseline이 S0·S3를 통과한다, residual 초기 행동 = rule | `test_environment.py` 두 테스트, `test_ppo_v2.py::test_residual_policy_starts_as_the_rule_baseline` |
| `limit_velocity` 가감속·후진 한계 | `test_action.py::test_acceleration_and_deceleration_limits`, `test_dynamics.py::test_velocity_limit_allows_bounded_reverse_only_when_requested` |
| legacy 벽 앞 개입, NaN 정지, 접근 장애물 예측 | `test_safety_shield.py::test_emergency_stop_or_clamp_before_wall`, `::test_nan_action_stops`, `::test_approaching_obstacle_is_predicted` |
| predictive 속도 추정, 정면 접근 시 EVADE/REVERSE | `::test_predictive_shield_estimates_velocity_from_observed_positions`, `::test_predictive_shield_uses_escape_candidate_for_head_on_risk` |
| scan shield: 좌표 변환, PASS 가속 제한, 벽 앞 감속이 도달 가능, NaN 정지, track 예측, 후진 옵션 | `test_scan_shield.py`의 `test_transform_points_round_trip`, `test_clear_path_passes_with_limited_command`, `test_wall_ahead_is_braked_with_reachable_commands`, `test_invalid_target_stops`, `test_approaching_track_is_predicted_without_static_points`, `test_reverse_candidates_only_when_enabled` |
| tracker: 벽 무시·속도 추정·timeout, 정적 필터 | `::test_tracker_ignores_walls_and_estimates_velocity`, `::test_tracker_static_filter_drops_map_obstacles` |
| env에서 scan shield가 S3 충돌을 막는다 | `::test_scan_shield_environment_prevents_driving_into_an_obstacle` |
| ROS: STOP 지속 → `EMERGENCY` | ROS `test_planner_core.py::test_shield_saturation_raises_emergency_for_a_replan` |

**테스트하지 않는 것**

- `noisy_raw_ranges`·`noisy_contract_scan` 단위 동작, dropout 경로 전체(어떤 테스트도 켜지 않는다).
- `angular_bias_deg`, `collision_radius_margin`, `localization_noise_*`, `localization_bias_theta`의 효과,
  `scan_phase`가 갱신 시각에 주는 효과(범위만 검사).
- env의 `static_filter_distance` 배선(`_update_scan_obstacles`). tracker의 callable만 검사한다.
- `RuleBasedPlanner`의 수치 동작(clearance 감속, 정지·회전 분기, braking 상한). 통과 여부와 parity만 본다.
- legacy SLOW/STOP의 도달 가능성, scan shield에서 모든 후보가 위험할 때의 선택 규칙과 EVADE 순서.
- tracker의 연관 gate·`max_speed`·`min_age`, contract 모드(72점) tracker.
- `SHIELD_SATURATION` 분류, FP/FN·`shield_actual_false_negative`·`shield_ineffective_intervention` 계측.
- `simplify_path`·`_RasterView` 단독 동작, `shield_strategy` 오타 처리.

## 7. 확장 방법

**새 shield 전략**

1. `planning/`에 클래스를 만들고 `ShieldResult`를 반환한다. 최종 명령과 rollout은 `limit_velocity`를
   거쳐 도달 가능해야 한다.
2. `action_type`은 기존 다섯 값 중 하나를 쓴다. 새 값이면 `WarehouseEnv.__init__`·`reset`의
   `shield_action_counts`와 evaluator `episode_row` 열을 함께 늘린다(아니면 `KeyError`).
3. `WarehouseEnv.step` 분기에 새 `shield_strategy`를 추가한다. 지금은 `"scan"`이 아니면 모두
   `SafetyShield.apply`로 가고, 그 안에서 `"predictive"`가 아니면 legacy다.
4. ROS에서 쓰려면 LiDAR 점·odom만 입력으로 받아야 하고 `planner_core.py`와 bundle `safety` 기록을
   함께 바꾼다. FP/FN이 필요하면 env에서 정답 audit를 따로 계산한다(predictive `audit` 참고).

**새 randomization knob**

1. `SensorDynamicsModel`에 no-op 기본값 필드를 추가한다(자동으로 `_RANDOMIZABLE`, 정수면
   `INTEGER_KEYS`에도 추가). `env.yaml`의 `sensor_model`에 기본값, 필요하면
   `domain_randomization.ranges`에 범위를 넣는다.
2. env 적용 지점에서 knob이 꺼져 있으면 아무 일도 하지 않게 하고 identity 속성
   (`actuation_identity` 등)을 갱신한다. 꺼진 상태에서 `noise_rng`를 소비하면 이후 잡음 수열이 바뀌어
   golden 관측(`opticore_gazebo` σ 0.03)과 legacy parity가 깨진다.
3. `test_sensor_model.py`에 효과 테스트를 추가한다.

**resample 규칙 변경**: `RESAMPLE_RULE_VERSION`과 관측 계약 version을 올리고 golden을 재생성한다
(`tests/golden/make_observation_golden.py`). ROS core가 같은 함수를 쓰므로 양쪽이 동시에 바뀐다.

**rule baseline 변경**: BC 교사 dataset, residual base, legacy parity 기준값이 모두 바뀐다. residual
bundle은 base의 env·baseline 설정만 기록하고 코드는 기록하지 않으므로(`deploy/runtime.py`), 코드 변경은
기존 residual bundle의 동작도 바꾼다(golden parity 실패 가능, 추정).

## 8. 주의할 점과 알려진 한계

1. **rule baseline의 clearance 임계값이 footprint 안이다.** `clearance_stop` 0.12 <
   `robot_radius` 0.36이고 거리는 로봇 중심 기준이다. 접촉 시 목표 속도가 0.49/0.57 m/s다(§4.5).
   Shield OFF 동적 충돌 30건의 충돌 속도 0.41–1.38 m/s가 이와 맞다. 임계값을 `robot_radius` 기준으로
   바꾸면 legacy parity·BC 교사·residual base가 모두 바뀐다.
2. **그 밖의 의외 동작.** rule baseline braking이 local goal 거리를 써서 순항 상한이 1.285/1.483 m/s다
   (§4.5). Lite 투영은 전역 검색이라 접히는 조밀 경로에서 ROS와 다를 수 있다(§4.4). 모든 후보가
   위험할 때 predictive는 STOP 아닌 후보, scan은 STOP을 선호하고, predictive CPA는 현재 heading 직진을
   가정한다(§4.7, §4.8).
3. **legacy shield는 물리적으로 도달 불가능한 명령을 낸다.** SLOW·STOP이 `_limit`을 거치지 않아 한
   step에 0.75 m/s 감속(허용 0.075)이나 즉시 정지가 가능하다. horizon은 `int(0.7/0.05)` = 13 step,
   즉 0.65 s다(DECISIONS D-007의 "0.7초"와 다름). legacy 결과는 scan보다 낙관적이다.
4. **scan tracker가 정적 구조물에서 가짜 track을 만든다.** `static_filter_distance` 0.0(기본)에서
   정적 시나리오에도 `SCAN_TRACK` 위험이 잡힌다. Shield OFF selection 측정에서 `P-HALL` 382/1015,
   `WH-RANDOM` 212/1315, `S3` 143/806 step이었고, Shield ON에서는 개입으로 `P-HALL` 총 step이
   1015 → 1638, `WH-RANDOM` 1315 → 1728로 늘었다(모두 성공). `static_filter_distance: 0.3`이면
   `S3`·`P-HALL`·`WH-RANDOM`의 `SCAN_TRACK`이 0이 됐다. ROS core는 필터가 없다
   ([NEXT_STEPS.md](../NEXT_STEPS.md)의 `/map` 필터 항목).
5. **scan·legacy의 FP/FN은 항상 0이다.** protocol v2 결과의 `shield_false_positive_per_episode`,
   `shield_false_negative_per_episode`는 의미가 없다. `unsafe_without_shield`,
   `shield_ineffective_intervention`, `shield_actual_false_negative`, `shield_risk_counts`를 쓴다.
6. **정지 장애물 앞 교착.** 전방이 막히면 SLOW·EVADE가 모두 위반이라 scan shield는 매 tick STOP을
   고른다(§4.9). rule baseline은 전방 거리 0.12 미만에서만 회전하고 Lite는 재계획하지 않으므로 D3·D4가 전부
   SHIELD_SATURATION이 된다. ROS에서는 `EMERGENCY`가 A* 재계획을 부르지만(ARCHITECTURE §8) Lite에는
   대응 동작이 없어 Lite의 saturation 비율은 ROS보다 비관적이다(추정).
7. **전진 전용.** scan의 `reverse_enabled`가 false라 정면 접근(D1)에서 로봇이 멈춘 채 충돌한다.
   후진 허용은 ROADMAP G2 결정 사항이다. predictive는 반대로 기본 true다.
8. **`SHIELD_SATURATION` 판정**은 STOP *mode* step(감속 중 포함) 수로, TIMEOUT보다 먼저, `success`와 무관하게 한다(§4.11).
9. **`shield_strategy` 오타는 조용히 legacy가 된다**(`WarehouseEnv.step`, `SafetyShield.apply`).
10. **localization 오차는 shield에 닿지 않는다.** Lite scan shield·tracker는 참 pose(`self.robot`,
    `last_scan_pose`)를 쓰지만 ROS는 odometry pose를 쓴다.
11. **tracker 기하 편향.** 원형 장애물(반경 0.28)을 raw scan으로 보면 cluster 중심이 실제 중심보다
    로봇 쪽으로 0.22–0.25 m 치우친다(1–4 m 거리 측정). contract 모드(72점)는 3 m 이상에서 점이 2개뿐이라
    cluster가 생기지 않고, 반경도 0.18–0.19로 작게 잡힌다. 360° raw scan은 beam 순서상 ±π에서
    끊기므로 로봇 바로 뒤 물체가 두 cluster로 나뉠 수 있다(추정).
12. **`RuleBasedPlanner()` 기본 생성자는 `lite_default`를 쓴다.** `opticore_gazebo` 평가에서 인자 없이
    만들면 `w_max` 1.8로 정규화돼 실행 각속도가 의도의 1.5/1.8배가 되고 braking도 `decel_max` 1.5로
    계산된다. `cli/evaluate.py`, `cli/m1.py`, `cli/train.py`, `cli/demo.py`가 기본 생성자를 쓴다
    (protocol v1 용도). sweep·paired·PPO v2는 env config를 넘긴다.

Phase 0 시절의 한계(정면 접근 `D1`, 전진 전용)는
[KNOWN_LIMITATIONS.md](../../experiments/phase0_amr_lite/KNOWN_LIMITATIONS.md)에도 있다.
