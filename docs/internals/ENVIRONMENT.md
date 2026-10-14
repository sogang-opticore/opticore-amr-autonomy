# AMR-Lite 시뮬레이션 환경

이 문서는 AMR-Lite 학습 환경 `WarehouseEnv`와 그 주변(설정 해석, 동역학, 지도, ray casting,
시나리오 생성)이 **코드에서 실제로 어떻게 동작하는지** 설명한다. 관측·행동·bundle 계약은
[ARCHITECTURE.md](../ARCHITECTURE.md), 마일스톤과 gate는 [ROADMAP.md](../ROADMAP.md)를 따르며 여기서
다시 정의하지 않는다. Safety Shield, tracker, LiDAR 재표본화, 센서 randomization 내부는
[SENSORS_AND_SAFETY.md](SENSORS_AND_SAFETY.md)에서 다룬다.

## 1. 역할과 범위

`WarehouseEnv`는 ROS 없이 도는 결정적 2D differential-drive 주행 환경이다. Gymnasium과 같은
`reset(seed, options) → (obs, info)`, `step(action) → (obs, reward, terminated, truncated, info)`
시그니처를 갖지만 gymnasium에 의존하지 않는다. env 인스턴스 하나가 정적 지도, A* 전역 경로,
scripted 동적 장애물, LiDAR, shield 호출, reward, 실패 분류를 모두 소유한다.

| 구성 | 도입 | 근거 |
|---|---|---|
| 기본 루프, 직사각형 지도, `S0`–`S7`, `D0`–`D4`, `legacy` shield | Phase 0 | `experiments/phase0_amr_lite/DECISIONS.md` |
| Reward-v2(경로 arc-length progress, heading·turn·time 항), `CIRCLING` 분류 | Phase 0 | DECISIONS D-011, KNOWN_LIMITATIONS |
| `scenario_randomization`(S/D jitter), selection/test seed 분리 | M1 (추정) | `configs/evaluation_selection.yaml`, legacy parity의 `D1_rule_randomized` |
| robot profile, raw LiDAR, sensor·actuation model, vectorized ray cast, `scan` shield 연동 | M2 | ROADMAP M2 |
| 절차적 `P-*`/`PD-*`, `GridMap`, `WH-RANDOM`/`WHD-RANDOM`, 학습 seed namespace | M3 | ROADMAP M3, §4.1 |
| paired leg `WH-<set>-<NN>`, `WaypointObstacle` | M6 | ROADMAP M6 |

범위 밖: shield·tracker·재표본화·randomization 함수 내부(SENSORS_AND_SAFETY.md), 관측 벡터
구성(ARCHITECTURE §4), A* 내부(`packages/amr_lite/amr_lite/planning/astar.py`), 학습 루프(`amr_lite.rl`).

## 2. 파일 지도

본문에서 파일명만 쓴 경우 아래 경로를 가리킨다. 테스트는 `packages/amr_lite/tests/` 아래에 있다.

| 파일 | 책임 |
|---|---|
| `packages/amr_lite/amr_lite/config.py` | YAML 로드, `deep_merge`, robot profile 해석 |
| `packages/amr_lite/amr_lite/envs/warehouse_env.py` | `WarehouseEnv`, `Box` fallback, 프로세스 전역 A* 캐시 `_PATH_CACHE`(`envs/__init__.py`가 순환 import를 피하려 lazy import) |
| `packages/amr_lite/amr_lite/envs/dynamics.py` | `RobotState`, `limit_velocity`, `integrate` |
| `packages/amr_lite/amr_lite/envs/geometry.py` | `Rectangle`, `wrap_angle`, circle–rect clearance, scalar ray 교차, `bilinear_centres` |
| `packages/amr_lite/amr_lite/envs/map.py` | `WarehouseMap`(축 정렬 직사각형 지도) |
| `packages/amr_lite/amr_lite/envs/grid_map.py` | `GridMap`, `feature_transform`(정확한 EDT), PGM·ROS map 로더, `named_map` |
| `packages/amr_lite/amr_lite/envs/raycast.py` | `cast_rays`, `lidar_scan`, 기준 구현 `lidar_scan_reference` |
| `packages/amr_lite/amr_lite/envs/scenarios.py` | `Scenario`, `DynamicObstacle`, `WaypointObstacle`, `make_scenario`(S/D 정의와 라우팅), jitter |
| `packages/amr_lite/amr_lite/envs/procedural.py` | `P-*`/`PD-*` 생성기, `place_agents`, split 규칙 |
| `packages/amr_lite/amr_lite/envs/warehouse_scenarios.py` | `WH-RANDOM`, `WHD-RANDOM`, paired leg |
| `packages/amr_lite/amr_lite/envs/gymnasium_adapter.py` | `GymnasiumWarehouseEnv` |
| `packages/amr_lite/configs/env.yaml` | 환경 기본값 |
| `packages/amr_lite/configs/procedural.yaml` | 절차적 family와 split별 파라미터 범위 |
| `packages/amr_lite/configs/robot_profiles/*.yaml` | `lite_default`(빈 override), `opticore_gazebo` |
| `packages/amr_lite/configs/paired/warehouse_v1.yaml` | paired leg 8개와 Gazebo mover agent 4개(ROS `map` 좌표) |
| `packages/amr_lite/assets/maps/opticore_warehouse/` | `opticore-amr` 지도 사본(1207×840 cell, 0.05 m, origin (−3.18, −16.9)) |

## 3. 동작 흐름

### 3.1 설정 해석

```text
configs/env.yaml ─load_config("env")─► base            (resolve_env_config(base=...)로 교체 가능)
   ─deep_merge─► robot_profiles/<name>.yaml 의 overrides, robot_profile=<name> 기록
   ─deep_merge─► 실험 config의 environment_overrides   (ppo_v2.yaml, evaluation_*_v2.yaml, dagger.yaml)
   ─deep_merge─► curriculum stage의 environment_overrides (rl/vector_env.py EpisodeSlot.reset)
   ─► WarehouseEnv(config)
```

### 3.2 `step` 한 번 (policy_dt 0.05 s)

```text
action ─ shape≠(2,) 또는 비유한 → [0,0], invalid=True
  └─ action_delay_steps>0 → queue에 넣고 가장 오래된 action 실행 (초기 queue = [-1,0] 정지)
scale_action → (v_target, w_target)
shield 1회 (step 시작 시점 state, 가감속 제한 포함)
  ├─ shield_strategy == "scan" → _apply_scan_shield (최근 scan 점 + scan track)
  └─ 그 외 → SafetyShield.apply (legacy | predictive, 정답 지도·장애물)
for 5 × physics_dt(0.01):
    동적 장애물 update(elapsed, dt) → _actuate(v,w) → integrate → elapsed += dt
    _collides(next) → collision=True, 겹친 pose 유지, break
통계 갱신 → success 판정 → scan_history.append(_sensor_scan())  # scan_period hold
path_state(실제 pose) → reward 항 → terminated / truncated
_observation(perceived pose) → observation_delay queue → (obs, reward, terminated, truncated, info)
```

## 4. 핵심 구성요소 상세

### 4.1 설정 해석 (`config.py`)

- `deep_merge(base, overrides)`: 재귀 복사본. 양쪽이 dict일 때만 병합하고 list·scalar는 통째로 교체한다
  (예: `warehouse.path_length`). `load_config(name, overrides)`는 **최상위 값이 `None`인 override를 버린다**.
- `load_robot_profile(name)`: `{name, description, overrides}`. 파일이 없으면 사용 가능 목록과 함께 `FileNotFoundError`.
- `resolve_env_config(robot_profile, overrides, base)`: base → profile overrides → `robot_profile` 키에 이름 기록 →
  overrides 순서다. profile 인자가 없으면 base 라벨(`lite_default`)이 남고, overrides에 `robot_profile: X`를 넣으면
  라벨만 바뀌고 profile 값은 적용되지 않는다. `WarehouseEnv(config=None)`은 `load_config("env")`만 쓴다.

### 4.2 `WarehouseEnv.__init__`과 `reset`

- 생성자는 config를 deepcopy하고 shield·tracker를 만든 뒤 곧바로 `reset(seed=config["seed"])`을 호출한다.
  `reset(seed=None)`은 매번 `config["seed"]`(기본 7)로 재시드하므로 같은 episode가 반복된다.
- `options["scenario_id"]`는 이번 episode에만 쓰인다(`self.scenario_id`는 생성자 값 유지).
  `options["pose_perturbation"] = a`는 x, y, θ에 각각 `U(−a, a)`를 더한다(`learning/collect_teacher.py`가 0.08 사용).
- 전역 경로: `path_key = (instance_id, grid_resolution, robot_radius + obstacle_inflation, 직사각형 장애물
  좌표[, clearance_key(astar_clearance)])`. 생성기가 검증한 `reference_path`는 planner 설정이 같고 로봇이
  시작점 그대로일 때만 캐시에 넣는다. 캐시에 없으면 `astar(map, 로봇 xy, goal, grid_resolution, inflation,
  astar_clearance)`를 돌린다. 캐시는 프로세스 전역 LRU이며 상한은 `_PATH_CACHE_LIMIT = 4096`이다.
  A*는 **정적 지도만** 본다. 동적 장애물(정지한 `D3`, `static` agent 포함)은 경로에 반영되지 않는다.
- `sensor_model.perturbs_path`면 경로를 `perturb_path`로 조밀화·흔든다. timeout은 `episode_timeout_per_meter > 0`일 때
  `max(episode_timeout, episode_timeout_base + per_meter × 경로 길이)`(perturb 이후 길이)다.
- action 지연 queue는 `[-1, 0]`(정지), observation 지연 queue는 첫 관측으로 채운다. 첫 scan(`force=True`)으로 4 frame을 모두 채운다.

### 4.3 `step`: 시간, 구동, 충돌

- **시간**: 한 step은 `physics_substeps`(5) × `physics_dt`(0.01) = 0.05 s다. 코드는
  `physics_dt × physics_substeps == policy_dt`를 검사하지 않는다. `idle_time`, `total_abs_rotation`,
  shield 가감속 제한은 `policy_dt`를, 물리는 substep 합을 쓴다.
- **가감속 제한은 env가 아니라 shield 안에서** 적용된다. 세 전략 모두 PASS 명령을
  `dynamics.limit_velocity(dt=policy_dt)`로 제한해 돌려주며, `shield_enabled: false`여도 shield는
  호출되어 제한된 명령을 PASS로 낸다. 예외는 `legacy`의 SLOW(`v × 0.5`, `v × 0.2`)와 STOP(`(0, 0)`)이다(§8).
  `limit_velocity`는 `dv ≥ 0`이면 `accel_max`, 아니면 `decel_max`, 각속도는 `angular_accel_max`로 한 dt 변화를
  제한하고 `v ∈ [v_min, v_max]`로 자른다(`v_min`은 후진 후보에서만 음수).
- shield가 낸 `(v, w)`는 substep 동안 고정이다. `_actuate`는 sensor model이 identity면 그대로, 아니면
  `v_scale`, `w_scale`을 곱하고 `actuator_time_constant τ > 0`이면 실제 속도 `robot.v`에서
  `1 − exp(−dt/τ)` 비율로 다가가는 1차 지연을 substep마다 적용한다. 지연된 속도가 다음 step shield의
  측정 속도가 된다.
- `integrate`: `|w| < 1e-9`면 직선, 아니면 정확한 원호 적분이고 θ는 `wrap_angle`로 `[−π, π)`다.
- `_collides`: 반경 `r = robot_radius + collision_radius_margin`. 정적은 `map.collides`(clearance ≤ 0, 접촉도 충돌),
  동적은 중심거리 ≤ `r + r_obs`. 충돌하면 겹친 pose를 robot에 넣고 substep loop를 끊는다. 장애물이 substep마다 먼저 움직인다.
- 통계: `idle_time`은 `|v| < 0.03`인 step마다 `policy_dt`, `total_abs_rotation`은 `|w| × policy_dt`,
  `angular_oscillation`은 `|w| > 0.15`인 구간의 부호 반전 수다. `success`는 step 끝에서 `scenario.goal`까지 실제
  거리 ≤ `goal_tolerance`이고 충돌이 아닐 때다.
- scan은 물리 이후 갱신된다. `_scan_due`는 `scan_period ≤ policy_dt`이고 phase가 0이면 매 step, 아니면
  `_next_scan_time` 도달 시에만 새 scan을 찍고 그 사이는 같은 배열을 다시 넣는다(zero-order hold).
- 관측은 `_perceived_robot()`(localization bias·noise 적용 pose)로, reward·충돌·success는 실제 pose로 계산한다.

### 4.4 Reward

reward는 아래 항의 합이고 항별 값은 `info["reward_components"]`에 있다(`WarehouseEnv._reward_components`).

| 항 | config 키 | 기본값 | 계산 | 부호 |
|---|---|---:|---|---|
| `progress` | `reward_progress` | 10.0 | × (이전 경로 잔여 arc length − 현재) | 전진 +, 후퇴 − |
| `goal` | `reward_goal` | 20.0 | success인 step | + |
| `collision` | `reward_collision` | −10.0 | collision인 step | − |
| `near` | `reward_near` | −2.0 | × max(0, `safety_distance`(0.30) − clearance) | − |
| `smooth` | `reward_smooth` | −0.3 | × ‖a_t − a_{t−1}‖² | − |
| `path` | `reward_path` | −0.2 | × \|cross-track error\| | − |
| `idle` | `reward_idle` | −0.1 | \|v\| < 0.03이면 | − |
| `heading` | `reward_heading` | −0.05 | × \|경로 접선 − 로봇 heading\| | − |
| `turn` | `reward_turn` | −0.02 | × a_w² | − |
| `time` | `reward_time` | −0.01 | 매 step | − |

- 잔여 거리는 `planning/path_utils.py`의 `remaining_path_distance`이고, 투영은 경로 **전체**에서 가장 가까운 선분이다.
- `smooth`, `turn`은 env에 들어온 action(invalid면 0, 지연 queue 이전 값) 기준이며 첫 step의 이전 action은 `[0, 0]`이다.
- clearance는 `_minimum_clearance()` = min(정적 clearance(robot_radius), 동적 중심거리 − 반경 합)으로
  `collision_radius_margin`을 포함하지 않는다. `info["safety_cost"] = float(collision) + max(0, safety_distance − clearance)`는
  PPO v2 Lagrangian 비용과 평가 `mean_safety_cost`에 쓰인다.
- **v1과 v2**: 코드에는 위 형태 하나뿐이다. DECISIONS D-009/D-011에 따르면 Phase 0 최초 reward(v1)는 progress를
  Euclidean 목표거리 감소로 쟀고 heading·turn·time 항이 없었다. v1으로 돌아가는 설정 스위치는 없다.

### 4.5 종료와 `failure_type`

- `terminated = collision or success or invalid or not global_path`
- `truncated = elapsed ≥ episode_timeout and not terminated`. 기본 18 s는 360 step에서 truncation된다
  (실측 elapsed `18.000000000000014`). 평가 하네스의 `max_steps`(protocol v2: 2400)는 별도 상한이다.
- `_failure_type`은 아래 순서로 첫 조건을 고른다.

| 순서 | 값 | 조건 |
|---:|---|---|
| 1 | `NO_PATH` | 전역 경로가 비어 있음 |
| 2 | `INVALID_ACTION` | 이번 step action이 invalid |
| 3 | `COLLISION_DYNAMIC` / `COLLISION_STATIC` | 충돌, 최종 pose가 동적 장애물과 겹치면 DYNAMIC |
| 4 | `SHIELD_SATURATION` | STOP 횟수 > max(20, step_count // 2) |
| 5 | `OSCILLATION` | `angular_oscillation` > 12 |
| 6 | `CIRCLING` | `total_abs_rotation` > 4π |
| 7 | `STUCK` | `idle_time` > 0.6 × `episode_timeout` |
| 8 | `TIMEOUT` | elapsed ≥ `episode_timeout` |
| — | `""` | 해당 없음 |

`_failure_type`은 `_info`가 불릴 때마다(reset 포함, 매 step) 계산되며 success나 종료 여부를 보지 않는다(§8).

### 4.6 `info`

| 필드 | 내용 |
|---|---|
| `seed` / `sensor_model` | reset에서만 값(step의 `seed`는 `None`) / `SensorDynamicsModel.as_dict()` |
| `scenario_id`, `scenario_instance_id`, `robot_pose`, `goal_distance` | §4.11, 실제 pose `(x, y, θ)`, 목표점까지 거리 |
| `success`, `collision`, `failure_type`, `reward_components` | §4.5, §4.4(reset에서는 `{}`) |
| `minimum_clearance`, `minimum_clearance_episode` | 현재 값, episode 최솟값(reset 시 `inf`) |
| `elapsed_time`, `path_length`, `shortest_path_length`, `path_length_ratio` | 주행 거리 / 전역 경로 길이(경로 없으면 `inf`) |
| `mean_cross_track_error`, `max_cross_track_error` | step별 \|cross-track\| 통계(config의 정규화 상수와 이름만 같다) |
| `idle_time`, `angular_oscillation`, `total_abs_rotation`, `path_remaining_distance` | §4.3 통계 |
| `shield_counts`, `shield_action_counts` | `PASS`/`CLAMP`/`STOP`, `PASS`/`SLOW`/`EVADE`/`REVERSE`/`STOP` step 수 |
| `shield_actual_false_negative`, `shield_ineffective_intervention` | shield ON 충돌 step의 판단이 PASS였음 / 개입했는데도 충돌 |
| 그 밖의 `shield_*`, `unsafe_without_shield`, `tracking_speed_error_*`, `minimum_predicted_ttc` | shield 진단(SENSORS_AND_SAFETY.md) |
| `shield_mode`, `safety_cost` | step에서만 |

`record_shield_trace: true`면 step마다 `env.shield_trace`에 전후 pose·속도·shield 판단이 쌓인다.

### 4.7 정적 지도

두 지도 클래스는 같은 duck-typed 인터페이스(`clearance`, `collides`, `occupied`, `planning_clearance`,
`ray_distances`, `world_to_grid`, `grid_to_world`, `obstacles`, `width`, `height`)를 구현한다.

| 항목 | `WarehouseMap` | `GridMap` |
|---|---|---|
| 표현 | `Rectangle` 목록 + 지도 밖 0.05 m 두께 경계벽 4개 | bool 점유 격자(행 0 = 아래), `map_origin`으로 ROS `map` 좌표 변환 |
| 기본 크기 | 12 × 8 m | `opticore_warehouse`: 60.35 × 42.0 m |
| `clearance` | 모든 직사각형까지 원–사각형 거리의 최솟값 − r (정확) | EDT의 최근접 점유 cell(3×3 이웃)까지 거리 − r. `max_distance`(4.0 m)에서 포화, 지도 밖은 `−r − resolution` |
| `occupied`(A*용) | `collides`와 같음 | cell 중심 거리 − 0.5 cell ≤ inflation(오차 반 cell 미만) |
| `planning_clearance` | 5 cm raster bilinear, `(w, h, len, id(obstacles))`로 캐시 | cell 중심 거리장 bilinear |
| `ray_distances` | slab 방식 vectorized, scalar와 bit 동일 | 거리장 sphere tracing, 근사(§4.8) |

- `feature_transform`은 열마다 세로 최근접 점유 cell을 구한 뒤 가로 shift별 최솟값을 취해 `max_cells` 미만에서
  정확한 Euclidean 거리와 최근접 cell index를 내고, 점유 격자 fingerprint로 `_FEATURE_CACHE`에 캐시된다.
- `load_ros_map`은 trinary 규칙(`occupied_thresh` 0.65, `free_thresh` 0.25)에 더해 미탐색 값 205를 점유로 보고
  (선반 내부·외벽 바깥) 이미지를 `flipud`한다. `named_map(name)`은 `assets/maps/<name>/`의 첫 YAML을 읽는다.

### 4.8 Ray casting

- `lidar_scan(state, map, dynamics, rays=72, fov=270°, max_range=8.0)`(`contract` 모델)은 상대각
  `linspace(−fov/2, fov/2, rays)`로 `cast_rays`를 부르고 float32로 돌려준다. raw 모델은 단위벡터를 직접 넘긴다.
- `cast_rays`: 정적은 `map.ray_distances`, 동적 원은 `circle_ray_distances`의 최솟값을 `[0, max_range]`로 자르고
  `+ 0.0`으로 `−0.0`을 없앤다.
- **bit 동일 보장**은 `WarehouseMap`에만 해당한다. `ray_directions`가 ray마다 `math.cos`/`math.sin`을 쓰고,
  `WarehouseMap.ray_distances`가 `geometry.ray_rectangle_distance`의 연산 순서를 그대로 따른다. 기준 구현
  `lidar_scan_reference`와 바이트 단위로 비교하는 테스트가 있다.
- `GridMap.ray_distances`는 cell 중심 거리장에서 `max(d − √2·res, 0.25·res)`만큼 전진하다가 점유 cell에
  들어가면 멈춘다. 최대 `max_ray_iterations`(64)회 안에 수렴하지 못한 ray는 그때까지 간 **더 짧은**
  거리를 보고한다. 지도 밖은 padding 0으로 막혀 있다.

### 4.9 동적 장애물

모든 동적 장애물은 원이고 **scripted·비반응형**이다. 로봇을 피하거나 멈추지 않고, 벽과 다른 장애물을
통과하며 지도 밖으로도 나간다(`D0` 장애물은 t = 10 s에 y = 9.0, 지도 높이 8).

- `DynamicObstacle.update(time, dt)`: `start_delay ≤ time < stop_time` 동안 `(vx, vy)` 등속, `patrol_distance`가
  있으면 그만큼 간 뒤 속도를 뒤집는다. `motion_type`은 라벨일 뿐 동작에 영향이 없다(digest와 테스트에서만 쓰임).
- `WaypointObstacle`: `opticore-amr`의 `dynamic_obstacle_mover.py`를 모사한다. waypoint 사이를 smoothstep으로 보간하고
  도착마다 `pause_range`에서 뽑은 시간만큼 멈춘 뒤 순환한다. `vx`, `vy`는 shield용 유한차분 속도이고 `stop_time`은 무시한다.

### 4.10 시나리오 카탈로그

`make_scenario(id, seed, randomization, sources)`는 `P-`/`PD-` → `procedural`, `WH`로 시작 → `warehouse_scenarios`,
그 외 → 고정 S/D 정의로 보낸다. S/D 사전에 없는 id는 **오류 없이 `S0` 지도**를 쓴다(§8).

고정 시나리오(12 × 8 m 직사각형 지도, 기본 seed 0의 A* 길이):

| ID | 배치 | 시험 내용 | 경로 |
|---|---|---|---:|
| `S0` | 장애물 없음, (1,4,0) → (11,4) | 빈 직선 | 10.00 |
| `S1` | 벽 (5.0,0)–(5.5,5.7), (1,2) → (9.5,6.5) | 90° 코너 | 10.34 |
| `S2` | 벽 y≤3.25, y≥4.75 (x 3–9) | 폭 1.5 m 좁은 통로 | 10.00 |
| `S3` | 상자 (5.3,3.2)–(6.2,4.8) | 직선 위 정적 장애물 우회 | 10.43 |
| `S4` | 상자 (5.0,4.55)–(7.0,6.2) | 장애물 옆 통과(경로상 clearance 0.19 m < `safety_distance`) | 10.00 |
| `S5` | 막대 (4,5)–(8,5.6), 시작 (1,6.5,−0.4) | 경로에서 벗어난 시작 | 10.69 |
| `S6` | 엇갈린 벽 (4.8,0)–(5.5,5.9), (7.5,2.1)–(8.2,8.0) | 부분 폐쇄·S자 통로 | 15.18 |
| `S7` | 시작점을 벽과 경계로 둘러쌈 | no-path. 첫 step에 `NO_PATH`로 종료 | — |
| `D0` | `S0` 지도 + (6,2), v=(0,0.7) | 횡단 | 10.00 |
| `D1` | (9,4), v=(−0.45,0) | 정면 접근 | 10.00 |
| `D2` | (4,4), v=(0.25,0) | 같은 방향 저속 선행 | 10.00 |
| `D3` | (6,4) 정지 | 경로 위 정지 장애물(A*는 모름) | 10.00 |
| `D4` | (6,2), v=(0,0.7), `stop_time` 3.0 | 횡단 후 경로 위(y≈4.1) 정지 | 10.00 |

`D*` 장애물 반경은 모두 0.28이다. 절차적·창고 시나리오:

| ID | 내용 |
|---|---|
| `P-AISLE` | 가로 선반 열(`shelf_depth`)과 통로(`aisle_width`), 열마다 `gaps_per_row`개의 틈. 시작·목표는 통로 lane 위이고 목표 lane은 70% 확률로 무작위, 30%는 시작 lane |
| `P-HALL` | 열린 hall에 `pallets`개 직사각형(`pallet_size`), 시작·목표에서 1.0 m 이상 |
| `P-PINCH` | 폭 `corridor_width` 통로가 x 35–65% 지점에서 `jog`만큼 꺾이고, `pinches`개 돌출이 폭을 `pinch_width`로 좁힘 |
| `P-JUNCTION` | X/T/L 교차로(0–3회 90° 회전), 서로 다른 두 arm 끝이 시작·목표, arm 안 `clutter`개 0.4–0.7 m 상자 |
| `P-MIXED` | seed마다 `families` 중 하나 |
| `PD-<FAMILY>` | 같은 layout 규칙 + `place_agents`(아래). `P-X`와 `PD-X`는 같은 seed라도 layout이 다르다(id가 RNG에 들어감) |
| `WH-RANDOM` | 창고 지도 free cell(clearance > `warehouse.min_clearance` 0.7)에서 start/goal, 직선거리 ∈ [6, 25], A* 길이 ∈ [6, 37.5] |
| `WHD-RANDOM` | `WH-RANDOM` + 활성 split의 `dynamic` 규칙 agent |
| `WH-<set>-<NN>` | `configs/paired/<set>.yaml`의 leg NN. leg k 시작 = leg k−1 목표, heading = 직전 경로 마지막 선분. agent는 `WaypointObstacle` |

절차적 공통: 16 × 12 m 지도, 50% 확률 좌우 대칭, 시작·목표가 inflation + 0.05 안에 있으면 버림, A* 길이 ∈
[5.0, 22.0]이 될 때까지 최대 40회 시도 후 `RuntimeError`. 시작 heading = 첫 경로 선분 방향 + `U(−0.6, 0.6)`.

`place_agents`: agent 수는 `count`에서 정수 추출, agent마다 최대 20회 시도(실패하면 조용히 빠짐). 속도·반경은 split 범위,
로봇 예상 속도는 `nominal_robot_speed` [0.8, 1.2]이고, 생성 지점 정적 clearance < 0.05 또는 시작점 2.0 m 이내면 거절한다.

| pattern | `motion_type` | 배치와 움직임 |
|---|---|---|
| `cross` | `cross` | 경로 35–80% 지점의 법선 방향 1.5–3.0 m 옆에서 경로를 가로지름. `start_delay = max(0, arc/robot_speed − offset/speed + U(−1,1))`로 로봇 도착에 맞춤 |
| `stop` | `stop` | `cross`와 같고 `stop_time = delay + (offset + U(−0.4, 0.3))/speed`로 경로 근처 정지 |
| `patrol` | `patrol` | `cross` 위치, `start_delay U(0,2)`, 2×offset마다 왕복 |
| `head_on` | `approach` | 경로 55–95% 지점(횡 ±0.2 m)에서 그 지점 접선의 반대 방향 직선, `start_delay U(0,1.5)` |
| `follow` | `same_direction` | 경로 20–45% 지점에서 접선 방향 `U(0.15, 0.45)` m/s(split `speed`를 쓰지 않음) |
| `static` | `stopped` | 경로 35–80% 지점(횡 ±0.35 m) 정지. 그 원을 정사각형으로 넣은 A*가 성공해야 채택 |

### 4.11 Randomization, split, seed, `instance_id`

- **S/D jitter**(`scenario_randomization.enabled`): RNG = `SeedSequence([seed, crc32(id)])`. 시작 xy ±`start_xy`,
  θ ±`start_theta`, 목표 ±`goal_xy`(반경 0.40 원이 충돌하지 않을 때까지 최대 32회, 실패 시 원래 값), 동적 장애물 위치
  ±`dynamic_position`, 속도 × max(0.1, 1 ± `dynamic_speed_fraction`), 반경 ±`dynamic_radius`(최소 0.05),
  `start_delay += U(0, dynamic_start_delay)`, `stop_time += U(±dynamic_start_delay)`. 끄면 **seed와 무관하게 같은
  시나리오**이고, 절차적·창고 시나리오는 이 설정을 무시한다.
- **절차적 split**: `procedural.split`(train | validation | test)이 `procedural.yaml`의 파라미터 범위를 고른다.
  RNG = `SeedSequence([seed mod 2³², crc32(id), crc32(split)])`라 같은 seed도 split마다 다른 instance다.
- **창고 시나리오**: `WH-RANDOM`/`WHD-RANDOM` RNG = `SeedSequence([seed mod 2³², crc32(id), crc32(map)])`로 split이
  들어가지 않는다. `WHD`의 agent 규칙만 split을 따른다. paired leg은 start/goal이 고정이고 agent pause만
  `crc32("<name>:<seed>")`로 달라진다.
- **seed namespace**: 학습 episode seed = `1_000_000 + SeedSequence([training_seed, slot, counter])` 파생값
  (`rl/vector_env.py`의 `training_episode_seed`), selection = 20000부터 5개, test = 30000부터 20개
  (`configs/evaluation_selection_v2.yaml`, `evaluation_test_v2.yaml`).
- **`instance_id`**: 같은 id·seed·설정이면 같은 instance임을 나타내는 문자열이다. `_PATH_CACHE` key와
  평가 `episodes.csv`의 `scenario_instance_id`에 쓰인다.

| 계열 | 형식 | digest 입력 |
|---|---|---|
| S/D | `<id>-<seed 또는 0>-<sha16>` (jitter off면 0) | id, seed, randomized, start, goal, 장애물 |
| P/PD | `<id>-<split>-<seed>-<sha16>` | id, family, split, seed, 장애물, start, goal, agent |
| WH(D)-RANDOM | `<id>-<map>-<seed>-<sha16>` | id, map, seed, start, goal, agent |
| paired leg | `<id>-<seed>-<sha16>` | id, seed, start, goal, agent |

### 4.12 Gymnasium adapter

`GymnasiumWarehouseEnv`는 `gymnasium.Env`를 상속해 `WarehouseEnv`를 감싸고 `spaces.Box(−1, 1, (2,))`,
`spaces.Box(−1, 1, (296,))`를 선언한다. gymnasium은 optional 의존성(`pyproject.toml`의 `reference-rl`)이라 없으면 생성 시
`RuntimeError`다. seed를 그대로 넘기므로 seed 없는 reset은 config seed로 돌아간다. 저장소 안에 사용처와 테스트가 없다.

## 5. 설정

`configs/env.yaml` 기본값과 읽는 곳(reward 키는 §4.4).

| 키 | 기본값 | 읽는 곳 |
|---|---|---|
| `seed`, `scenario` | 7, `S0` | `WarehouseEnv.__init__`/`reset` |
| `robot_radius`, `obstacle_inflation` | 0.36, 0.04 | 충돌·clearance, A* inflation(합 0.40), 생성기 |
| `safety_distance` | 0.30 | `near` reward, `safety_cost` |
| `v_max`, `w_max`, `accel_max`, `decel_max`, `angular_accel_max` | 1.5, 1.8, 1.0, 1.5, 3.0 | `scale_action`, shield의 `limit_velocity` |
| `policy_dt`, `physics_dt`, `physics_substeps` | 0.05, 0.01, 5 | `step` |
| `lidar_rays`, `lidar_fov_deg`, `lidar_max_range` | 72, 270.0, 8.0 | `_scan`, `ContractScanSpec`, 관측 정규화 |
| `history_frames` | 4 | **읽는 코드 없음**(4는 `scan_history` maxlen과 `HISTORY_FRAMES` 상수로 고정) |
| `history_mode`, `lookahead_distance`, `goal_distance_scale`, `max_cross_track_error` | `stack`, 0.65, 8.0, 2.0 | `_observation`, `path_state`, `build_observation` |
| `goal_tolerance`, `episode_timeout`, `episode_timeout_per_meter`, `episode_timeout_base` | 0.25, 18.0, 0.0, 6.0 | success, `reset` |
| `grid_resolution` | 0.20 | A*(env, 생성기) |
| `astar_clearance` | `enabled: false`, `preferred_clearance` 0.50, `clearance_cost_weight` 9.0, `wall_avoid_clearance` 0.55, `wall_avoid_cost_weight` 3.0, `wall_avoid_min_margin` 0.05, `simplify_min_clearance` 0.45 | `astar`, path cache key |
| `scenario_randomization` | `enabled: false`, `start_xy` 0.12, `start_theta` 0.10, `goal_xy` 0.12, `dynamic_position` 0.25, `dynamic_speed_fraction` 0.15, `dynamic_start_delay` 0.40, `dynamic_radius` 0.03 | `make_scenario` |
| `procedural` | `spec: procedural`, `split: train` | `make_procedural_scenario`, `WHD-RANDOM` |
| `warehouse` | `map: opticore_warehouse`, `min_clearance` 0.7, `path_length` [6.0, 25.0], `max_attempts` 40 | `_random_pair` |
| `shield_enabled`, `shield_strategy`, `record_shield_trace` | true, `legacy`, false | `step` |
| `shield_*`, `scan_shield`, `scan_tracker`, `lidar_model`, `sensor_model`, `domain_randomization` | — | SENSORS_AND_SAFETY.md |
| `robot_profile` | `lite_default` | 라벨(체크포인트·bundle 기록) |

`configs/procedural.yaml`: `map_width` 16.0, `map_height` 12.0, `min_path_length` 5.0, `max_path_length` 22.0,
`max_attempts` 40, `nominal_robot_speed` [0.8, 1.2], `families` [AISLE, HALL, PINCH, JUNCTION]. split 범위:

| family | 파라미터 | train | validation | test |
|---|---|---|---|---|
| aisle | `aisle_width` / `gap_width` | [1.8, 2.6] | [1.7, 2.8] | [1.5, 1.8] |
| aisle | `shelf_depth` / `gaps_per_row` | [0.8, 1.2] / [1, 2] | [0.8, 1.3] / [1, 2] | [1.0, 1.4] / [1, 1] |
| hall | `pallets` / `pallet_size` | [5, 10] / [0.5, 1.2] | [4, 12] / [0.5, 1.3] | [10, 14] / [0.6, 1.4] |
| pinch | `corridor_width` / `pinch_width` | [2.4, 3.2] / [1.3, 1.8] | [2.3, 3.4] / [1.25, 1.9] | [2.0, 2.4] / [1.1, 1.3] |
| pinch | `pinches` / `jog` | [1, 2] / [0.0, 0.8] | [1, 3] / [0.0, 1.0] | [2, 3] / [0.6, 1.2] |
| junction | `corridor_width` / `clutter` | [2.0, 2.8] / [0, 2] | [1.9, 3.0] / [0, 3] | [1.7, 2.0] / [1, 3] |
| dynamic | `count` / `speed` / `radius` | [1, 2] / [0.2, 0.7] / [0.22, 0.32] | [1, 2] / [0.2, 0.8] / [0.22, 0.34] | [1, 3] / [0.3, 0.9] / [0.22, 0.36] |
| dynamic | `patterns` | cross, head_on, follow, stop, static | 같음 | cross, head_on, patrol, stop |

`shapes`는 세 split 모두 [X, T, L]이다. `opticore_gazebo` profile이 바꾸는 환경 값: `w_max` 1.5, `decel_max` 2.0,
`angular_accel_max` 1.5, `lidar_model.type: raw`, `sensor_model.scan_period` 0.1, `range_noise_std` 0.03,
`astar_clearance.enabled: true`, `episode_timeout_per_meter` 2.0(나머지 `robot_radius` 0.36, `v_max` 1.5,
`accel_max` 1.0, `policy_dt` 0.05는 기본값과 같다). paired set `warehouse_v1`의 leg 경로 길이는 11.3–42.3 m,
timeout은 28.6–90.5 s다.

## 6. 불변식과 검증

| 불변식 | 테스트 |
|---|---|
| 기본 설정 rollout(step 수, failure, shield 횟수, pose, reward 합, 관측 꼬리)이 변경 전과 같음 | `test_legacy_parity.py::test_default_environment_reproduces_pre_change_rollouts`(S3 legacy, D0 predictive+잡음, D1 jitter, S1 shield off) |
| 같은 seed → 같은 관측·reward·pose, jitter on에서 seed별 instance 구분 | `test_reproducibility.py::test_deterministic_reset_and_step`, `::test_randomized_scenario_same_seed_replays_and_different_seed_changes` |
| reward 항 집합, rule baseline의 `S0`·`S3` 성공 | `test_environment.py`의 두 테스트 |
| 접촉도 충돌·얇은 벽, substep이 고속 tunneling을 막음 | `test_collision.py::test_overlap_corner_boundary_and_thin_wall`, `::test_physics_substeps_prevent_high_speed_tunneling` |
| 원호 적분, 각도 wrap, 후진 한계 | `test_dynamics.py` 4개 |
| vectorized ray cast = scalar 기준(바이트 단위, 72·720 ray) | `test_raycast.py::test_vectorized_cast_is_bit_identical_to_reference` |
| EDT 정확성, grid clearance 오차 반 cell 이내 | `test_grid_map.py::test_feature_transform_is_exact_below_saturation`, `::test_clearance_matches_rectangle_map_within_half_a_cell` |
| grid ray cast 근사(중앙 오차 < 0.06 m, 0.1 m 초과 누수 < 0.2%) | `test_grid_map.py::test_grid_ray_cast_agrees_with_rectangle_ray_cast` |
| PGM·ROS map 의미, 지도 SHA-256 고정 | `test_grid_map.py`의 나머지 3개 |
| 절차적 instance 재현성·A* 해·길이 범위(train/test × 5 id) | `test_procedural.py::test_procedural_instances_are_reproducible_and_solvable` |
| split별 규칙과 agent pattern, 미지 family/split 거부 | `test_procedural.py::test_splits_use_different_generation_rules`, `::test_dynamic_agents_follow_the_split_patterns`, `::test_unknown_family_or_split_is_rejected` |
| env가 생성기 경로를 쓰고 timeout을 늘림, patrol 반전, waypoint 보간·정지 | `test_procedural.py`의 마지막 세 테스트 |
| paired leg 연결, `WH-RANDOM` 재현성, Gazebo profile 실행 | `test_warehouse_scenarios.py` 처음 세 테스트 |
| profile merge가 지정 키만 바꿈, scan hold, 지연, actuation lag, localization 분리 | `test_sensor_model.py::test_robot_profile_overrides_only_listed_keys`와 해당 테스트들 |
| `S7` → `NO_PATH` | `test_astar.py::test_no_path_is_classified` |
| sync·subprocess vector env 동일, 학습 seed ≥ 1,000,000 | `test_vector_env.py::test_subprocess_and_sync_vector_envs_are_identical`, `::test_curriculum_stages_and_training_seed_namespace` |
| golden observation | `test_observation_golden.py::test_lite_reproduces_golden_observations` |

테스트되지 않는 것:

- `_failure_type`의 `SHIELD_SATURATION`, `OSCILLATION`, `CIRCLING`, `STUCK`, `TIMEOUT`, `INVALID_ACTION`,
  `COLLISION_STATIC` 판정과 truncation 시점. reward 항별 값(합계만 legacy parity로 고정).
- `_PATH_CACHE` 동작(특히 `pose_perturbation`과의 상호작용), `GridMap.ray_distances`의 수렴 실패 빈도.
- S/D 사전에 없는 id의 처리, `history_mode: newest`의 env 경로, `render`/`save_render`.
- `GymnasiumWarehouseEnv` 전체(현재 개발 환경에 gymnasium이 없어 생성하면 `RuntimeError`).
- `WHD-RANDOM` agent 배치의 타당성(비어 있지 않은지만 확인), 동적 시나리오의 회피 가능성.

## 7. 확장 방법 (claude 제안사항)

**고정 시나리오 추가**: `scenarios.make_scenario`의 `specs`에 `(장애물, 시작, 목표, solvable)`을 넣고, 동적이면
`if scenario_id == ...` 분기에 `DynamicObstacle`을 만든다. 기존 id의 값은 바꾸지 않는다(legacy parity). 평가
목록에 넣으면 protocol 파일의 `scenarios`와 버전을 함께 바꾼다.

**절차적 family 추가**: `procedural.py`에 `_name(rng, width, height, p) -> (obstacles, start, goal)`을 쓰고
`GENERATORS`에 대문자 키로 등록한 뒤, `procedural.yaml`의 **세 split 모두**에 소문자 키로 범위를 넣는다(없으면
생성 시 `KeyError`). `families`에 넣으면 `P-MIXED`/`PD-MIXED`의 `rng.choice` 결과가 모든 seed에서 바뀌므로 protocol
버전을 올린다(`procedural.yaml` 머리 주석 규칙). `test_procedural.py`의 parametrize 목록에도 추가한다.

**agent pattern 추가**: `_make_agent`에 분기를 추가하고 split의 `patterns`에 넣는다.
`test_dynamic_agents_follow_the_split_patterns`의 `allowed` 사상(pattern → `motion_type`)도 갱신한다.

**reward 항 추가**: `env.yaml`에 기본값 0의 키를 두고 `_reward_components`에 항을 추가한다. 0이면 합계가 그대로라
legacy parity가 유지된다. `test_environment.py`가 항 이름 집합을 정확히 비교하므로 함께 고친다.

**robot profile 추가**: `configs/robot_profiles/<name>.yaml`에 `name`, `description`, `overrides`(env.yaml override, dict
병합·list 교체)를 쓴다. ROS 노드는 bundle profile과 자기 파라미터가 다르면 주행 권한을 주지 않으므로(ARCHITECTURE §6)
`integrations/ros2/amr_learned_planner` 파라미터도 같이 맞춘다.

**지도·paired set 추가**: `assets/maps/<name>/`에 YAML+PGM을 새 디렉터리로 넣고(기존 파일 수정 금지) SHA-256 고정
테스트를 추가한다. paired set은 `configs/paired/<set>.yaml`에 ROS `map` 좌표로 `start`, `legs`, `agents`를 쓴다.

## 8. 주의할 점과 알려진 한계

아래 실측은 `/tmp`의 일회성 스크립트로 이 브랜치 코드에서 얻었다(저장소에는 남기지 않았다).

1. **성공한 episode에도 `failure_type`이 붙는다.** `_failure_type`은 success를 보지 않는다. rule baseline
   (`opticore_gazebo`, scan shield, test split) 30 episode 중 성공 2건이 `OSCILLATION`이었다(`P-JUNCTION` seed
   30003, `WH-warehouse_v1-02` seed 30005). `rl/evaluation.py`의 `aggregate`는 `failure_type`을 먼저 보고,
   `experiments/gate.py`는 `configs/gates/g4.yaml`의 `CIRCLING`/`OSCILLATION`/`STUCK` 비율(≤ 0.02)을 `failure_type`으로
   세므로 성공이 실패 행동으로 집계될 수 있다.
2. **path cache key에 시작 pose가 없다.** `pose_perturbation`으로 시작점이 바뀌어도 먼저 실행된 episode의 경로를
   재사용한다. `S3`를 seed 5(perturbation 0.08), perturbation 없이, seed 6 순으로 reset하자 경로 시작점이 모두
   (1.0488, 4.0493)이었다. teacher 수집 경로가 실행 순서에 따라 달라진다(오프셋은 수 cm).
3. **`legacy` shield의 SLOW·STOP은 감속 한계를 무시한다.** `SafetyShield._apply_legacy`가 `v × scale`과 `(0, 0)`을
   다시 제한하지 않는다. `D1` jitter seed 20003에서 한 step에 1.28 → 0.64 m/s(12.8 m/s², `decel_max` 1.5)가
   관측됐다. Phase 0/M1(기본 `legacy`) 결과에 해당하며 `scan`·`predictive`는 제한된다.
4. **`GridMap` ray cast의 가짜 근거리 반환.** `max_ray_iterations` 64 안에 수렴하지 못한 ray가 짧은 거리를 낸다.
   창고 지도 무작위 400 pose(`opticore_gazebo` raw 모델)에서 contract scan의 6.75%가 5000회 기준보다 5 cm 이상
   짧은 bin을 가졌고 최악은 6.04 m 짧았다. 관측과 scan shield에 가짜 장애물로 들어간다(영향 크기는 추정).
5. **알 수 없는 S/D id는 조용히 `S0` 지도가 된다.** `"d0"`, `"S9"`도 오류 없이 빈 지도로 만들어진다. 평가 config
   오타가 드러나지 않는다.
6. **동적 시나리오는 회피 가능성을 보장하지 않는다.** agent는 비반응형이고 행동 계약상 후진이 없다. `PD-*` test
   split 40 episode(rule baseline, scan shield)의 `COLLISION_DYNAMIC` 18건 중 12건은 로봇 속도 < 0.05 m/s에서
   일어났다. 실패 분류는 책임 소재를 구분하지 않는다.
7. **split은 겹치지 않는 영역이 아니다.** `procedural.yaml` 주석은 test를 "disjoint extremes"라 하지만
   `shelf_depth`(test [1.0, 1.4], train [0.8, 1.2]), `pallet_size`(test [0.6, 1.4], train [0.5, 1.2]), `aisle_width`
   (test [1.5, 1.8], validation [1.7, 2.8]) 등이 겹친다. 테스트는 `aisle_width` 경계 하나만 본다. `WH-RANDOM`은
   split과 무관(같은 seed면 train/test 설정에서 같은 `instance_id`)하고 학습 curriculum·paired leg이 같은 지도를
   쓰므로 창고 test는 새 seed일 뿐 새 지도가 아니다. S/D도 layout은 같고 jitter만 다르다.
8. **seed 없는 `reset()`은 같은 episode를 반복한다.** Gymnasium 관례와 달라 외부 RL 라이브러리를 adapter로
   붙이면 매 episode가 config seed 7로 재현된다.
9. **invalid action은 정지가 아니다.** `[0, 0]`으로 바뀌어 `v_target = 0.75 m/s`로 한 step 실행된 뒤 종료된다.
10. **progress 투영은 전역 최근접 선분이다.** ROS core는 이전 투영 주변만 찾는다(ARCHITECTURE §4.3). 경로가
    접히는 곳(aisle U턴 등)에서 progress reward가 점프할 수 있다(추정).
11. **검증되지 않는 설정.** `physics_dt × physics_substeps ≠ policy_dt`, `lidar_rays` 변경(관측 296 계약 파괴),
    무시되는 `history_frames`를 막는 검사가 없다. jitter 유효성 반경 0.40은 `robot_radius`와 무관하게 하드코딩이다.
    truncation은 `elapsed`의 부동소수 누적에 의존한다(기본값에서 360 step 확인).
12. **paired leg의 agent는 leg마다 첫 waypoint에서 다시 시작한다.** Gazebo mover는 leg 사이에도 움직이므로
    같은 leg라도 agent 위상이 다르다(추정).
13. **문서 불일치.** `packages/amr_lite/README.md`는 `D3`을 "통로 정지"라 하지만 `D3`은 빈 `S0` 지도다.
    `experiments/phase0_amr_lite/KNOWN_LIMITATIONS.md`는 LiDAR 잡음·mount offset·actuator delay·frame-stack
    ablation·병렬 env가 없다고 적지만 M2–M3에서 구현됐다(Phase 0 시점 기록).
