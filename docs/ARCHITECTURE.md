# Learned Local Planner Architecture

이 문서는 학습형 local planner가 AMR-Lite 학습 환경과 `opticore-amr` ROS2 스택에서
**같은 관측, 같은 행동, 같은 안전 계층**으로 동작하기 위한 계약을 정의한다.
마일스톤과 gate는 [ROADMAP.md](ROADMAP.md)를 따른다.

## 1. 용어

| 용어 | 의미 |
|---|---|
| 기존 planner, DWA | `opticore-amr/ros2_ws/src/amr_navigation/amr_navigation/dwa_node.py`. Pure Pursuit + 적응 속도 + 상태 머신으로 된 rule-based hybrid local planner |
| `lite_rule_baseline` | AMR-Lite의 축소 Pure Pursuit. 기존 planner와 동등하지 않으며 Lite 비교 기준선이자 BC 교사, residual 정책의 base |
| 학습 정책 | 관측을 받아 정규화된 `[a_v, a_w]`를 제안하는 신경망 |
| 학습 planner | 학습 정책 + 명령 변환 + Safety Shield + 가감속 제한 + 상태 머신 |
| robot profile | 학습·평가·배포가 공유하는 로봇 속도·가감속·LiDAR 사양 묶음 |
| policy bundle | ROS 배포용으로 export한 정책 가중치, 계약, profile, provenance 묶음 |

## 2. 런타임 스택

```text
                   ┌──────────────────────── opticore-amr (변경 없음) ───────────────────────┐
/goal_pose ──────► │ astar_planner ──/global_path──┬──────────────────────────────┐            │
                   │      ▲                       │                              │            │
                   │      │ /dwa/status           ▼                              ▼            │
                   │      │              dwa_planner (fallback)          (같은 입력 토픽)       │
                   └──────┼──────────────── │ cmd_vel_topic=dwa_fallback/cmd_vel ─┼────────────┘
                          │                 │ /dwa/status → dwa_fallback/status  │
                          │                 ▼                                    ▼
                          │        ┌─────────────────┐                ┌────────────────────┐
                          └────────┤   cmd_arbiter   │◄───────────────┤  learned_planner   │
                                   │ mode: dwa |     │  learned cmd,  │ scan→관측→정책→     │
                     /cmd_vel ◄────┤ shadow | learned│  status, health│ shield→가감속 제한   │
                                   └─────────────────┘                └────────────────────┘
```

- `/cmd_vel`과 effective `/dwa/status`의 발행자는 `cmd_arbiter` 하나다. A*는 계속
  `/dwa/status`를 구독하므로 `opticore-amr` 코드는 바꾸지 않는다.
- DWA는 fallback으로 계속 실행되며 출력 토픽만 remap된다. DWA의
  `/dynamic_obstacle_layer` 발행은 그대로 유지된다.
- 학습 planner의 ROS 의존 부분은 얇은 rclpy wrapper이고, 판단 로직은 ROS 없는
  core에 있다. core는 AMR-Lite 패키지의 관측·재표본화·shield 함수를 import한다.

## 3. 계층별 책임

| 계층 | Lite 위치 | ROS 위치 | 계약 |
|---|---|---|---|
| raw LiDAR → contract scan | `amr_lite.sensors.lidar` | 같은 함수 import | §4.2 |
| 경로 투영과 local goal | `amr_lite.planning.path_utils` | core가 import, progress 단조 탐색 추가 | §4.3 |
| 관측 생성 | `amr_lite.learning.observation` | 같은 함수 import | §4.1 |
| 정책 추론 | torch(학습) / `amr_lite.deploy.runtime`(NumPy) | NumPy runtime | §7 |
| 행동 → 명령 | `amr_lite.learning.action` | 같은 함수 import | §5 |
| Safety Shield | `amr_lite.planning.scan_shield` | 같은 클래스 import | §5.2 |
| 상태 머신·health | — | `amr_learned_planner.core` | §8, §9 |
| fallback 중재 | — | `amr_learned_planner.core.arbiter` | §9 |

## 4. 관측 계약 `amr-lite-obs-v1`

### 4.1 벡터 구성

`float32[296]`, 모든 값은 `[-1, 1]` 범위다.

| index | 내용 | 정규화 |
|---|---|---|
| 0–71 | scan(t−3) | `clip(r, 0, 8) / 8` |
| 72–143 | scan(t−2) | 동일 |
| 144–215 | scan(t−1) | 동일 |
| 216–287 | scan(t) | 동일 |
| 288 | local goal 거리 ρ | `clip(ρ / 8, 0, 1)` |
| 289, 290 | local goal 방향 α | `sin α`, `cos α` |
| 291 | 측정 선속도 v | `clip(v / v_max, 0, 1)` |
| 292 | 측정 각속도 w | `clip(w / w_max, -1, 1)` |
| 293 | signed cross-track error | `clip(e / 2.0, -1, 1)`, 경로 왼쪽이 양수 |
| 294, 295 | 경로 접선 heading 오차 | `sin`, `cos` |

- frame 순서는 oldest → newest다.
- local goal은 로봇의 경로 투영점에서 경로를 따라 `lookahead_distance`(0.65 m) 앞의 점이다.
- 제어 tick(20 Hz)마다 가장 최근 scan 하나를 frame stack에 넣는다. scan이 10 Hz로
  들어오면 같은 scan이 두 번 들어가는 zero-order hold가 계약이다.
- episode 시작 또는 첫 scan 수신 시 4개 frame을 모두 첫 scan으로 채운다.

### 4.2 Contract scan과 raw LiDAR 재표본화

contract scan은 로봇 중심(`base_footprint`)에서 heading 기준 `−135°…+135°`를 72개
bin으로 나눈 거리 배열이다. bin 중심은 `linspace(−135°, +135°, 72)`다.

raw scan(`angle_min`, `angle_increment`, `ranges`, `range_min`, `range_max`, 장착 pose)을
contract scan으로 바꾸는 규칙 `resample-v1`:

1. `NaN`, `−inf`, `r < range_min`은 무효로 버린다. `+inf`와 `r > range_max`는 반사 없음이다.
2. 유효 점을 장착 pose로 `base_footprint` 좌표로 옮긴다.
3. 로봇 중심 기준 방위각과 거리를 계산하고, 가장 가까운 bin 중심에 배정한다.
   `|방위각| > 135° + bin폭/2`인 점은 버린다.
4. bin 값은 배정된 점 거리의 최솟값이고, 점이 없으면 `8 m`다.
5. 결과를 `[0, 8]`로 자른다.

Lite의 `contract` LiDAR 모델(Phase 0/M1 기본값)은 로봇 중심에서 72개 ray를 직접 쏜다.
`raw` 모델은 Gazebo와 같은 720 sample/360°/장착 offset raw scan을 시뮬레이션한 뒤 위
함수로 재표본화한다. 배포 후보는 `raw` 모델로 학습·평가한다.

### 4.3 경로 기준값

- 경로는 `odom_filtered` 좌표의 점열이다. ROS에서는 `/global_path`를 받을 때 한 번만
  `map → odom_filtered`로 변환한다(기존 DWA 원칙과 같음).
- 투영은 가장 가까운 선분이며, ROS core는 이전 투영 index 주변만 탐색해 경로가
  접힐 때 뒤 구간으로 점프하지 않는다.

## 5. 행동 계약 `amr-lite-act-v1`과 명령 처리 순서

### 5.1 행동

`a = [a_v, a_w] ∈ [−1, 1]²`

```text
v_target = (a_v + 1) / 2 × v_max      # 0 ≤ v_target ≤ v_max, 후진 없음
w_target = a_w × w_max
```

`a`가 유한하지 않으면 `(0, 0)`이며 health fault다.

### 5.2 처리 순서 (Lite와 ROS 동일)

1. 정책 출력 `a_policy`
2. residual 정책이면 `a = clip(a_base + s × a_policy, −1, 1)`, `a_base`는
   `lite_rule_baseline(관측)`
3. 단위 변환 → `(v_target, w_target)`
4. 가감속 제한 `limit_velocity` (robot profile의 `accel_max`, `decel_max`,
   `angular_accel_max`)
5. Safety Shield: 제한된 명령을 짧은 horizon으로 적분해 LiDAR 점(및 선택적 track
   예측)과의 여유를 검사, 실패 시 감속 → 회피 → 정지 후보 순서
6. 실행 명령 발행

Shield 전략은 `legacy`(정답 지도·장애물), `predictive`(정답 위치 기반 tracker),
`scan`(LiDAR 점 + scan tracker) 세 가지다. ROS에서 가능한 것은 `scan`뿐이므로 배포
후보의 Shield ON 평가는 `scan`으로 한다.

## 6. Robot profile

| 값 | `lite_default` | `opticore_gazebo` | 근거 |
|---|---:|---:|---|
| `v_max` | 1.5 | 1.5 | DWA `v_max` (명세 2.0은 팀 합의 대기) |
| `w_max` | 1.8 | 1.5 | DWA `w_max` |
| `accel_max` | 1.0 | 1.0 | DWA `a_max` |
| `decel_max` | 1.5 | 2.0 | DiffDrive `max_linear_acceleration` |
| `angular_accel_max` | 3.0 | 1.5 | DWA `alpha_max`(측정 필요) |
| `robot_radius` | 0.36 | 0.36 | 외접원 |
| LiDAR 모델 | `contract` | `raw` | §4.2 |
| raw sample / 범위 | — | 720, −π…π | URDF `gpu_lidar` |
| raw range | — | 0.12–25 m | URDF |
| 장착 offset | — | (0.25, 0, 0) | URDF `lidar_x` |
| scan 주기 | 0.05 s | 0.1 s | 10 Hz |
| range 잡음 σ | 0 | 0.03 m | URDF noise |

profile은 정책 bundle에 기록된다. ROS 노드는 자신의 로봇 파라미터와 bundle profile의
행동 한계·LiDAR 장착값이 다르면 `PROFILE_MISMATCH`로 주행 권한을 받지 않는다
(`profile_check: warn`이면 shadow에서만 허용).

## 7. Policy bundle `format_version: 1`

```text
<bundle>/
  bundle.json   계약·정책 구조·profile·provenance·파일 SHA-256
  weights.npz   actor 가중치(float32)
  golden.npz    관측 N×296과 export 시점 torch 출력 N×2
```

`bundle.json` 핵심 필드:

| 필드 | 내용 |
|---|---|
| `format`, `format_version` | `opticore-learned-planner-bundle`, `1` |
| `observation_contract` | 버전 `amr-lite-obs-v1`, 크기, ray·FOV·range, frame 수, tail 이름, 정규화 상수 |
| `action_contract` | 버전 `amr-lite-act-v1`, `v_max`, `w_max` |
| `robot_profile` | 이름과 실제 값 |
| `policy` | 정책 종류, 네트워크 graph(층·활성화), `direct`/`residual`, residual scale, base 정책 설정 |
| `safety` | 학습·평가에 쓴 shield 설정 |
| `provenance` | checkpoint 경로·SHA-256, 학습 config SHA-256, git commit, seed, timesteps, 선정 근거 |
| `files` | `weights.npz`, `golden.npz`의 SHA-256 |

NumPy runtime은 torch 없이 bundle을 읽고, 로드 시 SHA-256과 golden parity를 확인한다.

## 8. ROS 토픽 계약 (`amr_learned_planner`)

이름은 모두 상대 이름이라 robot namespace 아래에서 그대로 동작한다.

| 토픽 | 타입 | 발행 | 구독 | 비고 |
|---|---|---|---|---|
| `cmd_vel` | `geometry_msgs/Twist` | `cmd_arbiter` | Gazebo DiffDrive | 단일 발행자 유지 |
| `dwa/status` | `std_msgs/String` | `cmd_arbiter` | A* | 활성 planner의 상태를 전달 |
| `dwa_fallback/cmd_vel` | `Twist` | DWA(remap) | `cmd_arbiter` | |
| `dwa_fallback/status` | `String` | DWA(remap) | `cmd_arbiter` | |
| `learned_planner/cmd_vel` | `Twist` | `learned_planner` | `cmd_arbiter` | |
| `learned_planner/status` | `String` | `learned_planner` | `cmd_arbiter` | DWA status 어휘 사용 |
| `learned_planner/health` | `String`(JSON) | `learned_planner` | `cmd_arbiter`, recorder | fault 목록, 추론 시간 |
| `cmd_arbiter/state` | `String`(JSON) | `cmd_arbiter` | recorder | mode, 활성 source, 전환 이유 |

입력 토픽은 DWA와 같다: `odometry/filtered`(fallback `odom`), `global_path`, `lidar`,
`goal_pose`.

학습 planner가 발행하는 상태 값:

| 상태 | 조건 | A* 동작 |
|---|---|---|
| `WAITING_ODOM` | odom 미수신 | — |
| `STOPPED` | 경로 없음 또는 빈 경로 | 재계획 reset |
| `NORMAL` | 정상 추종(PASS/CLAMP) | — |
| `EMERGENCY` | Shield STOP이 `emergency_after_sec` 이상 지속 | 즉시 재계획 |
| `PATH_LOST` | 경로 이탈 > `path_lost_offset` | 즉시 재계획 |
| `GOAL_REACHED` | 도착 순간 1회 | — |
| `REACHED` | 도착 후 유지 | — |

## 9. Arbiter 모드와 fallback

| 모드 | `/cmd_vel` | `dwa/status` | 용도 |
|---|---|---|---|
| `dwa` | DWA | DWA | 기존 동작, rollback |
| `shadow` | DWA | DWA | 학습 planner는 계산·기록만 |
| `learned` | 학습 planner, 고장 시 DWA | 활성 source | 학습 planner 주행 |

`learned` 모드 전이:

```text
LEARNED ──fault 또는 학습 cmd stale──► FALLBACK_HOLD(정지, hold_sec) ──► FALLBACK(DWA)
FALLBACK ──학습 planner가 recover_sec 동안 정상 && allow_recovery──► LEARNED
```

- 두 source가 모두 stale이면 정지 명령과 `STOPPED`를 발행한다.
- 전환마다 이유와 시각을 `cmd_arbiter/state`에 남긴다.

## 10. Health fault

| fault | 의미 | 기본 처리 |
|---|---|---|
| `BUNDLE_LOAD_FAILED` | 파일 없음, 포맷 오류, SHA-256 불일치 | 주행 권한 없음 |
| `CONTRACT_MISMATCH` | 관측·행동 계약 버전 불일치 | 주행 권한 없음 |
| `PROFILE_MISMATCH` | bundle profile과 노드 로봇 파라미터 불일치 | 주행 권한 없음(`warn`이면 경고) |
| `GOLDEN_PARITY_FAILED` | runtime 출력이 golden과 다름 | 주행 권한 없음 |
| `ODOM_STALE`, `SCAN_STALE` | 입력 timeout | 정지 + fault |
| `NON_FINITE_OBSERVATION`, `NON_FINITE_ACTION` | NaN/inf | 정지 + fault |
| `DEADLINE_MISS` | 추론+후처리가 deadline 초과 연속 N회 | fault |
| `SHIELD_SATURATION` | Shield STOP 지속 | `EMERGENCY` 상태, 설정 시 fault |

## 11. 패키지 구조와 의존성

```text
packages/amr_lite/amr_lite/
  envs/         Lite 환경, 지도(직사각형·grid), 절차적 시나리오, vector env
  sensors/      LiDAR raw 시뮬레이션·재표본화, 센서·동역학 randomization
  planning/     A*, 경로 투영, rule baseline, Safety Shield(legacy/predictive/scan), tracker
  learning/     관측·행동 계약, BC, DAgger, M1 PPO(동결)
  rl/           PPO v2, 네트워크, rollout buffer, curriculum
  deploy/       bundle export, NumPy runtime
  experiments/  manifest, sweep, 선정·test gate
  evaluation/   paired 평가, 통계, 그래프
integrations/ros2/amr_learned_planner/
  amr_learned_planner/core/   ROS 없는 planner·arbiter·health·recorder
  amr_learned_planner/*.py    rclpy 노드
  launch/ config/ test/
```

```text
integrations ──uses──► packages        (core는 amr_lite 계약 함수를 import)
experiments  ──uses──► packages
packages     ──never──► integrations, experiments, rclpy
```

ROS 컨테이너에는 `amr_lite`를 `--no-deps`로 설치해 torch 없이 NumPy·PyYAML만으로
동작하게 한다.

## 12. 검증 계약

- **Golden observation**: 고정 raw scan·odom·경로 입력과 기대 관측 벡터를 저장하고,
  Lite parity env와 ROS core가 모두 일치해야 한다.
- **Legacy parity**: 기본 설정(Phase 0/M1)의 trajectory·reward가 변경 전과 같아야 한다.
- **Runtime parity**: bundle의 NumPy 출력과 torch 출력 차이 ≤ `1e-5`.
- **Vector env parity**: 같은 seed에서 동기와 subprocess 실행 결과가 같아야 한다.
- **Residual identity**: residual 정책의 초기 행동은 rule baseline과 같아야 한다.
- **Arbiter**: 모드·fault·stale 조합별 출력 source가 표 §9와 일치해야 한다.
