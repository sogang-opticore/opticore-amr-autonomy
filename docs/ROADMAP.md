# OptiCore AMR Autonomy Roadmap — 규칙 기반 DWA에서 학습형 Local Planner로

## 0. 문서의 역할

이 문서는 `opticore-amr`의 규칙 기반 local planner(`dwa_node`)를 PPO 중심의 학습형
local planner로 대체하는 프로그램의 **마일스톤, 의사결정 gate, 완료 기준의 단일
기준**이다. 인원별 업무가 아니라 기술 산출물과 합격 조건을 정의한다.

| 문서 | 역할 |
|---|---|
| 본 문서 | 마일스톤, gate, 완료 기준, 위험 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 목표 스택, 관측·행동·bundle·토픽 계약 |
| [EXPERIMENT_PLAYBOOK.md](EXPERIMENT_PLAYBOOK.md) | M3 확인부터 M4 후보 선정까지의 학습 실험 절차와 예산 |
| [RUNBOOK.md](RUNBOOK.md) | 실행·검증 절차와 합격 판정 |
| [amr_learned_planner README](../integrations/ros2/amr_learned_planner/README.md) | opticore-amr 설치·실행·fault injection·paired 평가 (M5·M6) |
| [NEXT_STEPS.md](NEXT_STEPS.md) | 마일스톤 밖 후속 항목과 팀 결정 질문 |
| [REPOSITORY_STRUCTURE.md](REPOSITORY_STRUCTURE.md) | 디렉터리와 의존성 원칙 |

일정은 팀 가용 인원에 따라 달라지므로 날짜 대신 sprint 단위의 상대 규모를 쓴다.

## 1. 목표

### 1.1 연구 질문

> 경량 2D 환경에서 여러 번의 반복 실험으로 학습한 local 정책이, 현재 규칙 기반
> hybrid planner와 **동등 이상의 안전성**을 유지하면서 도달 시간, 정체 회복,
> 미학습 환경 일반화를 개선하고, 같은 가중치 그대로 `opticore-amr`의 ROS2/Gazebo
> 스택에서 동작하는가?

ML 적용 자체는 목표가 아니다. 학습 정책이 gate를 통과하지 못하면 규칙 기반
planner를 유지하는 것도 정상적인 결론이다.

### 1.2 무엇을 유지하고, 무엇을 학습하고, 무엇을 규칙으로 남기는가

| 구분 | 대상 |
|---|---|
| 유지 | A* 전역 경로와 `/global_path` 계약, 로봇 외형·동역학 한계, `/goal_pose` 인터페이스, `/dwa/status` 기반 A* 재계획 이벤트, Gazebo 최종 평가 환경 |
| 학습 | 로컬 관측(LiDAR 이력, local goal, 속도, 경로 오차)으로부터 목표 선속도·각속도를 제안하는 정책 |
| 규칙으로 남김 | 명령 범위와 가감속 제한, 센서·odom timeout, 단기 궤적 충돌 검사와 비상 정지, 정책 고장 시 기존 planner fallback |

권장 범위는 "end-to-end navigation"이 아니라 **learned local command proposal +
deterministic safety projection**이다.

### 1.3 비목표

- A* 전역 계획기 대체
- 실차(real robot) 이전 — M6 이후 별도 결정
- 특정 알고리즘 채택 강제 — PPO는 1순위 후보이며, 필요하면 SAC 등은 M4 결과로 결정

## 2. 출발점

### 2.1 현재 `opticore-amr` local planner

`dwa_node.py`(약 5,900줄)는 이름과 달리 정상 주행에서 속도 격자 탐색을 하지 않는다.
14차 튜닝 이후 Pure Pursuit 곡률과 적응 속도 상한을 중심으로, REJOIN·ALIGN·SPIN·
FORWARD_ONLY·EMERGENCY 상태 머신과 LiDAR cluster tracking, dynamic no-go layer 등
규칙이 누적된 **rule-based hybrid local planner**다. 설정 파일 스스로가 1–13차
patch chain이 진동·과회전·벽 고착을 반복해서 만들었다고 기록한다. 이 문서는 이
planner를 "DWA" 또는 "기존 planner"로 부르고, 비교 기준선 B0로 둔다.

### 2.2 AMR-Lite M1 기준선

| Policy | Shield | 성공률 (95% CI) | 충돌률 | Timeout |
|---|---:|---:|---:|---:|
| Rule (`lite_rule_baseline`) | OFF | 86.2% (84.4–87.5) | 13.8% | 0.0% |
| BC | OFF | 83.1% (80.0–85.6) | 16.9% | 0.0% |
| Custom Stable PPO 5-seed best | OFF | 85.4% (84.0–86.5) | 14.6% | 0.0% |
| Custom Stable PPO 5-seed best | ON | 83.0% (79.8–86.1) | 12.5% | 4.5% |
| Custom Stable PPO 5-seed final | OFF | 77.2% (69.9–84.2) | 22.8% | 0.0% |

상세는 [M1 결과](../experiments/m1_reproducibility/RESULTS.md)를 따른다. 남은 문제:

1. `D1` 정면 접근 충돌 100%, `S3` Shield saturation
2. best와 final checkpoint의 큰 차이 — 학습 안정성 부족
3. 시나리오 13종 고정 → 일반화 근거 없음
4. Lite와 Gazebo 로봇의 차이 (아래 표)
5. ROS2 실행 경로 부재 — 학습 결과를 `opticore-amr`에서 돌릴 방법이 없음

| 항목 | AMR-Lite 기본값 | `opticore-amr` Gazebo |
|---|---|---|
| LiDAR | 72 ray, 270°, 8 m, 로봇 중심 | 720 sample, 360°, 0.12–25 m, `+0.25 m` 전방 장착 |
| LiDAR 주기 / 잡음 | 20 Hz(매 제어 tick), 잡음 없음 | 10 Hz, Gaussian σ=0.03 m |
| `w_max` / 각가속 | 1.8 rad/s / 3.0 rad/s² | 1.5 rad/s / 1.5 rad/s²(DWA 설정) |
| 감속 한계 | 1.5 m/s² | DiffDrive 2.0 m/s² |
| 위치 | 정답 pose | EKF/AMCL, 보정 점프 존재 |
| 지도 | 직사각형 장애물 | 1207×840 occupancy grid(0.05 m) |

## 3. 목표 스택

```text
/goal_pose → A* ─/global_path─┬─→ learned_planner ─→ learned cmd/status/health ─┐
                ▲             └─→ dwa_node(fallback) ─→ fallback cmd/status ────┤
                │                                                               ▼
                └──────────── /dwa/status (effective) ←── cmd_arbiter ──→ /cmd_vel (단일 발행자)
```

- `learned_planner`는 Lite와 **같은 코드**로 관측을 만들고, export된 정책을 NumPy로
  추론한 뒤, 같은 안전 계층과 가감속 제한을 적용한다.
- `cmd_arbiter`만 `/cmd_vel`과 effective `/dwa/status`를 발행해 A* 계약을 지킨다.
- 모드는 `dwa`(기존 그대로), `shadow`(DWA 주행·학습 정책 기록), `learned`(학습 정책
  주행·고장 시 DWA fallback) 세 가지다.

계약과 상태 전이는 [ARCHITECTURE.md](ARCHITECTURE.md)를 따른다.

## 4. 마일스톤 한눈에 보기

| M | 이름 | 규모 | 코드 산출물 | 코드 상태 | 학습·검증 작업 | 종료 gate |
|---|---|---|---|---|---|---|
| M1 | 재현 가능한 평가 기준선 | 1 sprint | 완료 | ✅ | 완료 | G1 승인 |
| M2 | Sim-to-Sim parity와 배포 가능한 안전 계층 | 1–2 sprints | robot profile, LiDAR 재표본화, 센서·동역학 randomization, scan shield·tracker | ✅ 구현 | parity·D0–D4 stress 평가 | G2 |
| M3 | 대량 경량 학습 인프라 | 1–2 sprints | 절차적 시나리오, warehouse map, vector env, PPO v2, DAgger loop, sweep runner | ✅ 구현 | 장비별 처리량 측정 | G3 |
| M4 | 학습 실험과 배포 후보 선정 | 2–3 sprints | 선정·test gate, policy bundle, NumPy runtime | ✅ 구현 | **sweep 학습 실행**, 선정, test 1회 | G4 |
| M5 | `opticore-amr` ROS2 통합 | 1–2 sprints | `amr_learned_planner` 노드·arbiter·launch | ✅ 구현(ROS 실기동 전) | DWA pass-through → shadow → learned | G5 |
| M6 | Gazebo paired 검증과 release | 1–2 sprints | paired scenario, episode recorder, gap report | ✅ 구현(ROS 실기동 전) | Lite↔Gazebo paired 실행, release | G6 |

이 저장소의 현재 작업 범위는 **학습 실행을 제외한 M2–M6의 코드 산출물 전체**다.
학습 실행과 결과 판정은 각 마일스톤의 "학습·검증 작업"으로 남는다.

### 4.1 구현 현황 (2026-10-13)

| 영역 | 위치 | 검증 |
|---|---|---|
| 관측 parity | `amr_lite.sensors.lidar`, `learning.observation` | 기본 설정 rollout·reward·checkpoint가 변경 전과 bit 단위 동일, golden vector |
| 센서·동역학 randomization | `amr_lite.sensors.randomization`, `configs/robot_profiles/` | seed 재현성, 지연·hold·scale 단위 테스트 |
| 시나리오 | `envs.procedural`, `envs.grid_map`, `envs.warehouse_scenarios`, `configs/procedural.yaml`, `configs/paired/` | split별 재현성, A* 경로 보장, 정확한 거리장 |
| 안전 계층 | `planning.scan_shield` | 도달 가능한 명령, track 예측, Lite env 통합 |
| 학습 | `amr_lite.rl` (vector env, PPO v2), `learning.dagger_loop` | sync·subprocess 동일 결과, residual 초기 행동 = rule, pause/resume |
| 실험 운영 | `experiments.sweep`, `experiments.gate`, `configs/sweeps/`, `configs/gates/g4.yaml` | smoke sweep의 train→select→test(1회)→gate |
| 배포 | `amr_lite.deploy` (bundle export, NumPy runtime) | torch 출력과 `1e-5` 이내, torch 미import |
| ROS2 | `integrations/ros2/amr_learned_planner` | ROS 없는 core·노드 배선·launch 배선 테스트, Lite↔ROS 관측 일치(`1e-6`), Lite 폐루프 주행 |

ROS2/Gazebo 위에서의 실제 실행과 모든 학습 실험은 아직 수행하지 않았다.

## 5. 마일스톤 상세

### M1 — 재현 가능한 평가 기준선

상태: **완료**. 결과는 `experiments/m1_reproducibility/`에 보존하고 다시 쓰지 않는다.

- 학습 seed 5개, test 4,160 episode, scenario-stratified hierarchical bootstrap 95% CI
- selection(seed 20000)과 test(seed 30000) 분리, best/final 분리 보고
- manifest에 commit, dirty scope, dataset·checkpoint SHA-256, 환경 기록

M1 trainer(`amr_lite.learning.train_ppo`)와 protocol v1은 기준선 재현용으로 동결한다.
새 학습은 M3의 PPO v2와 protocol v2를 사용한다.

### M2 — Sim-to-Sim parity와 배포 가능한 안전 계층

상태: **코드 구현 완료, 평가 대기.** 기본 shield 전략은 여전히 `legacy`이며
`opticore_gazebo` 학습 설정은 `scan`을 쓴다.

초기 smoke 관찰(rule baseline, 시나리오당 6–8 seed, 공식 결과 아님): Gazebo
profile에서 scan shield는 정답 정보 없이 legacy shield와 같은 결과를 냈고 `D1`은
해결하지 못했다. scan shield에 후진 후보를 허용하면 동적 충돌은 줄었지만 정적
충돌과 timeout이 늘었다. 후진 허용 여부는 G2에서 결정한다.

목표:

- Lite 관측이 Gazebo/ROS 노드의 관측과 **같은 코드, 같은 숫자**가 되게 한다.
- 정답 장애물 정보 없이 LiDAR만으로 동작하는 안전 계층을 Lite와 ROS가 공유한다.

코드 산출물:

- `opticore_gazebo` robot profile: Gazebo 로봇의 속도·가감속·LiDAR 사양
- raw LiDAR 시뮬레이션(720 sample/360°/장착 offset)과 contract 72-ray 재표본화 함수
  — ROS 노드가 import하는 **동일 함수**
- scan hold(10 Hz), range 잡음, dropout, 각도 bias, 관측·명령 지연, 속도 scale,
  actuator lag, footprint 여유, localization 잡음, 경로 밀도·흔들림 randomization
- vectorized ray casting(기존 결과와 bit 단위 동일)
- LiDAR 점 기반 Safety Shield와 경량 obstacle tracker
- golden observation vector와 parity 테스트

학습·검증 작업:

- `D0`–`D4`와 절차적 동적 시나리오에서 legacy/predictive/scan shield 비교
- shield false-positive/negative와 saturation 계측
- randomization 강도별 rule baseline 성능 저하 곡선

완료 기준:

- 같은 raw scan·odom·path 입력에서 Lite parity env와 ROS core의 관측 차이가
  `1e-6` 이하이고 golden vector 테스트가 통과한다.
- 기본 설정(M1 protocol)의 trajectory, reward, checkpoint가 변경 전과 동일하다.
- scan shield가 `D1` 충돌률 10% 이하 또는 원인과 남은 위험을 문서화한다.
- Shield ON이 정적 시나리오 성공률을 OFF 대비 5%p 넘게 낮추지 않는다.

### M3 — 대량 경량 학습 인프라

상태: **코드 구현 완료.** 처리량 측정과 예산은
[EXPERIMENT_PLAYBOOK.md](EXPERIMENT_PLAYBOOK.md) §1을 따른다.

목표: 한 대의 개발 장비에서도 여러 설정 × 여러 seed 실험을 반복할 수 있게 한다.
"지도"가 아니라 "지도 생성 규칙"을 기준으로 train/validation/test를 분리한다.

코드 산출물:

- 절차적 시나리오 family: aisle, hall, pinch, junction과 각 family의 동적 에이전트
  변형(횡단·정면·추종·정지·왕복). family별 파라미터 범위를 split마다 분리
- `opticore-amr`의 `warehouse_map.pgm`을 읽는 grid map(거리장 기반 충돌·ray casting)과
  실제 창고 지도 위 무작위 start/goal 시나리오
- 동기·subprocess vector env와 결정적 seed 체계
- PPO v2: vector rollout, direct/residual 정책, 분리 actor·critic, value 정규화, KL
  조기 종료, learning-rate/entropy schedule, 비용 critic과 Lagrangian 제약(PPO-Lag),
  curriculum과 randomization ramp, MLP/1D-CNN encoder, resume
- BC → DAgger 반복 loop와 위험 상태 가중치
- sweep runner: 설정 grid × 학습 seed 확장, 병렬 실행, 재개, 집계

학습·검증 작업:

- 모든 trainer와 sweep의 smoke 통과
- 장비별 env step/s와 trial 소요 시간 실측 → M4 예산 확정

완료 기준:

- 같은 seed에서 동기·subprocess vector env 결과가 동일하다.
- 절차적 시나리오는 seed로 재생성되고 A* 경로가 보장되며 split 간 파라미터 범위가
  문서화돼 있다.
- residual 정책의 초기 행동이 rule baseline과 일치한다.
- sweep이 중단 후 재개되고 완료 trial을 다시 학습하지 않는다.

### M4 — 학습 실험과 배포 후보 선정

상태: **코드 구현 완료, 학습 실험 대기.** 실험 절차는
[EXPERIMENT_PLAYBOOK.md](EXPERIMENT_PLAYBOOK.md), sweep 정의는
`packages/amr_lite/configs/sweeps/e1_dagger.yaml`–`e6_candidate.yaml`이다.

목표: 사전 등록한 protocol v2로 후보를 고르고 test split은 한 번만 연다.

코드 산출물:

- protocol v2 selection/test 설정(고정 시나리오 + 절차적 validation/test family +
  warehouse map)
- 사전 등록 선정 규칙: 안전 제약 → 성공률 중앙값 → 최악 seed → 시간
- reward hacking·행동 이상(선회, 정지, 진동) 자동 flag
- rule baseline 대비 paired 비열등성 test gate와 보고서
- policy bundle export(weights, 계약, robot profile, provenance, golden vector,
  SHA-256)와 torch 없는 NumPy runtime

학습·검증 작업 (이번 범위에서 실행하지 않음):

| 실험 | 내용 | 목적 |
|---|---|---|
| E1 | BC → DAgger 3회 | 교사 모방 상한과 covariate shift 감소 확인 |
| E2 | PPO v2 direct vs residual | 초기 붕괴·checkpoint 의존성 해결 여부 |
| E3 | Lagrangian on/off, cost limit | 충돌 비용 제약 효과 |
| E4 | randomization off/on | Gazebo 전이 대비 |
| E5 | MLP vs 1D-CNN, frame 1 vs 4 | 관측 표현 ablation |
| E6 | 최종 후보 5 seed 장기 학습 | 배포 후보 |

완료 기준 (G4 제안):

- 후보는 robot profile `opticore_gazebo`로 학습·평가된다.
- test split에서 rule 대비 성공률 차이의 95% CI 하한 ≥ −2%p, 충돌률 차이의 95% CI
  상한 ≤ +1%p (비열등)
- 동적 family 성공률 또는 평균 도달 시간 중 하나 이상에서 rule 대비 개선
- 학습 seed 최악값이 중앙값보다 5%p 넘게 낮지 않고, final이 best보다 10%p 넘게
  저하되지 않는다.
- 행동 이상 flag가 없고 Shield ON 개입률과 saturation이 보고된다.
- bundle의 NumPy runtime과 torch 정책 출력 차이가 golden vector에서 `1e-5` 이하다.

### M5 — `opticore-amr` ROS2 통합

상태: **코드 구현 완료, ROS2/Gazebo 실기동 대기.** 설치와 단계별 명령은
[amr_learned_planner README](../integrations/ros2/amr_learned_planner/README.md).
학습 전에는 Phase 0 Stable PPO bundle(`lite_default`, `profile_check:=warn`)로
shadow 파이프라인만 점검할 수 있다.

목표: 같은 bundle을 `opticore-amr`에서 실행하되, 기존 DWA를 끄지 않고 단계적으로
권한을 넘긴다.

코드 산출물 (`integrations/ros2/amr_learned_planner`):

- ROS 의존성 없는 core: LaserScan 재표본화(=Lite 함수), progress 단조 경로 투영,
  frame stack, 관측 생성, 정책 추론, scan shield, 가감속 제한, 목표 도달·상태 머신,
  health monitor, command arbiter
- rclpy 노드: `learned_planner`, `cmd_arbiter`, `nav_episode_recorder`,
  `goal_sequence_runner`
- 단일 로봇 launch와 다중 로봇 helper, 파라미터 YAML
- bundle 계약 검사(포맷, SHA-256, 관측 계약, robot profile, golden parity)
- fault injection 파라미터(NaN, deadline, stale sensor)

검증 단계:

1. `mode:=dwa` — 기존 동작과 차이가 없음을 확인
2. `mode:=shadow` — 학습 정책은 명령만 기록, DWA가 주행
3. `mode:=learned` — 학습 정책 주행, 고장 시 DWA fallback

완료 기준 (G5 제안):

- `dwa` 모드에서 A* 재계획과 `GOAL_REACHED` 흐름이 기존과 같다.
- 20 Hz에서 추론 deadline miss가 측정·보고되고 0.1% 이하다.
- NaN 행동, 모델 로드 실패, bundle 불일치, stale scan/odom 주입 시 1 제어 주기 안에
  정지 또는 DWA로 전환된다.
- shadow 모드 최소 30 goal에서 학습 정책의 shield 개입률과 DWA 명령 차이가 기록된다.

### M6 — Gazebo paired 검증과 release

상태: **도구 구현 완료, 실행 대기.** paired set `warehouse_v1`(8개 leg, Gazebo mover
4개 포함), Gazebo runner·recorder, Lite paired 실행과 gap report가 준비됐다.

목표: 같은 창고 지도·같은 goal 순서로 Lite와 Gazebo를 짝지어 실행해 sim-to-sim
차이를 측정하고, 통과하면 학습 planner를 기본 모드로 승격한다.

코드 산출물:

- 지도 좌표로 쓴 paired scenario set과 Lite 실행기
- Gazebo episode recorder와 Lite와 같은 CSV schema
- Lite↔Gazebo gap 보고서 생성
- release 절차: bundle checksum 고정, release note, `mode:=dwa` rollback

학습·검증 작업:

- paired set을 Lite(seed 다수)와 Gazebo(경로당 3회 이상)에서 DWA·학습 정책으로 실행
- 실패 taxonomy 정리와 허용 운용 범위 문서화

완료 기준 (G6 제안):

- Gazebo paired set에서 학습 정책의 충돌 0건, 성공률이 DWA 이상
- 평균 도달 시간 또는 정체 회복에서 DWA 대비 개선이 재현된다.
- Lite 대비 Gazebo 성공률 하락 폭이 보고되고 원인이 taxonomy로 분류된다.
- 깨끗한 환경에서 문서만으로 설치·smoke·rollback이 가능하다.

## 6. 의사결정 Gate

| Gate | 시점 | 결정 |
|---|---|---|
| G1 | M1 종료 | 평가 seed, 통계, checkpoint 선택 규칙 승인 — **완료** |
| G2 | M2 종료 | 학습·배포 robot profile, LiDAR 경로(raw 재표본화), 기본 shield 전략, randomization 범위 |
| G3 | M3 종료 | protocol v2 동결, 실험 매트릭스와 예산, 선정 규칙 사전 등록 |
| G4 | M4 종료 | ROS로 가져갈 정책과 모드(direct/residual) 선정. PPO 채택은 필수가 아님 |
| G5 | M5 종료 | shadow → learned 전환 허용 |
| G6 | M6 종료 | 학습 planner를 기본 모드로 release, rollback 절차 승인 |

Gate가 정한 기준은 결과를 본 뒤 바꾸지 않는다. 바꿔야 하면 결정 문서에 이유를 남기고
protocol version을 올린다.

## 7. 실험 예산 산정

학습 실행은 M4의 작업이다. 예산은 M3에서 측정한 처리량으로 확정한다.

```text
trial 수 = 설정 수 × 학습 seed 수
CPU 시간 ≈ trial 수 × timesteps ÷ (env step/s × worker 수)
         + 평가 episode 수 × 평균 episode step ÷ env step/s
```

권장 초기 매트릭스는 E1–E5 각각 설정 2–4개 × seed 3개, 1M timestep이며, E6만
seed 5개와 장기 budget을 쓴다. 평가는 selection을 학습 중 주기적으로, test를
후보 확정 후 한 번만 실행한다.

## 8. 위험과 대응

| 위험 | 신호 | 대응 |
|---|---|---|
| 관측 wrapper 불일치 | Gazebo에서만 실패, golden test 실패 | 재표본화·정규화를 Lite와 ROS가 같은 함수로 사용, golden vector |
| PPO 붕괴·checkpoint 의존 | best–final 차이, seed 분산 | residual 정책, KL 조기 종료, BC anchor, seed 5개 보고 |
| 보상 해킹 | 선회·정지·진동 flag | 실패 taxonomy 자동 판정, 비용 제약, 행동 flag 시 후보 제외 |
| 안전 계층이 성공률을 잠식 | saturation, timeout 증가 | 개입률을 핵심 지표로 보고, shield ON 학습·비용 제약 비교 |
| Lite↔Gazebo 동역학 차이 | Gazebo 충돌·진동 | profile 정합, actuator lag·지연 randomization, paired gap 보고 |
| 추론 지연 | deadline miss | NumPy runtime, 지연 계측, deadline 위반 시 fallback |
| fallback 전환 중 불안정 | 전환 직후 급회전 | 전환 시 정지 hold, 최소 유지 시간, 전환 로그 |
| 테스트 split 오염 | 반복 test 실행 | test는 후보 확정 후 1회, 재실행은 protocol version 증가 |

## 9. 공통 보고 원칙

- 성공률만 보지 않고 충돌률, timeout, minimum clearance를 함께 보고한다.
- 학습 seed와 평가 seed를 구분한다. 학습 seed는 20000 미만을 쓴다.
- Shield OFF와 ON 결과, best와 final checkpoint를 분리한다.
- shield 개입률과 saturation, fallback 횟수를 핵심 지표로 기록한다.
- 실패한 실험도 설정과 원인을 기록한다.
- 목표 수치가 바뀌면 결과가 아니라 결정 문서에 변경 이유를 남긴다.

## 10. 기존 로드맵 대비 변경

| 이전 | 현재 | 변경 이유 |
|---|---|---|
| M2 동적 장애물·안전 | M2 parity + 배포 가능한 안전 계층 | ROS에는 정답 장애물이 없으므로 scan 기반 안전 계층과 관측 parity가 선행돼야 함 |
| M3 정책 안정화 | M3 학습 인프라 + M4 학습 실험 | "여러 번의 경량 실험"을 위한 인프라와 실제 학습 실행을 분리 |
| M4 ROS2/Gazebo 통합 | M5 ROS2 통합 + M6 Gazebo paired 검증 | 통합(동작)과 검증(성능 주장)의 gate를 분리 |
| M5 release | M6에 포함 | release는 paired 검증 결과에 종속 |
