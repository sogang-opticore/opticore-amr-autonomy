# Spike 회고 — AMR-Lite 탐색(2026-09-06 ~ 2026-10-13)에서 배운 것

이 문서는 학습형 local planner의 가능성을 보려고 진행한 탐색 spike를 팀 인수 관점에서
정리한다. spike는 구현 대부분을 AI에 위임해 빠르게 진행했다. 무엇을 만들었고, 어떤
근거가 생겼고, 무엇을 배웠고, 무엇을 의심해야 하는지를 남긴다. 구성요소별 근거 수준은
[VERIFICATION_STATUS](VERIFICATION_STATUS.md), 앞으로의 선택지는
[DIRECTION_OPTIONS](DIRECTION_OPTIONS.md)를 따른다.

## 1. 요약

- **코드는 M6까지 있지만 근거는 M1까지만 있다.** M2–M6(센서 parity, scan shield,
  PPO v2, sweep·gate, bundle, ROS2 통합)는 모두 2026-10-13 하루에 들어왔다. 정식 학습
  sweep, ROS2 실기동, Gazebo 실행은 한 번도 하지 않았다.
- 유일한 공식 결과(M1)는 **학습 정책이 단순 규칙보다 낫다는 증거가 없다**는 것이다.
  custom Stable PPO 5-seed best 85.4%(95% CI 84.0–86.5), rule baseline 86.2%
  (84.4–87.5), Shield OFF 기준.
- spike의 가장 큰 자산은 정책이 아니라 **평가 체계**(seed 분리, paired 평가, 계층적
  bootstrap, manifest)와 **Lite↔ROS가 같은 함수를 쓰는 계약 구조**다.
- 이번 검토에서 G4 gate와 선정이 shield 없는 rule baseline과 비교하도록 정의돼 있어 너무
  쉽게 통과할 수 있고, rule baseline 자체도 장애물이 footprint 안에 들어온 뒤에야 멈춘다는 것을
  확인했다. 학습·평가·Gazebo 기록을 오염시키는 결함도 여럿 찾았다(§6).

## 2. 진행 경과

| 날짜 | 단계 | 내용 | 남은 근거 |
|---|---|---|---|
| 2026-09-06 | Phase 0 실행 | `opticore-amr/amr_lite_spike/`(untracked)에서 Lite 환경, rule, BC, DAgger-lite, PPO, shield를 한 번에 구현·실행 | `experiments/phase0_amr_lite/` |
| 2026-09-08 | Phase 0 이관 | 이 저장소로 옮기고 결과·결정·한계 문서화 | commit `6f194a7`–`585969a` |
| 2026-09-09 | 구조 정리 | `packages/`·`experiments/`·`docs/` 분리 | `180a066`, `f2ffd88` |
| 2026-09-14 | 예측 shield | `opticore-amr`의 별도 branch에서 predictive shield core 작성 | 이후 `c44806c`로 이관 |
| 2026-09-14~15 | 연구계획서 | 창의융합 자유연구 계획서 작성(5인, 16주, 200만 원) | `opticore-amr/docs/*.docx` |
| 2026-09-30 | M1 | 5 seed × 4,160 test episode 재현성 평가. `opticore-amr`에 잘못 들어간 작업은 revert(`447d08c`) | `experiments/m1_reproducibility/` |
| 2026-10-13 | M2–M6 코드 | 로드맵 재정의, 센서 parity·randomization, 절차적·창고 시나리오, scan shield, PPO v2·DAgger, sweep·G4·bundle, ROS2 패키지, 문서 동기화 | commit `ca61e34`–`4e83d29` (8개) |
| 2026-10-14 | 인수인계 검토 | 테스트·smoke 재확인, rule baseline 진단, 이 문서 묶음 작성 | `docs/handover/`, `docs/internals/` |

## 3. 만든 것

| 영역 | 위치 | Python 줄 수 | 최고 근거 수준 |
|---|---|---:|---|
| Lite 환경·지도·시나리오 | `amr_lite/envs/`, `config.py` | 약 2,060 | E4 (고정 시나리오) / E1 (절차적·창고) |
| 센서 모델·randomization | `amr_lite/sensors/` | 약 330 | E1 |
| A*·경로·rule·shield | `amr_lite/planning/` | 약 910 | E4 (legacy) / E1+E3 (scan) |
| 관측·BC·DAgger·M1 PPO | `amr_lite/learning/` | 약 900 | E4 (BC, M1 PPO) |
| PPO v2·vector env | `amr_lite/rl/` | 약 1,000 | E1+E2 |
| 평가·통계 | `amr_lite/evaluation/` | 약 580 | E4 |
| sweep·gate·manifest·paired | `amr_lite/experiments/` | 약 690 | E2 |
| bundle export·runtime | `amr_lite/deploy/` | 약 310 | E1+E2 |
| CLI | `amr_lite/cli/` | 약 630 | E2 |
| ROS2 core | `integrations/ros2/.../core/` | 약 800 | E1 |
| ROS2 노드·launch | `integrations/ros2/.../*.py`, `launch/` | 약 710 | E1 (가짜 rclpy) |
| 테스트 | `tests/`, `test/` | 약 2,490 | 145개 통과 |

근거 수준(E0–E5)의 정의는 [VERIFICATION_STATUS](VERIFICATION_STATUS.md) §1에 있다.

## 4. 근거로 확인된 사실

| 출처 | 사실 |
|---|---|
| Phase 0 | 100k step PPO from scratch는 목표 대신 작은 원을 도는 reward hacking에 수렴했다(성공 0%) |
| Phase 0 | reward-v2(경로 progress, 회전·heading·시간 비용)로 바꿔도 성공 0%였다 |
| Phase 0 | BC warm-start PPO의 best는 timestep 0이었다. PPO update는 BC actor를 망가뜨렸다 |
| Phase 0 | actor/critic 분리, value 정규화, KL 조기 종료를 넣은 Stable PPO는 51.2k step에서 BC보다 나은 checkpoint를 만들었다 |
| M1 | Phase 0의 87.5%(16 episode)는 randomized test에서 68.8%였다. 작은 평가가 크게 과대평가했다 |
| M1 | 5 seed best 85.4%, rule 86.2%, paired 차이 −0.9%p(95% CI −2.0–0.0) |
| M1 | final checkpoint는 best보다 최대 19.4%p 낮았다(seed 7). checkpoint 선택 의존이 남았다 |
| M1 | 정적 S0–S5는 PPO best가 Shield OFF 100% 성공. 실패는 D1(정면 접근 100% 충돌)과 D0에 몰렸다 |
| M1 | Shield ON은 S3의 36%를 `SHIELD_SATURATION` timeout으로 바꿨다 |
| M2 smoke (2026-10-13) | scan shield는 정답 정보 없이 legacy와 같은 결과를 냈지만 D1을 풀지 못했다. 후진을 허용하면 동적 충돌은 줄고 정적 충돌·timeout은 늘었다 |
| 진단 (2026-10-14) | protocol v2 selection에서 rule baseline은 Shield OFF 55/85 성공, 동적 충돌 30건 전부 주행 중. Shield ON 59/85, saturation 11건 |

## 5. 배운 것

### L1. 작은 평가는 착시를 만든다
16 episode의 87.5%가 4,160 episode에서 68.8%가 됐다. seed 분리, 계층적 bootstrap,
best/final 분리 보고가 spike의 가장 값진 산출물이다. **앞으로 모든 주장은 protocol
결과로만 한다.**

### L2. 정적 주행에서는 규칙이 이미 상한에 가깝다
정적 시나리오는 rule과 PPO가 모두 거의 100%다. 학습의 이점은 동적 상호작용에서만
증명할 수 있다. 연구 질문과 gate는 동적 시나리오 중심이어야 한다.

### L3. PPO의 문제는 "학습되느냐"가 아니라 "학습된 상태로 남느냐"다
from scratch는 reward hacking, warm-start는 붕괴, Stable PPO는 best/final 차이로 나타났다.
residual 정책(rule + 학습 보정, 초기 행동 = rule)이 코드상 가장 유력한 해법이지만
아직 학습해 본 적이 없다. custom PPO는 표준 구현과 비교한 적도 없다.

### L4. reward를 고쳐서 해결하려 하면 안 된다
reward-v2는 성공률을 바꾸지 못했다. 행동 이상(선회·정지·진동)은 reward 조정보다
failure taxonomy로 자동 판정하고 후보에서 빼는 편이 낫다(현재 G4 behaviour 항목).

### L5. Safety shield는 충돌을 timeout과 saturation으로 바꾼다
shield 개입률과 saturation을 충돌률과 같은 급의 지표로 봐야 한다. 전진만 하는 로봇에
비반응형 장애물이 정면으로 다가오는 D1은 shield만으로는 풀리지 않는다. `PD-*` test split에서
rule baseline의 동적 충돌 3건 중 2건은 로봇이 거의 멈춘 상태에서 받힌 것이었다. 시나리오의
비반응형 가정 자체를 검토하고, 정지 중 피격을 따로 집계해야 한다.

### L6. 무엇과 비교하느냐가 결론을 정한다
Lite의 `lite_rule_baseline`(44줄)은 실제 `dwa_node`(5,875줄의 Pure Pursuit 기반 hybrid)와
다르고, shield 없이는 지도에 없는 장애물을 피하지 않으며, 정지 임계값이 footprint 안쪽이다.
그런데 G4는 Shield OFF에서 이 baseline과 비교하고, 같은 코드가 BC 교사와 residual base로도
쓰인다(§6.1). 실제 B0와의 비교는 Gazebo에서만 가능하다.

### L7. Sim-to-sim 차이는 학습 전에 고정해야 한다
robot profile, raw LiDAR 재표본화, randomization은 구현됐지만 Gazebo 실측값이 없다.
profile이 바뀌면 학습을 다시 해야 하므로 `v_max`, `angular_accel_max`, 후진 허용,
`goal_tolerance` 결정이 학습보다 먼저다([NEXT_STEPS](../NEXT_STEPS.md) §3).

### L8. AI 위임 개발은 "배선은 촘촘하고 거동 근거는 없는" 코드를 만든다
테스트 145개는 대부분 계약·재현성·배선을 확인한다. 정책이 잘 달리는지는 확인하지
않는다. 코드와 문서가 함께 생성돼 "구현됨"과 "검증됨"의 경계가 흐려지기 쉽다.
팀은 코드를 소유권 단위로 나눠 읽고, 직접 실행하고, 의심 항목을 기록하며 인수해야 한다.

### L9. 최종 주장은 Gazebo 스택 상태에 달려 있다
`opticore-amr`의 2026-10-13 서버 진단은 4대 실행 시 RTF 약 0.3, namespace 때문에 튜닝
YAML이 적용되지 않는 문제를 보고했다. M5·M6은 이 저장소 밖의 문제를 먼저 풀거나
단일 로봇으로 범위를 좁혀야 진행된다.

## 6. 이번 검토에서 새로 확인한 것

2026-10-14 검토에서 코드를 직접 읽고 작은 실행으로 확인한 결과다. 영향이 큰 것만 고른다.
전체 목록과 근거는 각 [internals](../internals/README.md) 문서의 "주의할 점과 알려진 한계"
절에 있고, 실행 전 수정 계획은 [DIRECTION_OPTIONS](DIRECTION_OPTIONS.md) §4에 있다.

### 6.1 평가 전제

| 발견 | 영향 | 근거 |
|---|---|---|
| G4 비열등성·개선·robustness와 학습 중·sweep 선정이 모두 Shield OFF + `lite_rule_baseline` 기준이다. 학습과 배포는 Shield ON이다 | 배포 구성의 성능을 판정하지 못하고, 동적 성공률 개선 기준이 쉽게 통과된다 | VERIFICATION_STATUS §4.1, EVALUATION_AND_EXPERIMENTS §8.1 |
| rule baseline은 Shield ON에서도 saturation 12.9%로 G4 상한 5%를 넘는다 | 상한이 비현실적이거나 shield 튜닝이 먼저 필요하다 | 같음 |
| rule baseline의 `clearance_stop` 0.12 m가 로봇 반경 0.36 m보다 작고 scan 거리는 로봇 중심 기준이다. 장애물이 footprint 안에 들어온 뒤에야 멈춘다 | 기준선이 약하고, 같은 코드가 BC 교사와 residual base라 학습에도 전파된다 | SENSORS_AND_SAFETY §8-1 |
| 성공한 episode에도 `OSCILLATION` 등 행동 이상 label이 붙고 G4 behaviour 비율은 성공을 포함해 센다 | 성공이 실패 행동으로 집계될 수 있다(rule 30 episode 중 성공 2건 실측) | ENVIRONMENT §8-1 |
| `PD-*` test split에서 rule baseline의 동적 충돌 18건 중 12건은 로봇 속도 < 0.05 m/s였다 | 비반응형 agent에 정지 상태로 받힌 충돌이 정책 실패로 집계된다 | ENVIRONMENT §8-6 |
| 학습 중 best 선정 seed(20000–20001)가 sweep selection seed(20000–20004)와 겹친다 | selection 점수가 낙관적이다 | LEARNING §8-6 |

### 6.2 시뮬레이션과 안전 계층

| 발견 | 영향 | 근거 |
|---|---|---|
| scan tracker가 `static_filter_distance` 0.0에서 벽을 동적 track으로 만든다 | 정적 시나리오에서도 shield가 개입한다(P-HALL 1015 → 1638 step) | SENSORS_AND_SAFETY §8-4 |
| legacy shield의 SLOW·STOP이 감속 한계를 거치지 않는다(한 step 12.8 m/s²) | M1 Shield ON 결과가 물리적으로 낙관적이다 | SENSORS_AND_SAFETY §8-3, ENVIRONMENT §8-3 |
| `GridMap` ray cast가 수렴하지 못하면 짧은 거리를 낸다(창고 지도 scan의 6.75%) | `WH-*` 시나리오에 가짜 근거리 장애물이 생긴다 | ENVIRONMENT §8-4 |
| scan·legacy shield의 FP/FN 열은 항상 0이다 | protocol v2의 해당 지표는 의미가 없다 | SENSORS_AND_SAFETY §8-5 |
| 절차적 split의 파라미터 범위가 겹치고 `WH-RANDOM`은 split과 무관하다 | "미학습 환경 일반화" 주장의 근거가 약하다 | ENVIRONMENT §8-7 |

### 6.3 학습과 실험 운영

| 발견 | 영향 | 근거 |
|---|---|---|
| PPO-Lag의 λ 갱신과 `step_cost`에 truncation bootstrap 값이 섞인다 | E3 비용 제약이 실제 비용이 아니라 critic 예측을 따른다 | LEARNING §8-1 |
| crash 후 resume하면 `training.csv` 행이 중복되고 `best.pt`가 더 나쁜 checkpoint로 덮일 수 있다 | 장시간 sweep 결과가 오염된다 | LEARNING §8-2 |
| smoke와 실제 sweep이 같은 기본 출력 폴더를 쓰고 완료 판정이 smoke·config 변경을 보지 않는다 | 실제 학습이 smoke trial을 완료로 보고 건너뛴다 | EVALUATION_AND_EXPERIMENTS §8.3 |
| sweep은 manifest를 남기지 않는다 | M4 결과의 commit·source hash 추적이 안 된다 | EVALUATION_AND_EXPERIMENTS §8.4 |
| residual scale 0.5에서는 rule 명령이 `v_max`의 25% 이하일 때만 정책이 로봇을 멈출 수 있다 | D1류 정면 접근에서 residual 정책의 권한이 제한된다(추정) | LEARNING §8-4 |
| custom PPO를 참조 구현과 비교한 적이 없고 Gymnasium adapter도 사용처·테스트가 없다 | 학습 실패 시 알고리즘 결함과 문제 난이도를 구분할 수 없다 | LEARNING §4.9 |

### 6.4 ROS2·Gazebo

| 발견 | 영향 | 근거 |
|---|---|---|
| `opticore-amr`에 learned 패키지를 붙일 단일 로봇 bring-up이 없다(`robot.launch.py`는 A*·DWA를 항상 띄운다) | README의 M5 절차가 launch 수정 없이는 실행되지 않을 가능성이 높다 | DEPLOY_AND_ROS2 §4.9 |
| recorder가 이전 goal의 `REACHED`를 새 episode 성공으로 기록하고, 위치(odom)와 goal(map) 좌표계를 섞고, 건너뛴 leg를 CSV에 남기지 않는다 | M6 paired 결과가 오염된다 | DEPLOY_AND_ROS2 §8-1–8-3 |
| `profile_check: warn`이 shadow로 제한되지 않는다 | profile이 다른 bundle이 learned 모드에서 주행 권한을 받을 수 있다 | DEPLOY_AND_ROS2 §8-5 |
| G5의 "학습 정책–DWA 명령 차이"를 기록하는 노드가 없다 | G5 판정에 rosbag이 따로 필요하다 | DEPLOY_AND_ROS2 §8-10 |
| 토픽 이름·타입·status 문자열은 `dwa_node`·`astar_node`와 일치한다 | 인터페이스 계약 자체는 맞다 | DEPLOY_AND_ROS2 §4.9 |

### 6.5 문서

| 발견 | 조치 |
|---|---|
| RUNBOOK Gate 1 "약 6초" → 실제 105.6 s | RUNBOOK 수정 |
| `packages/amr_lite/README.md`가 D3을 "통로 정지"로 설명하지만 D3은 빈 `S0` 지도 위 정지 장애물이다 | README 수정 |
| Phase 0 `KNOWN_LIMITATIONS.md`가 이후 구현된 항목(잡음, 병렬 env 등)을 미구현으로 적는다 | 시점 기록이라는 머리말 추가 |
| EXPERIMENT_PLAYBOOK §0의 "학습 seed는 코드가 강제"는 PPO v2에만 맞다. teacher·DAgger 수집은 `seed + i`를 쓴다 | LEARNING §8-11에 기록 |
| 연구계획서 연구 질문 2는 "선행 DWA가 8방향 행동"이라고 전제하지만 현재 `dwa_node`는 Pure Pursuit 연속 제어 중심 hybrid다(ROADMAP §2.1) | 연구 질문 2의 비교 대상을 다시 정의해야 한다 |

## 7. 연구계획서와의 대응

| 계획서 단계 | 대응 마일스톤 | 코드 | 근거 |
|---|---|---|---|
| 1단계 평가 기준 고정 | M1 | 완료 | E4 완료 |
| 2단계 동적 장애물 인식과 안전 제어 | M2 | 구현 | E1+E3, G2 미판정 |
| 3단계 학습 정책 안정화 | M3·M4 | 구현 | E1+E2, 학습 미실행 |
| 4단계 ROS2·Gazebo 통합 검증 | M5·M6 | 구현 | E1, 실기동 없음 |
| 5단계 결과 정리와 의사결정 | G6 | — | — |

계획서 2단계의 제안 기준(D1 충돌률 10% 이하, saturation·timeout 5% 이하, 정적 Shield ON
성공률 저하 5%p 이내)과 3단계 기준(성공률 중앙값 85% 이상, 충돌률 중앙값 10% 이하,
best–final 차이 10%p 이내)은 ROADMAP의 G2·G4와 대체로 같다.

## 8. 유지할 것, 보류할 것, 의심할 것

| 구분 | 대상 | 이유 |
|---|---|---|
| 유지 | protocol·seed namespace·계층적 bootstrap·manifest | M1에서 착시를 잡아냈다 |
| 유지 | 관측 계약, golden vector, Lite↔ROS 공유 함수 | sim-to-sim 불일치의 가장 흔한 원인을 구조로 막는다 |
| 유지 | ROS 판단 로직을 rclpy 없는 `core/`에 둔 구조와 arbiter fallback | ROS 없이 테스트할 수 있다 |
| 유지 | sweep 사전 등록 선정과 test 1회 제한 | test split 오염을 막는다 |
| 보류 | M1 trainer(`learning/train_ppo.py`) | M1 재현용으로 동결. 새 학습은 PPO v2 |
| 보류 | legacy·predictive shield | 정답 정보를 써서 ROS로 가져갈 수 없다. 비교 기준으로만 쓴다 |
| 보류 | GRU·SAC·사람 이동 모델 등 확장 | 기본 경로의 근거가 생긴 뒤에 판단한다 |
| 의심 | custom PPO의 정확성 | 외부 구현과 비교한 적 없다 |
| 의심 | G4 기준과 saturation 상한 | §6 |
| 의심 | `opticore_gazebo` profile 값 | Gazebo 실측 없음 |
| 의심 | 실행해 본 적 없는 rclpy 노드·launch | 가짜 rclpy 테스트만 통과 |
| 의심 | 비반응형 동적 장애물 시나리오 | D1류 실패가 정책 문제인지 시나리오 문제인지 분리되지 않았다 |
