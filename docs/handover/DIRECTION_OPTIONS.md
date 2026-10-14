# 향후 방향 — 선택지, 추천, 2주 실행 계획

spike를 5인 팀이 이어받을 때의 방향 선택지를 비교하고 추천안과 2주 실행 계획을 제안한다.
판단 근거는 [SPIKE_REVIEW](SPIKE_REVIEW.md)와 [VERIFICATION_STATUS](VERIFICATION_STATUS.md),
마일스톤과 gate 정의는 [ROADMAP](../ROADMAP.md)를 따른다. 이 문서는 제안이며 팀 합의
후 결정 기록으로 확정한다.

## 1. 전제

- 기간은 **2주(작업일 10일, D1–D10)**, 인원은 5명이다. 남은 M2–M6의 학습·검증 작업과
  G2–G6 판정까지 모두 이 기간에 끝낸다.
- 코드는 M6까지 있다. 부족한 것은 실행과 근거이고, 실행 전에 고쳐야 할 결함이 있다(§4).
- 학습 연산은 CPU 위주다. PPO v2 MLP 1M step이 MacBook Air에서 약 17분, E1–E6 전체가
  순차 약 20시간이다([EXPERIMENT_PLAYBOOK](../EXPERIMENT_PLAYBOOK.md) §1). 정책망이 작고
  병목이 Python 환경 step이라 GPU 이득은 작을 것이다(추정). 원격 장비는 vCPU 수로 고른다.
- Gazebo 쪽은 `opticore-amr` 서버 진단(2026-10-13)대로 4대 실행 시 느리고 namespace 아래에서
  YAML이 적용되지 않는다. 이 계획은 **단일 로봇** 실행을 기준으로 한다.

## 2. 선택지 비교

| | A. 그대로 따라가기 | B. 다시 만들기 | C. 새 방향 | **D. 검증하며 인수 (추천)** |
|---|---|---|---|---|
| 내용 | 문서의 명령대로 E1–E6, M5, M6 실행 | 표준 라이브러리(SB3 등)로 학습 스택 재구현, 구조 재설계 | spike에서 배운 것을 바탕으로 다른 연구 질문 선택 | spike 코드를 쓰되 실행 전 결함을 고치고, 소유자가 검증하며 인수한 뒤 실행 |
| 2주 안 완료 | 가능 | 불가능 | 불확실 | 가능 |
| 장점 | 가장 빠르다 | 코드 이해도가 가장 높다 | 연구 질문을 새로 고를 수 있다 | 속도를 유지하면서 결과를 방어할 수 있다 |
| 위험 | §4의 결함이 결과를 오염시킨다. G4가 너무 쉬워 결론을 방어할 수 없다 | 이미 있는 parity·평가 자산을 다시 만든다 | 남은 자산을 버리고 근거 0에서 다시 시작한다 | 첫 2–3일을 수정과 결정에 써야 한다 |
| 결과물 | 수치는 나오지만 신뢰할 수 없다 | 2주 뒤에도 학습 결과가 없을 가능성이 크다 | 탐색 단계로 돌아간다 | gate 판정과 Lite↔Gazebo 근거 |

### 추천: D. 검증하며 인수

- **버리지 않는다.** 관측 계약, golden parity, 평가 통계, sweep, bundle, ROS core는 2주
  안에 다시 만들 수 없고 구조도 타당하다.
- **그대로 돌리지도 않는다.** 인수인계 검토에서 학습·평가·Gazebo 기록을 오염시키는 결함을
  16개 찾았다(§4). 그대로 E1–E6와 M6를 돌리면 수치가 나와도 쓸 수 없다.
- **평가 전제부터 고친다.** G4와 선정이 Shield OFF baseline과 비교하는 문제(VERIFICATION_STATUS
  §4.1)를 학습 전에 바로잡는다.
- **연구 질문을 동적 장애물로 좁힌다.** 정적 주행은 규칙이 이미 상한이다(SPIKE_REVIEW
  L2). 주장은 "Shield ON 배포 구성에서 동적 시나리오 성공률·도달 시간이 rule보다 낫고
  안전은 비열등하며, 같은 bundle이 Gazebo에서 DWA fallback과 함께 동작한다"로 한다.
- **PPO 채택은 결과로 정한다.** G4를 통과하지 못하면 rule 또는 BC·DAgger를 후보로 두고
  M5·M6를 계속한다. bundle export는 BC도 지원한다.

## 3. D1에 내릴 결정

학습 결과를 무효로 만드는 결정이므로 D1에 모두 정한다. 제안값은 출발점이다.

| # | 결정 | 제안 | 근거 |
|---|---|---|---|
| 1 | G4·선정의 shield 구성 | 비열등성·개선·학습 중 best 선정·sweep 순위를 모두 **Shield ON(scan)**, 기준선은 rule+Shield ON. Shield OFF는 정책 품질 지표로 보고만 한다 | 배포는 Shield ON. 학습도 Shield ON이다. Shield OFF rule은 미등록 장애물을 피하지 않는다 |
| 2 | saturation 상한 | 절대값 5% 대신 "rule+Shield ON 이하"로 판정하거나, G2에서 shield를 튜닝한 뒤 5%를 유지 | rule도 Shield ON saturation 12.9% |
| 3 | rule baseline 수정 | `clearance_stop`·`clearance_slow`를 footprint 기준으로 고친 새 baseline 설정을 만들고 v3 protocol의 기준선·교사·residual base로 쓴다. M1용 기존 값은 그대로 둔다 | 지금 값은 장애물이 footprint 안에 들어온 뒤에야 멈춘다([SENSORS_AND_SAFETY](../internals/SENSORS_AND_SAFETY.md) §8-1) |
| 4 | scan tracker 정적 필터 | Lite `static_filter_distance` 0.3. ROS는 `/map` 필터 추가 또는 차이를 한계로 기록 | 0.0에서 벽이 가짜 동적 track이 되어 정적 시나리오에서도 개입한다(같은 문서 §8-4) |
| 5 | 수동 충돌 분리 | 충돌 시 로봇 속도를 기록하고, 정지 중 피격은 별도 집계 | 비반응형 장애물에 정지 상태로 받히는 경우를 정책 실패와 구분 |
| 6 | 학습 중 선정 seed | 학습 중 best 선정에 sweep selection과 겹치지 않는 seed(예: 25000+)를 쓴다 | 지금은 20000–20001을 둘 다 쓴다([LEARNING](../internals/LEARNING.md) §8-6) |
| 7 | `v_max` | 1.5 m/s 유지 | DWA와 같다. 바꾸면 profile·학습·ROS를 함께 바꿔야 한다 |
| 8 | `angular_accel_max` | 1.5 rad/s²로 학습 시작, D3까지 Gazebo 실측. 차이가 크면 E6만 다시 학습 | E1–E5는 상대 비교라 작은 차이에 둔감하다 |
| 9 | 후진 후보 | 기본 off 유지. G2 stress 결과로 D3에 최종 결정 | 2026-10-13 smoke에서 정적 충돌·timeout 증가 |
| 10 | `goal_tolerance` | 학습·평가는 0.25 m 유지, ROS는 bundle 값 사용 | 바꾸면 protocol version 증가 |
| 11 | `allow_recovery` | `learned` 모드에서 true | false면 시작 handover까지 막혀 learned 모드가 DWA와 같아진다([DEPLOY_AND_ROS2](../internals/DEPLOY_AND_ROS2.md) §8-7) |
| 12 | `opticore-amr` launch 변경 | 내비게이션 노드(A*·DWA)를 끄는 옵션을 `robot.launch.py`에 추가하는 PR을 그쪽 저장소에 낸다 | 지금은 learned 패키지를 붙일 단일 로봇 bring-up이 없다(같은 문서 §4.9) |
| 13 | G6 반복 횟수 | `warehouse_v1` 8 leg × 3회 × (DWA, 후보) | ROADMAP M6 최소 기준 |

결정 1–6은 test split을 열기 전의 변경이므로 결과 오염이 없다. 기존 v2 파일은 두고
`evaluation_selection_v3.yaml`, `evaluation_test_v3.yaml`, 새 gate 파일을 만든 뒤 결정 기록을
남긴다([EVALUATION_AND_EXPERIMENTS](../internals/EVALUATION_AND_EXPERIMENTS.md) §7.3).

## 4. 실행 전에 고칠 것

인수인계 검토에서 찾은 결함 중 결과를 오염시키는 것만 골랐다. 각 항목의 근거는 해당 internals
문서의 "주의할 점과 알려진 한계" 절에 있다. "막는 작업"이 시작되기 전에 PR로 병합한다.

| # | 결함 | 막는 작업 | 담당 | 근거 |
|---|---|---|---|---|
| F1 | G4·선정이 Shield OFF baseline 기준 | E1–E6 | R1 | EVALUATION §8.1 |
| F2 | rule baseline이 footprint 안에서야 정지 | E1–E6 (교사·residual base·기준선) | R3 | SENSORS_AND_SAFETY §8-1 |
| F3 | scan tracker가 벽을 동적 track으로 만듦 | E1–E6, G2 | R3 | SENSORS_AND_SAFETY §8-4 |
| F4 | PPO-Lag λ 갱신과 `step_cost`에 truncation bootstrap이 섞임 | E3 | R2 | LEARNING §8-1 |
| F5 | resume 시 `training.csv` 중복, `best.pt`가 더 나쁜 checkpoint로 덮일 수 있음 | 장시간 학습 전부 | R2 | LEARNING §8-2 |
| F6 | smoke와 실제 sweep이 같은 기본 출력 폴더를 쓰고, 완료 판정이 smoke·config 변경을 보지 않음 | 모든 sweep | R1 | EVALUATION §8.3 |
| F7 | sweep이 manifest(commit·source hash·runtime)를 남기지 않음 | 결과 보존 | R1 | EVALUATION §8.4 |
| F8 | 성공 episode에도 행동 이상 label이 붙고, gate check 일부가 조용히 생략됨 | G4 판정 | R1 | EVALUATION §8.5–8.6 |
| F9 | `opticore-amr`에 learned 패키지를 붙일 단일 로봇 bring-up이 없음 | M5 | R4·R5 | DEPLOY_AND_ROS2 §4.9 |
| F10 | namespace 아래 fallback DWA·A*에 YAML이 적용되지 않음 | M5 (namespace 사용 시) | R5 | DEPLOY_AND_ROS2 §8-8 |
| F11 | `profile_check: warn`이 shadow로 제한되지 않음 | M5 learned | R4 | DEPLOY_AND_ROS2 §8-5 |
| F12 | G5의 "학습 정책–DWA 명령 차이"를 기록하는 노드가 없음 | G5 | R4 | DEPLOY_AND_ROS2 §8-10 |
| F13 | recorder가 이전 goal의 `REACHED`를 새 episode 성공으로 기록 | M6 | R5 | DEPLOY_AND_ROS2 §8-1 |
| F14 | recorder 위치(odom)와 goal(map) 좌표계 혼용, 건너뛴 leg가 CSV에서 빠짐 | M6 | R5 | DEPLOY_AND_ROS2 §8-2·8-3 |
| F15 | `paired_runner`가 namespace에서 goal을 보내지 못함 | M6 (namespace 사용 시) | R5 | DEPLOY_AND_ROS2 §8-4 |
| F16 | 창고 grid map ray cast가 수렴하지 못하면 짧은 거리를 냄(scan의 6.75%) | E1–E6의 `WH-*` 시나리오 | R3 | ENVIRONMENT §8-4 |

F6은 코드를 고치기 전이라도 "smoke는 항상 별도 `--output`"으로 운영하면 피할 수 있다. F12는
코드 대신 rosbag 기록으로 대체할 수 있다.

## 5. 역할

| 역할 | 담당 영역 | 기준 문서 | 책임 gate |
|---|---|---|---|
| R1 평가·실험 리드 | protocol v3, G4 개정, sweep 운영, 통계, 최종 보고서 | [EVALUATION_AND_EXPERIMENTS](../internals/EVALUATION_AND_EXPERIMENTS.md) | G3, G4 판정 |
| R2 학습 | PPO v2·DAgger, E1–E6 실행, 연산 장비 관리 | [LEARNING](../internals/LEARNING.md) | G4 후보 |
| R3 안전·환경 | rule baseline·scan shield 수정, D0–D4 stress, randomization, profile parity | [SENSORS_AND_SAFETY](../internals/SENSORS_AND_SAFETY.md), [ENVIRONMENT](../internals/ENVIRONMENT.md) | G2 |
| R4 ROS2 통합 | `amr_learned_planner` 빌드, dwa → shadow → learned, fault injection | [DEPLOY_AND_ROS2](../internals/DEPLOY_AND_ROS2.md) | G5 |
| R5 Gazebo 검증 | `opticore-amr` 단일 로봇 실행, profile 실측, recorder 수정, paired 실행, gap report | [DEPLOY_AND_ROS2](../internals/DEPLOY_AND_ROS2.md), `opticore-amr/docs/server-run-prescription.md` | G6 |

## 6. 2주 계획

연산 병목을 피하려고 **sweep 단위로 장비를 나눈다.** sweep CLI는 한 출력 디렉터리 안에서
`--parallel`로만 병렬화하므로, 서로 다른 sweep을 서로 다른 장비에서 돌린다.

```text
             D1   D2   D3   D4   D5 │ D6   D7   D8   D9   D10
결정·수정      ██   ██   █          │
G3 동결            ██               │
E1·E2                ██████████     │
E3·E4·E5                     ███████│██
E6·test·G4                          │  ██ ██
G2 stress          ████ ██          │
ROS 빌드·dwa  ███ ███ ██             │
shadow·fault             ██████████ │
learned·G5                          │       ██ ██
Gazebo 기동·실측 ███ ███ ██          │
recorder 수정      ███ ███           │
DWA paired                ████ █████│
후보 paired·G6                       │            ██ ██ ██
보고서                                │                  ████
```

### 1주차 — 결정, 수정, 학습 시작, Gazebo 기동

| 일 | R1 평가·실험 | R2 학습 | R3 안전·환경 | R4 ROS2 | R5 Gazebo |
|---|---|---|---|---|---|
| D1 | §3 결정 회의 진행. CI(두 테스트 스위트) 추가. F1 착수 | 연산 장비 확보, vCPU별 처리량 측정. F4·F5 착수 | F2·F3·F16 착수, 수동 충돌 계측(결정 5) | opticore-amr 컨테이너에 패키지 연결·`colcon build`. F9 PR 작성 | 단일 로봇 Gazebo 기동, RTF·`GL_RENDERER` 확인 |
| D2 | F1·F6·F7·F8 병합, v3 protocol·gate 동결(**G3**), 결정 기록 | F4·F5 병합, teacher dataset 수집 | F2·F3·F16 병합, D0–D4·PD stress 시작 | F9로 단일 로봇 bring-up, `mode:=dwa`에서 A* 재계획·`GOAL_REACHED` 흐름 비교 | `angular_accel_max`, 감속 실측. F13·F14 착수 |
| D3 | E1·E2 시작 확인, `training.csv` 이상 징후 점검 | **E1·E2 시작**(장비 2대) | **G2 판정**: shield 전략, 후진, randomization 범위 | Phase 0 bundle(`profile_check:=warn`)로 `shadow` 실행, F11 | 실측값과 `opticore_gazebo` 비교. F13·F14 병합 |
| D4 | E1·E2 중간 점검 | E1·E2 진행, 예산 재계산 | randomization 강도별 rule 성능 곡선 | fault injection(NaN, slow inference, stale scan) | `warehouse_v1`을 Gazebo에서 DWA로 실행 시작 |
| D5 | E2 select, `SELECTION.md`로 base 설정 확정 | **E3·E4·E5 시작**(장비 3대) | Lite paired 실행(rule·BC) | shadow 30 goal 이상, deadline miss 측정, 명령 차이 rosbag(F12) | DWA paired 계속(경로당 3회) |

### 2주차 — 후보 선정, 실기동, 판정

| 일 | R1 평가·실험 | R2 학습 | R3 안전·환경 | R4 ROS2 | R5 Gazebo |
|---|---|---|---|---|---|
| D6 | E3–E5 select. E6 `variants.candidate` 확정·commit | **E6 시작**(5 seed × 5M, 장비 분산 또는 `--parallel`) | 후보 설정의 Shield ON 개입률·saturation 사전 점검 | 중간 후보 bundle(E2 best)로 shadow 재실행 | DWA paired 완료, 실패 taxonomy 정리 |
| D7 | E6 select, **test split 1회 개봉, G4 판정** | 채택 bundle export·`verify_bundle` | G4 실패 시 대안 후보(rule·BC·DAgger) 비교표 | 채택 bundle을 `learned` 모드로 단일 goal 시험 | recorder CSV와 Lite schema 대조 |
| D8 | G4 보고서 작성 | 학습 곡선·seed 안정성 그래프 | Lite에서 후보 paired 실행 | `learned` 모드 fallback·복귀 시험, **G5 판정** | 후보 paired 실행 시작 |
| D9 | gap report 생성(Lite↔Gazebo, DWA↔후보) | 실험 결과를 `experiments/m4_learned_planner/`에 보존 | 실패 사례 분석 | 운영 절차·rollback 문서 확인 | 후보 paired 완료 |
| D10 | **G6 판정**, 최종 보고서 | 보고서 학습 절 | 보고서 안전 절 | 보고서 통합 절, release note | 보고서 Gazebo 절 |

E6 5 seed × 5M step은 MacBook Air 기준 seed당 약 90분이다(EXPERIMENT_PLAYBOOK §1의 1M ≈ 17분으로 계산).
D6 오후에 시작하면 장비 3대 이상에서 D7 오전에 끝난다.

### 매일 지킬 것

- 아침 15분 sync: 각자 어제 결과, 오늘 할 일, 막힌 것을 공유한다.
- 결과는 sweep 디렉터리와 결정 기록으로 남긴다. 말로만 공유한 수치는 보고서에 쓰지 않는다.
- smoke는 항상 별도 `--output`으로 돌린다(F6).
- test split은 D7에 sweep당 한 번만 연다.
- 실패한 실험도 설정과 원인을 기록한다.

## 7. 축소 기준

막히면 아래 순서로 범위를 줄인다. 줄인 내용은 보고서의 한계 절에 남긴다.

| 상황 | 시점 | 축소 |
|---|---|---|
| F1–F5가 D2까지 병합되지 않음 | D2 | 병합된 것만 반영해 E1·E2를 시작하고, 미반영 항목은 E6 전에 반영. F4가 없으면 E3 제외 |
| Gazebo 단일 로봇이 D3까지 안정적으로 돌지 않음 | D3 | M6를 Lite paired와 shadow 기록으로 대체. Gazebo는 G5까지만 |
| E1–E5가 D6까지 끝나지 않음 | D6 | 끝난 sweep만으로 E6 설정을 고르고 E6 seed를 3개로 줄임 |
| E6가 D7 오전까지 끝나지 않음 | D7 | E2 best 설정의 3 seed를 후보로 test |
| PPO 후보가 G4 실패 | D7 | rule·BC·DAgger 중 G4 기준이 가장 나은 정책을 후보로 M5·M6 진행. "PPO 미채택"을 결론으로 보고 |
| `learned` 모드에서 fallback 전환이 불안정 | D8 | shadow 결과만 보고하고 G5 보류 |

## 8. 2주 뒤 결과물

- 결정 기록: G2–G6 판정과 근거, §3의 결정
- `experiments/m2_.../`, `m4_learned_planner/`, `m6_gazebo_paired/`: sweep spec, leaderboard,
  `SELECTION.md`, `GATE_REPORT.md`, paired·gap report, 대표 그래프, manifest
- 채택 bundle의 `bundle.json`과 checksum, 또는 PPO 미채택 결론
- Gazebo 실행 절차, rollback(`mode:=dwa`) 확인 기록
- 최종 보고서: 연구 질문별 결론, 실패 taxonomy, 허용 운용 범위, 알려진 한계
