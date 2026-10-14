# Next Steps and Team Decisions

마일스톤과 gate는 [ROADMAP.md](ROADMAP.md)가 기준이다. 이 문서는 (1) Phase 1 후보
목록 중 이미 코드로 구현된 항목, (2) 마일스톤 코드 밖에 남은 작업, (3) 팀이 정해야
할 질문을 정리한다.

## 1. Phase 1 후보 중 구현된 항목 (2026-10-13)

| 항목 | 구현 위치 |
|---|---|
| vectorized ray casting (기존과 bit 단위 동일) | `amr_lite/envs/raycast.py`, `envs/map.py` |
| vector environment | `amr_lite/rl/vector_env.py` (sync·subprocess, curriculum) |
| `warehouse_map.pgm` importer | `amr_lite/envs/grid_map.py`, `assets/maps/opticore_warehouse/` |
| LiDAR noise·latency·dropout·domain randomization | `amr_lite/sensors/randomization.py` |
| 1-frame 대 4-frame ablation 경로 | `history_mode: newest`, sweep `e5_representation` |
| frame stack과 explicit tracker 비교 경로 | `planning/scan_shield.py`의 `ScanObstacleTracker` |
| 제한적 후진 후보 | `scan_shield.reverse_enabled` (기본 off) |
| teacher·DAgger 반복과 위험 상태 가중치 | `learning/dagger_loop.py`, `train_bc.risk_weight` |
| 여러 학습 seed·sweep·사전 등록 선정 | `experiments/sweep.py`, `configs/sweeps/` |
| 비용 제약(PPO-Lagrangian) | `rl/ppo.py`, sweep `e3_lagrangian` |
| ROS adapter와 Lite↔Gazebo paired test 도구 | `integrations/ros2/amr_learned_planner`, `experiments/paired.py` |

## 2. 마일스톤 코드 밖에 남은 작업

### 학습·평가 실행
- M2 shield 비교, M3 처리량 측정, M4 E1–E6 실행 — [EXPERIMENT_PLAYBOOK.md](EXPERIMENT_PLAYBOOK.md)
- M5 ROS2 실기동(dwa → shadow → learned)과 fault injection, M6 Gazebo paired 실행

### 기술 부채·확장 후보
- 검증된 PPO 라이브러리(SB3 등)와 PPO v2의 수치 비교. Gymnasium adapter는 있으나
  비교 스크립트는 없다.
- SAC 등 off-policy 비교: PPO v2가 G4를 통과하지 못할 때만 진행
- GRU/LSTM 정책: bundle runtime과 ROS 노드에 hidden state 관리가 필요
- scan tracker의 정적 지도 필터를 ROS에서도 `/map`으로 적용
- Lite 동적 agent를 사람 이동 모델(목적지 기반 보행, 무리)로 확장
- recorder의 충돌 proxy를 Gazebo contact sensor로 교체
- 차체 직사각형 footprint(0.60 × 0.40 m)를 Lite 충돌·shield에 반영
- `opticore-amr`의 `multi_robot.launch.py`에 learned 노드 묶음을 넣는 변경
  (해당 저장소에서 팀 결정 후)
- CI: 두 테스트 스위트와 Gate 7 smoke를 PR마다 실행

## 3. 팀 결정 질문

1. 배포 robot profile의 `v_max`를 1.5 m/s로 둘 것인가, 명세 2.0 m/s로 올릴 것인가?
   (profile·학습·ROS 파라미터를 함께 바꿔야 한다)
2. `angular_accel_max` 1.5 rad/s²(DWA 설정)를 Gazebo에서 실측해 확정할 것인가?
3. scan shield에 후진 후보를 허용할 것인가? (초기 smoke: 동적 충돌 감소, 정적
   충돌·timeout 증가)
4. 학습 planner의 `goal_tolerance`를 Lite 0.25 m로 둘 것인가, 운영 launch의 DWA 값
   (0.20/0.40 m)에 맞출 것인가?
5. 학습 planner가 DWA의 동적 상태(`DYNAMIC_BLOCKED` 등)를 대신할 신호를 내야 하는가,
   아니면 dynamic layer 발행을 DWA에 계속 맡길 것인가?
6. `learned` 모드에서 fallback 후 자동 복귀(`allow_recovery`)를 허용할 것인가?
7. Gazebo paired 검증의 반복 횟수와 G6 통과 기준(충돌 0건, 성공률 DWA 이상)을 확정할 것인가?
8. G4의 비열등성·개선 판정을 Shield OFF 대신 배포 구성(Shield ON, `scan`)으로 할 것인가? (claude 제안사항)
   Shield OFF의 `lite_rule_baseline`은 지도에 없는 장애물을 피하지 않아 동적 성공률 개선
   기준이 쉽게 통과된다([검증 현황](handover/VERIFICATION_STATUS.md) §4.1).
9. G4의 Shield ON saturation 상한 5%를 유지할 것인가? rule baseline도 protocol v2 selection에서
   12.9%다. (claude 제안사항)
10. 동적 장애물에 정지 상태로 받힌 충돌을 정책 실패와 분리해 집계할 것인가? (claude 제안사항)
11. `lite_rule_baseline`의 정지 임계값(`clearance_stop` 0.12 m < 로봇 반경 0.36 m)을 고친 새 baseline을
    기준선·BC 교사·residual base로 쓸 것인가? M1 재현용 기존 값은 유지한다. (claude 제안사항)
12. learned 패키지를 붙일 단일 로봇 bring-up(내비게이션 노드를 끄는 옵션)을 `opticore-amr`에 추가할 것인가? (claude 제안사항)

제안값과 결정 시점(claude 제안사항)은 [향후 방향](handover/DIRECTION_OPTIONS.md) §3에 있다.
