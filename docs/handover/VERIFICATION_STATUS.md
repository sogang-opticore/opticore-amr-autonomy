# 검증 현황 — 무엇이 어디까지 확인됐나

이 문서는 spike에서 만든 구성요소마다 **어떤 근거가 실제로 있는지**를 정리한다.
"구현됨"과 "검증됨"을 구분하는 것이 목적이다. 구현 상세는
[internals](../internals/README.md), spike에서 배운 것은 [SPIKE_REVIEW](SPIKE_REVIEW.md)를
따른다. 기준일은 2026-10-14, 기준 commit은 `4e83d29`(`dev`)다.

## 1. 근거 수준 (claude 제안사항)

| 수준 | 의미 | 인용 가능 범위 |
|---|---|---|
| E0 | 코드만 있다. 실행 기록이 없다 | 인용 불가 |
| E1 | 단위·배선·parity 테스트가 통과한다 | "계약·경로가 의도대로 연결됐다" |
| E2 | end-to-end smoke가 끝까지 돈다 | "파이프라인이 실행된다". 성능 수치는 의미 없음 |
| E3 | 비공식 측정이 있다. 표본이 작거나 protocol 밖이다 | 방향성 참고. 결론 근거로 쓰지 않음 |
| E4 | 사전 등록 protocol의 공식 측정이 있다. CI·seed 분산 포함 | 결과로 인용 가능 (해당 환경 한정) |
| E5 | 목표 환경(ROS2/Gazebo)에서 확인했다 | 연구 질문에 대한 최종 근거 |

현재 E4는 M1(`experiments/m1_reproducibility/`) 하나뿐이고 E5는 없다.

## 2. 구성요소별 현황

### 2.1 AMR-Lite 환경과 센서

| 구성요소 | 수준 | 근거 | 빠진 것 |
|---|---|---|---|
| 고정 시나리오 S0–S7, D0–D4, kinematic 동역학, substep 충돌 | E4 | M1 4,160 episode, `test_collision.py`, `test_dynamics.py` | 동적 장애물이 scripted·비반응형이다 |
| contract LiDAR(72 ray)와 관측 `amr-lite-obs-v1` | E4 | M1 전체가 이 관측으로 학습·평가됨, `test_observation.py`, golden vector | — |
| raw LiDAR(720 sample) 시뮬레이션과 `resample-v1` | E1 | `test_lidar_resample.py` 7개 | 실제 Gazebo `LaserScan`을 넣어본 적 없음 |
| robot profile `opticore_gazebo` | E1 | `test_sensor_model.py` | `angular_accel_max` 1.5 등 Gazebo 실측 없음([NEXT_STEPS](../NEXT_STEPS.md) 질문 2) |
| 센서·동역학 randomization | E1 | `test_sensor_model.py` 11개(seed 재현성, 지연·hold·lag) | 강도별 성능 저하 곡선 없음(M2 작업) |
| 절차적 family(P-*, PD-*), warehouse grid map | E1 + E3 | `test_procedural.py` 16개, `test_grid_map.py` 6개, selection v2 85 episode 실행 | split 범위가 겹친다. grid ray cast가 scan의 6.75%에서 짧은 거리를 낸다([ENVIRONMENT](../internals/ENVIRONMENT.md) §8-4, §8-7) |
| vectorized ray casting | E1 | `test_vectorized_cast_is_bit_identical_to_reference` | — |
| 기본 설정 결과 보존(legacy parity) | E1 | `test_legacy_parity.py` 4개 rollout | — |

### 2.2 계획·안전 계층

| 구성요소 | 수준 | 근거 | 빠진 것 |
|---|---|---|---|
| A*, 경로 투영, local goal | E4 | M1, `test_astar.py`, `test_path_utils.py` | — |
| `lite_rule_baseline` | E4 (Lite contract) / E3 (Gazebo profile) | M1, selection v2 85 episode | 정지 임계값이 footprint 안쪽([SENSORS_AND_SAFETY](../internals/SENSORS_AND_SAFETY.md) §8-1). 실제 `dwa_node`(B0)와의 비교 없음 |
| legacy shield (정답 지도·장애물 사용) | E4 | M1 Shield ON 결과 | ROS에서 쓸 수 없음. SLOW·STOP이 감속 한계를 무시해 결과가 낙관적이다 |
| predictive shield | E3 | Phase 0 `shield_v2_results`, `test_safety_shield.py` | 공식 평가 없음 |
| scan shield + `ScanObstacleTracker` | E1 + E3 | `test_scan_shield.py` 9개, 2026-10-13 smoke, 2026-10-14 진단(§4) | D0–D4 stress 평가와 G2 판정 없음. 기본 설정에서 벽이 가짜 track이 되고 FP/FN 열은 항상 0이다 |
| 후진 후보(`reverse_enabled`) | E3 | 2026-10-13 smoke 관찰(ROADMAP M2) | 공식 비교 없음, 기본 off |

### 2.3 학습

| 구성요소 | 수준 | 근거 | 빠진 것 |
|---|---|---|---|
| teacher 수집, BC | E4 | M1 BC 83.1% | — |
| DAgger loop(`dagger_loop.py`, 위험 가중치) | E2 | `test_dagger.py`, smoke | 실제 반복 학습 없음(E1 실험) |
| M1 PPO(`learning/train_ppo.py`, 동결) | E4 | M1 5 seed | 외부 구현(SB3 등)과 수치 비교 없음 |
| PPO v2(direct/residual, PPO-Lag, CNN, curriculum) | E1 + E2 | `test_ppo_v2.py` 8개, smoke | **1M step 이상 학습이 한 번도 없음**. λ 갱신 버그, resume 시 기록 중복([LEARNING](../internals/LEARNING.md) §8) |
| residual 초기 행동 = rule | E1 | `test_residual_policy_starts_as_the_rule_baseline` | — |
| sync·subprocess vector env 동일성 | E1 | `test_subprocess_and_sync_vector_envs_are_identical` | — |
| 처리량 | E3 | [EXPERIMENT_PLAYBOOK](../EXPERIMENT_PLAYBOOK.md) §1 (MacBook Air) | 서버·RunPod 측정 없음 |

### 2.4 평가·실험 운영

| 구성요소 | 수준 | 근거 | 빠진 것 |
|---|---|---|---|
| paired 평가, 계층적 bootstrap CI | E4 | M1, `test_metrics.py` | — |
| protocol v2 (selection/test) | E3 | selection 85 episode 비공식 실행 | G3 동결 전. test split 미개봉 |
| sweep runner(전개·병렬·재개·선정·1회 test) | E2 | `test_gate_and_sweep.py`, Gate 7 smoke | 실제 sweep 없음. 재개·`--parallel` > 1 미검증, smoke와 출력 폴더 공유, manifest 없음 |
| G4 gate | E1 + E2 | `test_gate_and_sweep.py`, smoke | **기준 자체의 타당성 문제**(§4). 성공 episode의 행동 label 집계, 일부 check의 조용한 생략 |
| manifest provenance | E4 | M1 `manifest.json` | — |

### 2.5 배포와 ROS2

| 구성요소 | 수준 | 근거 | 빠진 것 |
|---|---|---|---|
| bundle export, NumPy runtime | E1 + E2 | `test_deploy.py` 7개(BC·PPO·residual·CNN), Gate 8 | 학습된 후보 bundle 없음 |
| `PlannerCore`(관측·추론·shield·상태 머신) | E1 | `test_planner_core.py` 11개, Lite 폐루프 도착 | 실제 ROS 메시지·타이밍 |
| Lite↔ROS 관측 parity `1e-6` | E1 | `test_core_observation_matches_lite_environment` 3개 시나리오 | 입력이 Lite가 만든 scan이다. 실제 Gazebo scan 아님 |
| `cmd_arbiter` 모드·fallback·stale | E1 | `test_arbiter_recorder.py` 8개 | 전환 순간의 실제 차량 거동 |
| rclpy 노드 4개 | E1 | 가짜 rclpy(`test/fake_ros.py`) 배선 테스트 5개 | **실제 rclpy·colcon build·launch 실행 없음** |
| launch(`learned_nav`, `paired_runner`) | E1 | helper 테스트 2개 | 실제 launch 없음 |
| Gazebo paired(`warehouse_v1`)와 gap report | E2 (Lite 쪽) / E0 (Gazebo 쪽) | `test_paired.py`, Gate 8 | Gazebo 실행 없음. recorder의 `REACHED` 오판·좌표계 혼용·누락 leg([DEPLOY_AND_ROS2](../internals/DEPLOY_AND_ROS2.md) §8) |

## 3. 테스트 지도

AMR-Lite 119개, ROS 통합 26개, 총 145개다. 테스트는 대부분 **계약·배선·재현성**을
지킨다. 정책이 잘 주행하는지는 테스트 대상이 아니라 평가(protocol) 대상이다. 영역별로
테스트하지 않는 코드 경로는 각 [internals](../internals/README.md) 문서 §6 끝에 있다.

ROS 컨테이너(실제 rclpy, torch 없음)에서 `colcon test`를 돌리면 가짜 rclpy 테스트와 bundle
테스트가 skip되어 26개 중 약 8개만 실행될 것으로 본다(추정, [DEPLOY_AND_ROS2](../internals/DEPLOY_AND_ROS2.md) §6).

| 영역 | 파일(개수) | 지키는 것 |
|---|---|---|
| 동역학·충돌 | `test_dynamics`(4), `test_collision`(2), `test_action`(2) | 운동학 적분, 각도 정규화, 고속 tunneling 방지, 가감속 제한, 후진은 요청 시에만 |
| 환경·재현성 | `test_environment`(2), `test_reproducibility`(4), `test_legacy_parity`(4) | reward 항 존재, 결정적 reset/step, seed namespace 분리, 기본 설정 rollout 불변 |
| 지도·ray | `test_raycast`(3), `test_grid_map`(6), `test_astar`(3), `test_path_utils`(2) | vectorized ray bit 동일, 거리장 정확도, PGM 읽기, 지도 SHA 고정, 경로 투영 |
| 센서 | `test_lidar_resample`(7), `test_sensor_model`(11), `test_observation`(1), `test_observation_golden`(1) | 재표본화 규칙, profile override, randomization seed·범위, 관측 계약·golden |
| 시나리오 | `test_procedural`(16), `test_warehouse_scenarios`(4) | family별 재현성·A* 해 존재, split 생성 규칙 분리, agent 이동 패턴, 창고 leg 좌표 |
| 안전 계층 | `test_safety_shield`(5), `test_scan_shield`(9) | 벽 앞 감속·정지, NaN 정지, 접근 예측, tracker 정적 필터, 후진 후보 조건 |
| 학습 | `test_ppo_v2`(8), `test_vector_env`(5), `test_dagger`(3), `test_smoke_pipeline`(1) | residual 항등, GAE 경계, checkpoint 저장·재개, vector env 동일성, curriculum, DAgger β |
| 평가·실험 | `test_metrics`(2), `test_gate_and_sweep`(5), `test_paired`(2) | CI 재현성, gate PASS/FAIL 경로, 선정 순서, test 1회 제한, paired schema |
| 배포 | `test_deploy`(7) | torch↔NumPy 출력 일치, bundle 변조 거부, torch 미import |
| ROS core | `test_planner_core`(11) | Lite↔ROS 관측 parity, 폐루프 도착, 상태 어휘, fault 주입, golden |
| ROS 노드 | `test_arbiter_recorder`(8), `test_nodes_fake_ros`(5), `test_launch_helpers`(2) | arbiter 모드·hold·복귀, stale·NaN 처리, recorder 판정, 가짜 rclpy 배선, namespace remap |

## 4. 2026-10-14 재확인 기록

이번 인수인계 검토에서 다시 실행한 결과다. 산출물은 `/tmp` 아래에 두었고 저장소에는
넣지 않았다.

| 항목 | 결과 |
|---|---|
| `pytest -q` (AMR-Lite) | `119 passed`, 105.6 s (MacBook Air). [RUNBOOK](../RUNBOOK.md) Gate 1의 "약 6초"는 M2–M6 이전 값이다 |
| `pytest -q integrations/ros2/amr_learned_planner/test` | `26 passed`, 6.5 s |
| RUNBOOK Gate 7·8 전체 (PPO v2·DAgger·sweep smoke, export·verify·paired) | 모든 명령 exit 0, test 재실행은 거부(exit 1), 합계 44 s. smoke G4 판정은 FAIL(정상, 성능 아님) |

### 4.1 rule baseline의 동적 충돌 진단

protocol v2 selection split(17 시나리오 × 5 seed = 85 episode, `opticore_gazebo`,
`scan` shield)에서 `lite_rule_baseline`을 실행하고, 동적 충돌 순간의 로봇 선속도를
기록했다. 비공식 1회 측정(E3)이다.

| Shield | 성공 | 동적 충돌 | 충돌 시 로봇 속도 | timeout | `SHIELD_SATURATION` |
|---|---:|---:|---|---:|---:|
| OFF | 55 | 30 | 30건 모두 ≥ 0.2 m/s | 0 | 0 |
| ON | 59 | 6 (D1 5, PD-MIXED 1) | < 0.05 m/s 3건, < 0.2 m/s 2건, 그 외 1건 | 9 | 11 (12.9%) |

- Shield OFF에서 D1–D4와 PD-MIXED는 5/5 모두 충돌했다. 정지 장애물인 D3도 포함된다.
  rule baseline은 shield 없이는 지도에 없는 장애물을 피하지 않고 주행 중에 부딪힌다.
- `configs/gates/g4.yaml`의 비열등성·개선·robustness 기준은 모두 Shield OFF에서
  `lite_rule_baseline`과 비교한다. 따라서 LiDAR에 반응하기만 하는 정책은 "동적
  시나리오 성공률 +5%p" 기준을 쉽게 넘는다.
- 배포 구성은 Shield ON(`learned_planner.yaml`의 `shield_enabled: true`)이다. Shield ON
  에서 rule baseline의 saturation 12.9%는 이미 G4 상한 5%를 넘는다.
- 학습과 선정도 같은 축에서 어긋난다. PPO v2는 Shield ON으로 학습하고(`ppo_v2.yaml`)
  학습 중 best 선정과 sweep 순위는 Shield OFF로 한다(`evaluation_selection_v2.yaml`).
- test split을 아직 한 번도 열지 않았으므로 지금 바꾸면 결과 오염이 없다. 파일 주석 규칙대로
  기존 v2를 두고 v3 protocol과 새 gate 파일을 만든다
  ([EVALUATION_AND_EXPERIMENTS](../internals/EVALUATION_AND_EXPERIMENTS.md) §7.3). (claude 제안사항)

재현 스크립트(`packages/amr_lite`에서 `PYTHONPATH=.`로 실행):

```python
import collections
import numpy as np
from amr_lite.config import load_config, resolve_env_config
from amr_lite.envs import WarehouseEnv
from amr_lite.planning.rule_baseline import RuleBasedPlanner

cfg = load_config("evaluation_selection_v2")
base = resolve_env_config(cfg["robot_profile"], cfg.get("environment_overrides"))
seeds = [int(cfg["seed"]) + i for i in range(int(cfg["episodes_per_scenario"]))]
for shield in (False, True):
    stats = collections.Counter()
    for scenario in cfg["scenarios"]:
        for seed in seeds:
            local = dict(base, shield_enabled=shield)
            env = WarehouseEnv(local, scenario)
            obs, _ = env.reset(seed=seed, options={"scenario_id": scenario})
            policy, info = RuleBasedPlanner(local), {}
            for _ in range(int(cfg["max_steps"])):
                obs, _, terminated, truncated, info = env.step(np.asarray(policy.predict(obs), dtype=np.float32))
                if terminated or truncated:
                    break
            stats[info.get("failure_type") or "SUCCESS"] += 1
            if info.get("failure_type") == "COLLISION_DYNAMIC":
                stats["moving" if abs(env.robot.v) >= 0.2 else "slow_or_stopped"] += 1
    print("shield", shield, dict(stats))
```

## 5. 아직 아무도 확인하지 않은 주장

우선순위 순서다. 이 항목들은 결과 보고에서 "확인됨"으로 쓰지 않는다 (claude 제안사항).

1. 학습 정책이 Shield ON 배포 구성에서 rule baseline보다 낫다 — PPO v2 학습 자체가 없다.
2. PPO v2의 residual·Lagrangian·curriculum이 M1의 best/final 붕괴를 줄인다 — 미실행.
3. custom PPO 구현이 표준 구현과 수치적으로 같다 — 비교 스크립트가 없다.
4. `opticore_gazebo` profile이 실제 Gazebo 로봇과 같다 — 실측 없음.
5. ROS 노드가 실제 rclpy·Humble·Gazebo에서 20 Hz로 동작하고 deadline miss가 0.1% 이하다.
6. arbiter fallback이 실제 차량에서 급회전 없이 전환된다.
7. Lite 결과가 Gazebo로 전이된다(M6 paired) — Gazebo 쪽 실행 없음.
8. 학습 정책이 실제 `dwa_node`보다 낫다 — `dwa_node`와의 직접 비교는 Gazebo에서만 가능하다.
9. 절차적 test split이 학습 분포 밖이다 — 파라미터 범위가 겹치고 `WH-RANDOM`은 split과 무관하다.
10. `failure_type`이 정책 책임을 나타낸다 — 성공에도 label이 붙고 정지 중 피격을 구분하지 않는다.
