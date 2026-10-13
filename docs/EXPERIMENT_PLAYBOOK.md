# Experiment Playbook — M3 확인부터 M4 배포 후보까지

이 문서는 이번 구현 범위에서 **실행하지 않은 학습 작업**을 팀이 그대로 수행할 수
있도록 순서, 명령, 예산, 판정 방법을 정리한다. 마일스톤과 gate는
[ROADMAP.md](ROADMAP.md), 계약은 [ARCHITECTURE.md](ARCHITECTURE.md)를 따른다.

모든 명령은 `packages/amr_lite`에서 실행한다.

```bash
cd packages/amr_lite
export PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib
```

## 0. 원칙

1. 학습 전에 protocol v2(`evaluation_selection_v2`, `evaluation_test_v2`)와 선정 규칙,
   G4 기준(`configs/gates/g4.yaml`)을 G3에서 동결한다. 결과를 본 뒤 바꾸지 않는다.
2. 학습 episode seed는 1,000,000부터, selection은 20000+, test는 30000+를 쓴다.
   코드가 강제하므로 seed를 손으로 고르지 않는다.
3. test split은 후보 확정 후 sweep당 **한 번만** 연다. `sweep --stage test`는 두 번째
   실행을 거부한다(`--force`는 실행이 중단된 경우의 복구용).
4. 실패한 실험도 sweep 디렉터리와 `SELECTION.md`를 보존하고 결정 기록에 남긴다.
5. 배포 후보는 `opticore_gazebo` profile(LiDAR raw 재표본화, scan shield)로 학습·평가한다.

## 1. 처리량과 예산 (M3 완료 조건)

장비마다 다시 측정해 예산을 확정한다. 아래는 개발 노트북(MacBook Air, 8 core:
4 performance + 4 efficiency)에서 측정한 값이다.

| 항목 | 측정값 |
|---|---:|
| Lite env 단일 process, `lite_default` (rule) | 약 2,400 step/s |
| Lite env 단일 process, `opticore_gazebo` (raw LiDAR) | 약 1,900 step/s |
| 동일 + scan shield + domain randomization | 약 1,000–1,500 step/s |
| PPO v2 MLP residual, subprocess env 8개, 평가 제외 | 약 960 step/s (1M ≈ 17분) |
| PPO v2 1D-CNN, 같은 조건 | 약 490 step/s (1M ≈ 34분) |
| protocol v2 selection 전체(85 episode), worker 4 | 약 15초 |

효율 core가 섞인 장비에서는 lockstep vector env가 가장 느린 worker를 기다리므로
worker 수를 늘려도 선형으로 빨라지지 않는다. 같은 core의 Linux 서버에서 다시 잰다.

```bash
# 처리량 측정용 짧은 실행 (결과는 artifacts/rl/throughput/result.json의 wall_seconds)
python -m amr_lite.cli.train_rl --output artifacts/rl/throughput \
  --set total_timesteps=32768 --set evaluation.interval_updates=0
```

예산 계산:

```text
trial 시간 ≈ timesteps ÷ (end-to-end step/s) + in-training selection 횟수 × selection 시간
sweep 시간 ≈ Σ trial 시간 ÷ 동시 trial 수 (CPU 수 / env worker 수 이내)
```

위 노트북 기준 E1–E6 전체는 순차 실행으로 약 20시간 규모다(아래 표의 trial 수 기준).

참고 기준: protocol v2 selection split에서 `lite_rule_baseline`(Shield OFF)은 성공
64.7%, 충돌 35.3%(전부 동적 장애물)였다. 85 episode 1회 측정이며 공식 test 결과가
아니다.

## 2. 사전 준비

### 2.1 Smoke로 코드 경로 확인

```bash
pytest -q                                    # 단위·parity·golden 테스트
python -m amr_lite.cli.train_rl --smoke --output artifacts/rl/smoke
python -m amr_lite.cli.dagger --smoke --output artifacts/dagger/smoke
python -m amr_lite.cli.sweep e3_lagrangian --smoke --output artifacts/sweeps/smoke-e3
python -m amr_lite.cli.sweep e3_lagrangian --smoke --output artifacts/sweeps/smoke-e3 --stage test
```

smoke 결과(성공률 등)는 학습 성능이 아니다. 실행 경로와 산출물만 확인한다.

### 2.2 교사 dataset (E2 `direct_bc` 변형용)

```bash
python - <<'PY'
from amr_lite.config import resolve_env_config
from amr_lite.learning.collect_teacher import collect
collect("artifacts/datasets/teacher_opticore_gazebo", episodes=48, max_steps=1500, seed=7,
        scenarios=["S0", "S1", "S2", "S3", "S4", "S5", "P-AISLE", "P-HALL", "P-PINCH", "P-JUNCTION", "WH-RANDOM"],
        env_config=resolve_env_config("opticore_gazebo", {"shield_strategy": "scan"}))
PY
```

## 3. 실험 매트릭스 (M4)

| ID | spec | 비교 | trial 수 | timesteps/trial |
|---|---|---|---:|---:|
| E1 | `configs/sweeps/e1_dagger.yaml` | BC only / DAgger 3회 / 가중치 없는 DAgger | 9 | — |
| E2 | `e2_policy_mode.yaml` | residual 0.5 / 0.3 / direct scratch / direct+BC | 12 | 1M |
| E3 | `e3_lagrangian.yaml` | 제약 없음 / cost limit 0.01 / 0.03 | 9 | 1M |
| E4 | `e4_randomization.yaml` | randomization curriculum / 없음 | 6 | 1M |
| E5 | `e5_representation.yaml` | MLP·CNN × 4-frame·newest-only | 12 | 1M |
| E6 | `e6_candidate.yaml` | G3에서 고른 설정, seed 5개 | 5 | 5M |

권장 순서: E1과 E2를 먼저 돌려 정책 형태(residual/direct)를 정하고, 그 설정을 base로
E3–E5를 진행한다. 각 spec의 `variants`를 바꿨다면 커밋으로 남긴 뒤 실행한다.

### 3.1 실행

```bash
python -m amr_lite.cli.sweep e2_policy_mode --stage train --parallel 2   # 재실행하면 이어서 학습
python -m amr_lite.cli.sweep e2_policy_mode --stage select
```

- trial별 산출물: `artifacts/sweeps/<name>/trials/<variant>-s<seed>/`
  (`trial_config.json`, `training.csv`, `best.pt`, `final.pt`, `train.log`, `selection_*.json`)
- 중단된 trial은 `state.pt`에서 자동 재개된다. 시간 제한이 있는 장비에서는
  `train_rl --max-updates N`으로 끊어서 학습하고 `--resume`으로 잇는다.
- `leaderboard.csv`와 `SELECTION.md`가 사전 등록 규칙(제약 → 성공률 중앙값 → 최악 seed →
  충돌 → 시간)으로 정렬된 결과다.

### 3.2 학습 곡선과 이상 징후

`training.csv`에서 아래를 확인한다.

| 열 | 확인 내용 |
|---|---|
| `success_rate`, `collision_rate`, `timeout_rate` | 학습 rollout 성능(탐색 noise 포함) |
| `eval_*` | 주기적 selection 평가(결정적 정책) — best checkpoint 기준 |
| `approx_kl`, `kl_stop`, `clip_fraction` | 정책 갱신 폭. `kl_stop`이 계속 1이면 학습률을 낮춘다 |
| `step_cost`, `lagrange` | E3: λ가 상한(10)에 붙으면 cost limit이 비현실적 |
| `log_std_mean` | 탐색 폭. 너무 빨리 줄면 entropy 계수를 검토한다 |
| `stage` | curriculum 단계 전환 시점 |

행동 이상(선회·정지·진동)은 selection/test의 `failure_counts`와 G4 behaviour 항목으로
자동 판정된다.

## 4. 배포 후보 선정과 test (G3 → G4)

1. E1–E5의 `SELECTION.md`를 근거로 E6 `variants.candidate`를 채우고 커밋한다(G3).
2. E6 학습과 selection을 실행한다.
3. test split을 한 번 연다.

```bash
python -m amr_lite.cli.sweep e6_candidate --stage train --parallel 2
python -m amr_lite.cli.sweep e6_candidate --stage select
python -m amr_lite.cli.sweep e6_candidate --stage test
```

`artifacts/sweeps/e6_candidate/test/`에 `episodes.csv`, `summary*.csv`,
`paired_comparisons.csv`, `gate.json`, `GATE_REPORT.md`가 생긴다. G4는
`GATE_REPORT.md`의 모든 항목이 통과해야 한다. 실패하면 PPO를 채택하지 않는 것도 정상
결론이며, 원인을 결정 기록에 남긴다.

## 5. Bundle export와 ROS 인계

```bash
python -m amr_lite.cli.export_policy \
  --checkpoint artifacts/sweeps/e6_candidate/trials/candidate-s7/best.pt \
  --output artifacts/bundles/e6-candidate-s7 \
  --selection artifacts/sweeps/e6_candidate/trials/candidate-s7/selection_best.json
python -m amr_lite.cli.verify_bundle artifacts/bundles/e6-candidate-s7 --robot-profile opticore_gazebo
```

`verify_bundle`이 exit 0이고 golden 오차가 `1e-5` 이하여야 한다. 이후 절차는
[amr_learned_planner README](../integrations/ros2/amr_learned_planner/README.md)의
M5 단계를 따른다.

## 6. 결과 보존

마일스톤 결과는 `experiments/m4_learned_planner/`처럼 새 디렉터리에 둔다.

- 보존: sweep spec, `plan.json`, `leaderboard.csv`, `SELECTION.md`, test의 summary·paired
  CSV, `GATE_REPORT.md`, 채택 bundle의 `bundle.json`, 대표 그래프
- 제외: checkpoint, episode 원자료 전체, dataset (SHA-256은 `bundle.json`과 manifest에 남는다)
