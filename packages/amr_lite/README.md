# OptiCore AMR-Lite

ROS2와 Gazebo 없이 실행되는 경량 2D AMR 로컬 플래닝 기술 스파이크다. 동일한 관측·행동 계약에서 `lite_rule_baseline`, Behavior Cloning(BC), PPO, BC+DAgger-lite를 실행하고, 공통 Safety Shield의 효과를 paired evaluation으로 비교한다.

이 구현은 기존 `ros2_ws`와 독립적으로 실행된다. 기존 `dwa_node.py`를 복사하거나 import하지 않으며, ROS 플래너의 제약값과 설계만 참고했다.

## 구현 범위

- differential-drive 운동 모델과 물리 substep 충돌 검사
- 직사각형 정적 장애물 및 scripted 원형 동적 장애물
- 72-ray, 270° 가상 LiDAR와 최근 4-frame stack
- 정적 지도 A* 전역 경로, path projection, arc-length local goal
- Pure Pursuit 기반 `lite_rule_baseline`
- 공통 normalized action 및 Safety Shield
- episode 단위 teacher NPZ, BC, DAgger-lite
- 외부 RL 라이브러리 없이 동작하는 소형 PyTorch PPO
- shield OFF/ON paired evaluation, CSV, 그래프, 최종-frame 데모

## 요구 환경

- Python 3.10 이상
- NumPy, PyYAML, PyTorch, Matplotlib
- 테스트에는 pytest

현재 저장소에서 확인한 `python` 3.11 환경에는 위 패키지가 설치되어 있다. 새 환경에서는 다음과 같이 프로젝트 전용 가상환경을 사용할 수 있다.

```bash
cd packages/amr_lite
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

설치하지 않고 저장소에서 직접 실행할 때는 아래 명령처럼 `PYTHONPATH=.`을 사용한다.

## 가장 짧은 실행

다음 한 줄은 teacher 데이터 생성 → BC smoke → PPO smoke → paired evaluation → 그래프와 데모 생성을 순서대로 실행한다.

```bash
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.train
```

Smoke는 코드 경로와 checkpoint 입출력을 검증하기 위한 실행이다. BC/PPO 수렴을 보장하거나 주장하지 않는다. Smoke 평가·그래프·데모는 `artifacts/smoke/`에 저장되어 전체 평가 결과를 덮어쓰지 않는다.

## 단계별 실행

테스트:

```bash
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib pytest -q
```

Teacher dataset:

```bash
PYTHONPATH=. python -m amr_lite.learning.collect_teacher \
  --episodes 8 --output artifacts/datasets/teacher
```

BC smoke/full:

```bash
PYTHONPATH=. python -m amr_lite.learning.train_bc
PYTHONPATH=. python -m amr_lite.learning.train_bc --full \
  --output artifacts/checkpoints/bc_full.pt
```

DAgger-lite 1회와 재학습:

```bash
PYTHONPATH=. python -m amr_lite.learning.collect_dagger \
  --checkpoint artifacts/checkpoints/bc_smoke.pt
PYTHONPATH=. python -m amr_lite.learning.train_bc \
  --dataset artifacts/datasets/teacher artifacts/datasets/dagger \
  --output artifacts/checkpoints/bc_dagger_smoke.pt
```

PPO smoke/full:

```bash
PYTHONPATH=. python -m amr_lite.learning.train_ppo
PYTHONPATH=. python -m amr_lite.learning.train_ppo --full \
  --output artifacts/checkpoints/ppo_full.pt
```

BC warm start, 고정 평가 callback, best checkpoint를 포함한 PPO 실험:

```bash
PYTHONPATH=. python -m amr_lite.learning.train_ppo --full \
  --warm-start-dataset artifacts/datasets/teacher \
  --output artifacts/checkpoints/ppo_bc_best_full.pt
```

최종 step은 `ppo_bc_best_full.pt`, 학습 중 평가가 가장 좋았던 모델은 `ppo_bc_best_full_best.pt`에 저장된다. 현재 실험에서 best는 RL 업데이트 전 timestep 0이므로, 성능을 `PPO가 향상시켰다`고 해석하면 안 된다.

권장 stable PPO는 actor/critic encoder 분리, value normalization, KL early stop을 추가한다.

```bash
PYTHONPATH=. python -m amr_lite.learning.train_ppo --stable --full \
  --warm-start-dataset artifacts/datasets/teacher \
  --output artifacts/checkpoints/ppo_stable_full.pt
```

현재 stable best는 51,200 timestep에서 선택됐으며 실제 PPO 업데이트 후 모델이다.

고정 seed paired evaluation:

```bash
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.evaluate
```

빠른 세 시나리오 평가에는 `--quick`을 추가한다. `configs/evaluation.yaml`의 `shield_modes`가 `[false, true]`이므로 모든 정책의 raw 결과와 shield 적용 결과가 함께 생성된다.

데모 스크린샷:

```bash
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.demo \
  --policy rule --scenario S3 --output artifacts/videos/rule_S3.png
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.demo \
  --policy bc --checkpoint artifacts/checkpoints/bc_dagger_smoke.pt \
  --scenario D0 --output artifacts/videos/bc_dagger_D0.png
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.demo \
  --policy ppo --checkpoint artifacts/checkpoints/ppo_smoke.pt \
  --scenario D0 --output artifacts/videos/ppo_D0.png
```

`--show`를 추가하면 실행 중 화면을 갱신한다. 헤드리스 환경에서는 최종-frame PNG를 사용한다.

## 관측·행동 계약

관측은 항상 `float32[296]`이다.

```text
scan_t-3[72], scan_t-2[72], scan_t-1[72], scan_t[72],
normalized local-goal distance, sin/cos local-goal angle,
normalized v/w,
normalized signed cross-track error, sin/cos path-heading error
```

LiDAR history 순서는 oldest → newest이며 거리는 `[0, 1]`로 정규화된다. 행동은 `[a_v, a_w] ∈ [-1, 1]^2`이고 다음과 같이 명령으로 변환된다.

```text
v_target = (a_v + 1) / 2 × v_max
w_target = a_w × w_max
```

그 뒤 모든 정책에 같은 가속도 제한과 Safety Shield를 적용한다.

## 시나리오

- `S0` 빈 직선, `S1` 90° 코너, `S2` 좁은 통로
- `S3` 전역 경로 위 정적 장애물, `S4` 장애물 옆 통과
- `S5` 경로에서 벗어난 시작, `S6` 부분 폐쇄 통로, `S7` no-path
- `D0` 횡단, `D1` 정면 접근, `D2` 같은 방향 저속 이동
- `D3` 통로 정지, `D4` 이동 후 갑작스러운 정지

## 설정

주요 설정은 모두 `configs/`에 있다.

- `env.yaml`: 로봇, 물리, LiDAR, 관측, reward, shield
- `baseline.yaml`: Pure Pursuit와 clearance 기반 감속
- `bc.yaml`: MLP, split, batch, epoch
- `ppo.yaml`: smoke/full timestep과 PPO hyperparameter
- `evaluation.yaml`: 시나리오, paired seed, shield mode

기본 로봇 반경 0.36 m, 제어율 20 Hz, lookahead 0.65 m는 기존 ROS 설정을 참고했다.

## 결과물

```text
artifacts/datasets/       episode/chunk NPZ
artifacts/checkpoints/    BC/PPO checkpoint와 학습 로그
artifacts/results/        episodes.csv, summary.csv, config.json
artifacts/plots/          success/collision/time/training 그래프
artifacts/videos/         주행 최종-frame PNG
artifacts/smoke/          통합 smoke 명령 전용 결과
artifacts/full/           full BC/PPO paired evaluation과 데모
artifacts/reward_v2/       path-progress reward PPO 실험
artifacts/ppo_best/        BC warm-start PPO best-checkpoint 평가
artifacts/ppo_stable/      안정화 PPO final/best 평가와 데모
```

위 경로는 새 실험을 실행할 때 이 패키지 아래에 생성되며 Git에서 제외된다. Phase 0에서 선별한 결과표·그래프·대표 데모와 Stable PPO best checkpoint는 [Phase 0 artifacts](../../experiments/phase0_amr_lite/artifacts/README.md)에 보존한다.

측정 결과와 해석은 [Phase 0 결과](../../experiments/phase0_amr_lite/RESULTS.md)와
[M1 재현성 결과](../../experiments/m1_reproducibility/RESULTS.md), 향후 마일스톤은
[ROADMAP.md](../../docs/ROADMAP.md), 설계 결정은
[DECISIONS.md](../../experiments/phase0_amr_lite/DECISIONS.md), 한계는
[KNOWN_LIMITATIONS.md](../../experiments/phase0_amr_lite/KNOWN_LIMITATIONS.md)를 참고한다.
현재 코드와 산출물을 단계별로 확인할 때는 [실행·검증 런북](../../docs/RUNBOOK.md)을 따른다.

## M1 재현 가능한 기준선

다음 명령은 Stable PPO를 학습 seed 5개로 반복하고, 분리된 selection/test seed에서
best와 final checkpoint를 평가한다.

```bash
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.m1
```

결과는 `artifacts/m1/m1-reproducible-baseline-v1/` 아래에 episode 원자료,
학습-seed·시나리오별 요약, 95% CI, paired comparison, manifest와 보고서로 저장된다.
완료된 seed checkpoint는 config와 timestep을 검증한 뒤 재사용한다. 검토용으로 선별한
M1 보고서·요약·그래프는 `experiments/m1_reproducibility/`에 보존한다.

예측형 Safety Shield 코어도 구현돼 있지만 기본 설정은 여전히
`shield_strategy: legacy`다. predictive 전략은 M2 평가 gate를 통과하기 전까지
승인된 기본 동작으로 간주하지 않는다.

## 학습형 local planner 확장 (M2–M6)

opticore-amr의 DWA를 학습 정책으로 대체하기 위한 확장이다. 계약은
[ARCHITECTURE.md](../../docs/ARCHITECTURE.md), 실험 절차는
[EXPERIMENT_PLAYBOOK.md](../../docs/EXPERIMENT_PLAYBOOK.md)를 따른다. 아래 기능은 모두
기본값에서 꺼져 있어 Phase 0/M1 결과는 그대로다.

### Robot profile과 센서 모델

```python
from amr_lite.config import resolve_env_config
config = resolve_env_config("opticore_gazebo", {"shield_strategy": "scan"})
```

- `configs/robot_profiles/opticore_gazebo.yaml`: Gazebo 로봇의 `w_max` 1.5, 가감속, 720
  sample/360°/0.25 m 장착 LiDAR(10 Hz, σ 0.03 m), A* 벽 거리 비용, 경로 길이 비례 timeout
- `lidar_model.type: raw`: raw scan을 시뮬레이션해 `amr_lite.sensors.lidar.resample_scan`으로
  72-ray 계약에 맞춘다. ROS 노드가 같은 함수를 쓴다.
- `sensor_model`·`domain_randomization`: scan hold, 잡음, dropout, 각도 bias, 관측·명령
  지연, 속도 scale, actuator lag, footprint 여유, localization 잡음, 경로 밀도·흔들림
- `shield_strategy: scan`: 정답 정보 없이 LiDAR 점과 cluster track만 쓰는 Safety Shield

### 시나리오

| ID | 내용 |
|---|---|
| `P-AISLE`, `P-HALL`, `P-PINCH`, `P-JUNCTION`, `P-MIXED` | 절차적 정적 layout |
| `PD-*` | 위 layout + 경로 기준 동적 agent(횡단·정면·추종·정지·정차·왕복) |
| `WH-RANDOM`, `WHD-RANDOM` | opticore-amr 창고 지도 위 무작위 start/goal (`WHD`는 동적 agent 포함) |
| `WH-warehouse_v1-00`…`07` | Gazebo와 짝을 이루는 paired leg (`configs/paired/warehouse_v1.yaml`) |

절차적 split(`procedural.split: train|validation|test`)은 `configs/procedural.yaml`의
파라미터 영역으로 나뉜다. 평가 protocol v2는 `evaluation_selection_v2.yaml`,
`evaluation_test_v2.yaml`이다.

### 학습과 실험 명령

```bash
python -m amr_lite.cli.train_rl --smoke --output artifacts/rl/smoke            # PPO v2
python -m amr_lite.cli.train_rl --set policy.mode=direct --set lagrangian.enabled=true   --output artifacts/rl/direct-lag                                              # 설정 override
python -m amr_lite.cli.dagger --output artifacts/dagger/run                     # BC → DAgger
python -m amr_lite.cli.sweep e2_policy_mode --stage train --parallel 2          # 실험 sweep
python -m amr_lite.cli.sweep e2_policy_mode --stage select
python -m amr_lite.cli.sweep e6_candidate --stage test                          # sweep당 1회
```

PPO v2(`amr_lite.rl`)는 vector env, residual/direct 정책, MLP/1D-CNN, 비용 critic과
Lagrange 승수, curriculum, pause/resume(`--max-updates`, `--resume`)을 지원한다.
M1 trainer(`learning/train_ppo.py`)는 기준선 재현용으로 그대로 둔다.

### 배포와 paired 평가

```bash
python -m amr_lite.cli.export_policy --checkpoint <best.pt> --output artifacts/bundles/cand
python -m amr_lite.cli.verify_bundle artifacts/bundles/cand --robot-profile opticore_gazebo
python -m amr_lite.cli.paired lite --policy bundle:artifacts/bundles/cand --seeds 0 1 2   --output artifacts/paired/lite.csv
python -m amr_lite.cli.paired gap --reference artifacts/paired/lite.csv   --candidate gazebo.csv --output artifacts/paired/GAP_REPORT.md
```

bundle은 torch 없이 `amr_lite.deploy.runtime.PolicyRuntime`으로 실행되며, ROS2 쪽 사용법은
[amr_learned_planner README](../../integrations/ros2/amr_learned_planner/README.md)에 있다.

## 해석 시 주의

`lite_rule_baseline`은 기존 ROS rule-based hybrid local planner의 축소판이며 동등 구현이 아니다. PPO smoke는 256 timestep이며 실행 경로 확인용이다. 초기 shared-encoder 100,000-timestep PPO는 작은 원을 도는 정책으로 붕괴했지만, stable PPO best는 51,200 timestep에서 전체 평가 성공률 87.5%를 기록했다. 다만 final checkpoint는 50%로 저하됐고 `D1` 정면 접근 실패가 남아 있으므로 수렴·일반화를 주장하지 않는다. 자세한 분석은 결과와 한계 문서에 명시했다.
