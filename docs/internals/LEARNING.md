# Learning Stack Internals — 관측·모방학습·PPO

이 문서는 `packages/amr_lite/amr_lite/learning/`과 `amr_lite/rl/`의 코드가 **실제로 어떻게
동작하는지**를 설명한다. 관측·행동·bundle 계약은 [ARCHITECTURE.md](../ARCHITECTURE.md),
마일스톤과 gate는 [ROADMAP.md](../ROADMAP.md), 실험 절차와 예산은
[EXPERIMENT_PLAYBOOK.md](../EXPERIMENT_PLAYBOOK.md)가 기준이며 여기서 다시 정의하지 않는다.

## 1. 역할과 범위

학습 스택은 같은 관측 `float32[296]`·행동 `[a_v, a_w]` 계약 위에서 세 계열의 정책을 만든다.

| 계열 | 코드 | 도입 | 현재 상태 |
|---|---|---|---|
| 관측·행동 함수 | `learning/observation.py`, `learning/action.py` | Phase 0 | Lite env와 ROS core가 같은 함수를 import. `*_contract`는 bundle export가 사용 |
| Teacher 수집, BC, DAgger-lite | `collect_teacher.py`, `dataset.py`, `train_bc.py`, `collect_dagger.py` | Phase 0 | DAgger loop가 `collect`의 `beta`·`env_config` 인자를 사용 |
| M1 PPO (shared / stable) | `learning/train_ppo.py`, `learning/networks.py` | Phase 0 (D-012, D-013), M1 5-seed | **동결**. M1 기준선 재현용 |
| BC → DAgger 반복 loop | `learning/dagger_loop.py`, `cli/dagger.py` | M3 | smoke만 실행됨 (E1 미실행) |
| PPO v2 | `rl/` 전체, `cli/train_rl.py` | M3 | smoke만 실행됨 (E2–E6 미실행) |

범위 밖: 환경 물리·시나리오([ENVIRONMENT](ENVIRONMENT.md)), 센서·Safety Shield·rule baseline
([SENSORS_AND_SAFETY](SENSORS_AND_SAFETY.md)), 통계·sweep·G4 gate
([EVALUATION_AND_EXPERIMENTS](EVALUATION_AND_EXPERIMENTS.md)), bundle·ROS
([DEPLOY_AND_ROS2](DEPLOY_AND_ROS2.md)). 이 문서는 그 경계에서 학습 스택이 기대하는 것만 적는다.

학습 이력: PPO from scratch 100,000 step은 작은 원을 도는 reward hacking에 수렴했고, 공유 encoder BC
warm-start PPO의 best는 timestep 0, Stable PPO best는 51,200 timestep이었다(Phase 0
[DECISIONS](../../experiments/phase0_amr_lite/DECISIONS.md)). M1 5-seed best는 Shield OFF 85.4%(rule
86.2%)이고 final은 저하됐다(seed 7: 85.6% → 66.2%,
[M1 RESULTS](../../experiments/m1_reproducibility/RESULTS.md)).

## 2. 파일 지도

경로는 `packages/amr_lite/` 기준이다.

| 파일 | 책임 | 주요 심볼 |
|---|---|---|
| `amr_lite/learning/observation.py` | 4개 scan + 8개 tail → 296 벡터, 계약 기술 | `build_observation`, `observation_contract` |
| `amr_lite/envs/warehouse_env.py` | frame stack, `history_mode`, 지연 queue (관측의 실제 조립 위치) | `WarehouseEnv._observation`, `_sensor_scan`, `step` |
| `amr_lite/learning/action.py`, `envs/dynamics.py` | 정규화 행동 ↔ `(v, w)`, 가감속 제한 | `scale_action`, `normalize_command`, `limit_velocity` |
| `amr_lite/learning/dataset.py` | episode NPZ 저장·로드, episode 단위 split | `save_episode`, `load_dataset`, `episode_split` |
| `amr_lite/learning/collect_teacher.py` | teacher rollout, DAgger relabel, β-mixture | `collect` |
| `amr_lite/learning/collect_dagger.py` | Phase 0 DAgger-lite CLI (위험 상태만 저장) | `main` |
| `amr_lite/learning/train_bc.py` | BC MLP 학습, 위험 가중 MSE | `train` |
| `amr_lite/learning/networks.py` | BC·M1 네트워크, BC/M1 추론 래퍼, checkpoint dispatch | `ActorMLP`, `ActorCritic`, `StableActorCritic`, `TorchPolicy`, `load_policy` |
| `amr_lite/learning/dagger_loop.py` | BC → DAgger round, selection 평가, best 복사 | `run_dagger` |
| `amr_lite/learning/train_ppo.py` | M1 PPO (동결) | `train`, `RunningReturnStats`, `_warm_start_actor`, `_evaluate_actor` |
| `amr_lite/rl/vector_env.py` | slot 자동 reset, curriculum 샘플링, sync/subprocess | `EpisodeSlot`, `SyncVectorEnv`, `SubprocVectorEnv`, `training_episode_seed` |
| `amr_lite/rl/ppo.py` | PPO v2 trainer | `train`, `_gae`, `RunningMeanStd`, `warm_start_actor`, `checkpoint_payload` |
| `amr_lite/rl/networks.py` | 분리 actor/critic/cost tower, MLP·1D-CNN | `PolicyNetwork`, `ScanCNNEncoder` |
| `amr_lite/rl/policy.py` | direct/residual 추론 래퍼 | `PPOPolicy`, `combine_residual`, `build_policy` |
| `amr_lite/rl/evaluation.py` | protocol 기반 결정적 병렬 평가, 선정 점수 | `evaluate_protocol`, `run_episodes`, `selection_score` |
| `amr_lite/cli/train.py` | Phase 0 통합 smoke (teacher→BC→M1 PPO→평가→데모) | `smoke_pipeline` |
| `amr_lite/cli/train_rl.py`, `cli/dagger.py` | PPO v2·DAgger CLI, `--set a.b=v` override | `load_trainer_config`, `parse_overrides` |
| `amr_lite/envs/gymnasium_adapter.py` | Gymnasium 래퍼 (사용·테스트하는 코드 없음) | `GymnasiumWarehouseEnv` |

## 3. 동작 흐름

### 3.1 모방학습 (teacher → BC → DAgger)

```text
RuleBasedPlanner(teacher) ──collect()──► round_0/teacher/episode_XXXX_<scn>.npz
                                               │  datasets = [round_0/teacher]
                                               ▼
                 train_bc(datasets) ──► round_0/bc.pt ──evaluate_protocol(selection v2)──► score_0
                         │
   round k = 1..rounds   ▼
   learner = round_{k-1}/bc.pt
   collect(learner, beta = beta·beta_decay^(k-1), seed + 1000k)
      매 step: teacher_action = rule(obs) 기록, 실행은 확률 beta로 teacher, 아니면 learner
                         │
                         ▼  datasets += [round_k/dagger]   (누적 aggregation)
                 train_bc(모든 datasets, 처음부터 재학습) ──► round_k/bc.pt ──► score_k
                         │
                         ▼
   argmax_k selection_score ──copy──► best.pt, rounds.csv, result.json
```

### 3.2 PPO v2 학습 loop

```text
train(cfg)
 ├─ resolve_env_config(robot_profile, environment_overrides)
 ├─ PolicyNetwork(architecture + cost_critic=lagrangian.enabled)
 │    residual: zero_mean_head()          direct+dataset: warm_start_actor()
 ├─ (resume) state.pt → network/optimizer/value·cost stats/λ/update/best/RNG
 ├─ make_vector_env([SlotSpec(slot_seed, curriculum) × num_envs], backend)
 └─ for update in [start_update, stop_at):
      progress = timesteps / total → set_progress, lr·entropy schedule
      rollout_steps × num_envs:
        latent ~ N(mean, exp(log_std)) (sampler generator) → a_pi = tanh(latent)
        a = a_pi  또는  clip(rule(obs) + s·a_pi, −1, 1)
        vector.step(a) → reward, cost(=info.safety_cost), terminated, truncated, final_observation
      시간 초과 step: reward += γ·V(final_obs), cost += γ·Vc(final_obs)
      GAE(reward) [+ GAE(cost)] → returns
      λ ← clip(λ + lambda_lr·(mean cost − cost_limit), 0, lambda_max)      (Lagrangian 켜짐)
      value(·cost) stats 갱신 → 정규화 target
      A = std(A_r)  또는  (std(A_r) − λ·std(A_c)) / (1 + λ)
      epochs × minibatch: KL > 1.5·target_kl면 즉시 중단, clipped PPO + value(+cost) loss − entropy
      training.csv 1행 append
      interval마다 + 마지막 update: selection 평가 → score 개선 시 best.pt
      checkpoint_interval마다 또는 일시정지 시: state.pt
 └─ 완료: final.pt, result.json, state.pt 삭제
```

## 4. 핵심 구성요소 상세

### 4.1 관측 생성

- `learning/observation.py`의 `build_observation`은 순수 함수다. 입력은 4개의 72-ray scan, `PathState`,
  `RobotState`, 정규화 상수이고 출력은 `float32[296]`다. 4와 72는 하드코딩이라 다르면 `ValueError`이고,
  결과가 유한하지 않아도 `ValueError`다(Lite에서는 health fault가 아니라 예외로 학습이 멈춘다).
- frame stack은 `WarehouseEnv`가 관리한다. `reset`은 `_sensor_scan(force=True)` 결과로
  `deque(maxlen=4)`를 채운다. `step`은 물리 substep 뒤 `_sensor_scan()`을 append하는데, `_scan_due()`가
  거짓이면 이전 scan을 그대로 돌려줘 zero-order hold가 된다(`opticore_gazebo`의 `scan_period` 0.1 s).
- `_observation`은 경로 특징을 `_perceived_robot()`(localization 잡음 포함)으로 계산하고,
  `history_mode == "newest"`면 `[scans[-1]] * 4`로 바꾼다. 관측 지연 queue는 reset 관측으로, 행동 지연
  queue는 정지 행동 `[-1, 0]`으로 초기화한다.
- `history_mode` 치환은 `build_observation` 밖의 **두 곳**에 있다: `WarehouseEnv._observation`과
  `integrations/ros2/amr_learned_planner/amr_learned_planner/core/planner_core.py`의 `build_observation`
  메서드. 평가는 정책 객체의 `observation_overrides`를 env config에 merge해 정책 쪽 값을 따른다.
- index 288은 경로 투영점에서 `lookahead_distance`(0.65 m) 앞 점까지의 거리라 경로 위에서는 대부분
  0.65/8 ≈ 0.081이다. 최종 목표까지의 거리가 아니다(`planning/path_utils.py`의 `path_state`).

### 4.2 행동 변환과 `limit_velocity`

- `scale_action`: shape가 `(2,)`가 아니거나 유한하지 않으면 `(0.0, 0.0)`, 그 외에는 `[-1, 1]` clip 후
  `v = (a_v + 1)/2·v_max`, `w = a_w·w_max`. 역변환 `normalize_command`는 rule baseline이 쓰며 `v < 0`을 0으로 자른다.
- `limit_velocity`는 목표를 속도 한계로 자른 뒤 `dv`를 가속이면 `accel_max·dt`, 감속이면 `decel_max·dt`,
  `dw`를 `angular_accel_max·dt`로 제한한다.
- Lite에서 가감속 제한은 **Shield 클래스 안에서** 적용된다. legacy·predictive(`SafetyShield._limit`)와
  scan(`ScanSafetyShield._limit`) 모두 enabled 검사 전에 `_limit`을 부르므로 Shield OFF에서도 적용된다.
  dt는 `policy_dt`이고, 이후 `physics_substeps`(5)번 같은 명령을 적분한다.
- `WarehouseEnv.step`은 유한하지 않은 행동을 0으로 바꾸고 episode를 끝낸다(`INVALID_ACTION`).

### 4.3 Teacher 수집과 dataset 형식

`learning/collect_teacher.py`의 `collect(output_dir, episodes, max_steps, seed, scenarios, learner,
dagger_risk_only, env_config, beta)`:

- episode `i`는 seed `seed + i`, 시나리오 `scenarios[i % len]`(기본 `S0`–`S5`),
  `options={"pose_perturbation": 0.08}`로 시작한다.
- 매 step `teacher_action = RuleBasedPlanner(env_config).predict(obs)`를 **항상** 기록한다. 실행 행동은
  learner가 없으면 teacher, 있으면 learner이며 `beta > 0`이면 별도 RNG(`seed + 104729`)로 확률 `beta`만큼
  teacher를 실행한다.
- `dagger_risk_only=True`면 `minimum_clearance < 0.7`, `|mean_cross_track_error| > 0.25`, 종료 step만 저장한다.
- episode당 `episode_{i:04d}_{scenario}.npz` 하나를 `np.savez_compressed`로 쓴다.

| NPZ key | 내용 |
|---|---|
| `observation`, `teacher_action` | `float32[N, 296]`(step 직전 관측), `float32[N, 2]` |
| `episode_id`, `seed`, `step_index` | episode seed(`seed + i`) 두 번, episode 내 step |
| `scenario_id`, `map_id` | 둘 다 시나리오 ID |
| `robot_pose`, `goal_distance`, `minimum_clearance`, `collision`, `success`, `terminated`, `truncated` | step **이후** `info` 값 |
| `path_error` | `info["mean_cross_track_error"]` = episode 누적 평균 \|CTE\| (순간값 아님) |

`dataset.py`의 `load_dataset`은 디렉터리(또는 목록)의 `*.npz`를 이름순으로 합치고, 모든 파일에
`minimum_clearance`가 있을 때만 그 배열을 채운다. `episode_split`은 고유 `episode_id`를 섞어
`max(1, round(n·fraction))`개를 validation으로 뺀다(episode 1개면 0개). D-008의 구현이다.

### 4.4 BC 학습

`learning/train_bc.py`의 `train(dataset_dir, output, smoke, config)`:

- `ActorMLP`: `296 → hidden → hidden → 2`, ReLU, 출력 `Tanh`. Adam, seed 고정 shuffle, `torch.set_num_threads(1)`.
- loss는 `MSELoss`. `risk_weight != 1.0`이고 `minimum_clearance`가 있으면 `clearance < risk_clearance`
  sample에 가중치를 준 `Σ w·(pred − a)² / (2·Σ w)`를 쓴다.
- early stopping 없이 마지막 epoch를 저장한다. validation이 비면 train 전체로 지표를 낸다.
- 출력: `{"kind": "bc", "hidden_size", "model_state", "config", "metrics"}`와 `<output>.metrics.json`
  (`train_loss`(마지막 batch), `mse`, `v_mae`, `w_mae`, sample 수).

### 4.5 DAgger loop

`learning/dagger_loop.py`의 `run_dagger(output, config, smoke)`:

| round | dataset | 실행 정책 | 수집 seed | β (기록값) |
|---|---|---|---|---|
| 0 | `round_0/teacher` | teacher | `seed` | 1.0 |
| k ≥ 1 | `round_k/dagger` | `round_{k-1}/bc.pt` + β-mixture | `seed + 1000·k` | `beta·beta_decay^(k−1)` (기본 0.5, 0.25, 0.125) |

- 방문한 모든 상태를 teacher로 relabel하고(`risk_only: false`), 매 round 누적된 **모든** dataset으로 BC를
  처음부터 재학습한다(같은 `seed`, `epochs_full`). "위험 상태 가중치"는 BC loss의 `risk_weight` 2.0이다.
- 각 `bc.pt`를 `evaluation_selection_v2`로 평가한다(시나리오당 2 seed). `selection_score`가 엄격히 클 때만
  best를 바꾸므로 동점이면 앞 round가 남는다.
- 산출물: `round_k/*/*.npz`, `round_k/bc.pt`, `best.pt`, `rounds.csv`(`round`, `beta`, `chunks`,
  `datasets`, `checkpoint`, `bc_*`, `eval_*`), `result.json`. smoke는 teacher 2 episode·40 step,
  `hidden_size` 32, 1 epoch, `rounds ≤ 1`, `S0` 1 episode 평가다.
- teacher가 `lite_rule_baseline`이므로 DAgger는 teacher를 넘을 수 없다(모듈 docstring).

### 4.6 M1 PPO (`learning/train_ppo.py`, 동결)

단일 `WarehouseEnv`(`env.yaml` 그대로: `lite_default`, legacy shield ON, scenario randomization OFF)에서
rollout을 모은다. 시나리오는 진행률 30% 전 `S0`, 60% 전 `S0`/`S1` 교대, 이후 `S0`/`S1`/`S3` 순환이고
seed는 `seed + episode_number`다. randomization이 꺼져 있어 **고정 layout 3개**만 학습한다
(`envs/scenarios.py`의 `make_scenario`는 randomization이 꺼지면 seed를 쓰지 않는다).

"Stable PPO"(`--stable`, `cli/m1.py`가 사용)는 다음 조합이다.

| 요소 | 구현 |
|---|---|
| actor/critic 분리 | `StableActorCritic`: actor·critic 각각 `296→128→128` tanh body. 공유 body에서는 critic loss가 BC actor 표현을 바꿨다(D-013) |
| value normalization | `RunningReturnStats`. rollout 값은 역정규화해 GAE, return으로 통계 갱신 후 정규화 target. `--stable` + `value_normalization`일 때만 |
| KL early stop | minibatch 갱신 **전** `mean(old_logp − new_logp) > target_kl`이면 남은 epoch 전체 중단 |
| BC warm start | `_warm_start_actor`: `tanh(mean)`을 teacher 행동에 MSE로 맞춘 뒤 `log_std = warm_start_log_std` |
| critic warmup | warm start 시 `rollout_start < critic_only_steps`인 update(기본 처음 10개)는 `critic_optimizer`로 critic만 학습 |
| BC anchor | 초기 actor 복사본과의 `mean((tanh μ − tanh μ_ref)²)` × `bc_anchor_coef` |
| fine-tune | warm start 시 전체 optimizer lr `fine_tune_learning_rate`, entropy `fine_tune_entropy_coef` |
| 선정 | warm start + full이면 timestep 0 평가, 이후 `evaluation_interval_updates`마다와 마지막에 `selection_config` 평가, 점수 개선 시 `<stem>_best.pt` |

Gaussian은 tanh 이전 latent에 정의되고 ratio는 저장한 latent로 계산한다. 시간 초과도 terminal로 처리하며,
gradient clip 0.5는 하드코딩, `training.csv`는 rollout 안 **마지막 episode 하나**의 통계만 기록한다.
산출물은 `<out>.pt`, `<out>_best.pt`, `<out>.training.csv`, `<out>.metrics.json`, kind는 `ppo`/`ppo_stable`이다.

### 4.7 PPO v2 (`rl/`)

#### 4.7.1 Vector env

- `EpisodeSlot`은 `WarehouseEnv` 하나와 자기 RNG를 갖고 episode가 끝나면 스스로 reset한다. 종료 step
  `info`에서 `reward_components`, `robot_pose`, `sensor_model`을 빼고 `episode_return`, `episode_cost`,
  `episode_length`, `action_smoothness`, `seed`, `stage`, `final_observation`을 붙인다. 반환 관측은 이미
  다음 episode의 첫 관측이다.
- 다음 episode: `spec.episodes`(평가용 고정 목록)가 있으면 순서대로, 다 쓰면 `exhausted`. 아니면
  `stage_for(curriculum, progress)`의 `scenarios` 가중치로 샘플링한다. stage `environment_overrides`가
  바뀔 때만 env를 새로 만든다.
- `SyncVectorEnv`는 한 process에서 차례로, `SubprocVectorEnv`는 slot당 `spawn` process 하나로 lockstep
  실행한다. slot이 독립 RNG를 가져 두 backend 결과가 같다. worker는 torch를 import하지 않는다. slot이
  1개면 subprocess 요청도 sync가 된다. `spawn`이라 stdin으로 넣은 스크립트에서는 worker가 죽는다.

#### 4.7.2 Seed 체계

| 대상 | 규칙 |
|---|---|
| 전역 (네트워크 초기화) | `random.seed(seed)`, `np.random.seed(seed % 2**32)`, `torch.manual_seed(seed)` |
| 행동 샘플링·minibatch 순열 | `torch.Generator().manual_seed(seed + 1)` (`sampler`) |
| slot 기준 seed | `slot_seed = seed + 7919·start_update` (resume하면 새 stream) |
| 시나리오 샘플러 | `SeedSequence([slot_seed, slot, crc32("scenario-sampler")])` |
| episode seed | `1_000_000 + SeedSequence([slot_seed, slot, counter]) % (2**31 − 1 − 1_000_000)` |
| 평가 | protocol `seed + offset` (selection 20000+, test 30000+) |

하한 `TRAINING_SEED_BASE`는 selection/test namespace와 겹치지 않게 한다. teacher·DAgger 수집은 이 체계를
쓰지 않고 `seed + i`, `seed + 1000k + i`(기본 7부터)를 쓴다.

#### 4.7.3 Curriculum과 randomization ramp

base override가 `domain_randomization.enabled: true`를 켜고 stage는 `probability`만 바꾼다.

| stage | `until` | 시나리오(가중치) | DR 확률 |
|---|---:|---|---:|
| `static_basics` | 0.2 | S0 1, S1 1, S3 1, S4 1, P-AISLE 2, P-PINCH 1 | 0.0 |
| `static_mixed` | 0.5 | S1 1, S2 1, S3 1, S4 1, S5 1, P-MIXED 4, WH-RANDOM 1 | 0.5 |
| `dynamic` | 1.0 | S2 1, S3 1, S4 1, D0–D4 각 1, P-MIXED 3, PD-MIXED 4, WH-RANDOM 1, WHD-RANDOM 1 | 1.0 |

진행률은 update 시작 시 `timesteps / total`로 갱신되고, stage 전환은 각 slot의 다음 reset부터 적용된다.

#### 4.7.4 Rollout, GAE, policy mode

- buffer는 `(rollout_steps, num_envs)` 모양이고 `values`는 역정규화 값으로 저장한다. `_gae`는 마지막
  step을 rollout 끝 관측의 V로 bootstrap하고 done step에서 끊는다. `returns = advantages + values`.
- 시간 초과(`truncated and not terminated`) step은 done으로 끊되 `reward += γ·V(final_observation)`로
  보정한다. advantage 표준화는 batch 전체에서 한 번 한다.
- `updates = ceil(total / batch)`라 step 수가 batch 배수로 올림된다(기본 2,000,000 → 977 update, 2,000,896 step).
- direct: 실행 행동 = `tanh(latent)`. residual: `combine_residual(base, a_pi, s) = clip(base + s·a_pi, −1, 1)`,
  `base`는 같은 관측의 `RuleBasedPlanner(env_config, baseline.yaml).predict`(newest scan 슬롯만 사용).
  PPO의 확률 변수는 residual latent이고 combine·clip은 환경의 일부처럼 취급된다.
- residual 초기화는 `PolicyNetwork.zero_mean_head()`(mean head weight·bias = 0)라 결정적 행동이 rule과
  **정확히 같다**. 학습 rollout에는 `log_std_init` −1.0(σ ≈ 0.37) 잡음이 있다. residual 모드에서
  `warm_start.dataset`은 무시된다.

#### 4.7.5 네트워크

`rl/networks.py`의 `PolicyNetwork`는 actor, critic, (선택) cost critic이 **각각 독립 tower**를 가진다.

| encoder | tower 구조 (기본값) |
|---|---|
| `mlp` | `296 → 256 → 256`, activation `tanh` |
| `cnn` | `ScanCNNEncoder`: scan 288개를 `(4 frame = channel, 72 ray)`로 reshape, `Conv1d(4→16, k5, s2)` → 길이 34, `Conv1d(16→32, k5, s2)` → 길이 15, flatten 480 + state branch `8 → 64` = 544 → `544 → 256 → 256` |

초기화는 Linear/Conv1d orthogonal gain √2·bias 0, actor mean head gain 0.01, critic·cost head gain 1.0이다.
`log_std`는 상태와 무관한 `Parameter(2)`이고 clamp가 없다. CNN encoder는 tower마다 따로 만든다.

#### 4.7.6 손실, 정규화, KL, schedule

- loss = `policy_loss + value_coef·(value_loss + cost_loss) − entropy_coef·entropy (+ anchor_coef·anchor)`.
  value clipping은 없고 entropy는 tanh 이전 Gaussian entropy다.
- value normalization은 `RunningMeanStd`를 return으로 갱신하고 critic이 정규화 return을 학습한다. cost
  critic은 별도 `cost_stats`를 쓴다.
- KL은 `mean((ratio − 1) − log ratio)`이며 minibatch 갱신 **전에** `> 1.5·target_kl`이면 `kl_stop = 1`로
  남은 epoch를 모두 건너뛴다.
- lr과 entropy 계수는 진행률에 선형 schedule이다. Adam `eps = 1e-5`, `clip_grad_norm`은 `max_grad_norm`.
- `warm_start.anchor_coef > 0`이면 초기 network 복사본과 `tanh` mean의 MSE를 더한다. residual에서 켜면
  기준은 zero head(=rule)다.

#### 4.7.7 비용 critic과 Lagrangian (PPO-Lag)

- 비용은 env의 `info["safety_cost"] = collision + max(0, safety_distance − clearance)`(0.30 m)다. reward의
  `reward_collision`·`reward_near`도 같은 사건을 벌점으로 준다.
- `lagrangian.enabled`가 architecture의 `cost_critic`을 켠다. λ는 update마다 한 번, policy 갱신 전에
  `λ ← min(max(0, λ + lambda_lr·(mean(costs) − cost_limit)), lambda_max)`로 갱신한다.
- 결합 advantage는 `(std(A_r) − λ·std(A_c)) / (1 + λ)`다.

#### 4.7.8 학습 중 평가와 checkpoint

- `evaluation.interval_updates`마다와 마지막 update에서 `_policy_for`가 만든 `PPOPolicy`를
  `evaluate_protocol`로 평가한다. 기본은 `evaluation_selection_v2`, 시나리오당 2 seed(20000, 20001),
  protocol의 `shield_modes: [false]`라 Shield OFF다.
- 점수 `selection_score = (success, −collision, −timeout, mean_clearance, −mean_time)` 사전식이며 엄격히
  클 때만 `best.pt`를 덮어쓴다. **timestep 0 평가는 없다**(M1과 다름). `final.pt`는 마지막 weight에
  `value_stats`, `lagrange`를 더 담는다.
- v2 checkpoint 필드: `kind: ppo_v2`, `architecture`, `model_state`, `policy{mode, residual_scale}`,
  `env_config`, `baseline_config`(residual만), `config`, `training_seed`, `total_timesteps`, `evaluation`.

#### 4.7.9 Pause/resume (`state.pt`)

- `checkpoint_interval_updates`마다 또는 `--max-updates`로 멈출 때 `state.pt`를 쓴다(마지막 update 제외):
  `model_state`, `optimizer_state`, `value_stats`, `cost_stats`, `lagrange`, `update`, `timesteps`,
  `best_score`, `best_evaluation`, `sampler_state`, `torch_rng`.
- config, NumPy/`random` 상태, 진행 중 episode는 저장하지 않는다. slot stream이 바뀌므로 재현 가능하지만
  중단 없는 실행과 bit 단위로 같지 않다(코드 주석). resume이 아니면 `training.csv`를 지우고, 완료하면
  `state.pt`를 지운다. sweep은 `state.pt`가 있으면 `--resume`을 붙인다(`experiments/sweep.py`의 `_run_trial`).

#### 4.7.10 `training.csv` 열

| 열 | 의미 |
|---|---|
| `update`, `timesteps`, `progress` | 1부터, 누적 step, rollout **후** 진행률 |
| `stage` | 이 update에서 끝난 **마지막 episode**의 stage. 없으면 빈 칸 |
| `learning_rate`, `entropy_coef` | 이 update에 쓴 값(rollout 전 진행률 기준) |
| `episodes`, `success_rate`, `collision_rate`, `timeout_rate`, `episode_return`, `episode_cost` | 이 rollout에서 끝난 episode 평균(확률적 정책, 학습 env). 없으면 `nan` |
| `step_cost`, `lagrange` | `costs.mean()`(truncation bootstrap 포함, §8-1), 갱신 후 λ |
| `log_std_mean` | `log_std` 평균 |
| `policy_loss`, `value_loss`, `cost_value_loss`, `entropy`, `approx_kl`, `clip_fraction` | **마지막으로 적용된 minibatch** 값 |
| `kl_stop` | KL 조기 중단 여부 |
| `eval_success_rate`, `eval_collision_rate`, `eval_timeout_rate`, `eval_mean_clearance`, `eval_mean_time` | 평가 update에서만 값, 나머지 `nan` |
| `wall_seconds` | 이 `train` 호출 시작부터 경과 시간(resume 시 0부터) |

### 4.8 M1 PPO vs PPO v2

| 항목 | M1 (`learning/train_ppo.py`) | PPO v2 (`rl/ppo.py`) |
|---|---|---|
| 환경 | 단일 env, `lite_default`, legacy shield ON, randomization OFF | vector env 8개(subprocess), `opticore_gazebo`, scan shield ON, scenario randomization·procedural train·DR |
| 시나리오 / seed | S0 → S0/S1 → S0/S1/S3 고정 순환 / `seed + episode_number` | 3단계 가중 curriculum / ≥ 1,000,000 |
| 정책 | direct | direct / residual |
| 네트워크 | `ActorCritic`(공유 128×2) 또는 `StableActorCritic`(분리), PyTorch 기본 초기화, `log_std` −0.5 | 분리 tower MLP 256×2 또는 1D-CNN, cost tower 선택, orthogonal, `log_std` −1.0 |
| batch / epoch / minibatch | 1024 / 4 / 64 | 8×256 = 2048 / 10 / 256 |
| lr, entropy | 0.0003 고정(warm start 시 0.00005), 0.005(warm start 시 0.0) | 0.0003→0.00003, 0.001→0.0 선형 |
| value normalization | `--stable` + config | config(기본 on) |
| KL stop | `mean(old−new) > 0.01` | `mean((r−1) − log r) > 1.5 × 0.02` |
| 시간 초과 | terminal | γV(final) bootstrap |
| critic warmup / BC anchor | warm start 시 10,000 step / 0.5 | 없음 / direct warm start에서 `anchor_coef`(기본 0.0) |
| 비용 제약 | 없음 | cost critic + λ (기본 off) |
| 선정 평가 | v1 selection 8 시나리오×5 seed, 10 update마다 + t=0 + 마지막 | v2 selection 17×2, 20 update마다 + 마지막 |
| resume / kind | 없음(M1 CLI는 완료 checkpoint 재사용) / `ppo`, `ppo_stable` | `state.pt`, `--max-updates`, `--resume` / `ppo_v2` |

### 4.9 정책 래퍼와 구현 검증 상태

모든 정책은 `predict(obs, deterministic=True) → float32[2]`를 가지며, `rl/evaluation.py`의 `_predict`는
`predict_batch`가 있으면 그것을 쓴다. 모든 checkpoint의 진입점은 `learning/networks.py`의 `load_policy`다.

| 래퍼 | 생성 | 결정적 행동 |
|---|---|---|
| `learning/networks.py`의 `TorchPolicy` | `load_policy`, kind `bc`/`ppo`/`ppo_stable` | BC는 네트워크 출력, PPO는 `tanh(mean)`. `predict_batch`는 1개씩 반복 |
| `rl/policy.py`의 `PPOPolicy` | `build_policy`(kind `ppo_v2`), `load_ppo_v2`, 학습 중 `_policy_for` | `tanh(mean)`, residual이면 `combine_residual`. `observation_overrides`에 `history_mode` |
| `deploy/runtime.py`의 `PolicyRuntime` | bundle 로드 | NumPy 재구현, residual 포함 |
| `planning/rule_baseline.py`의 `RuleBasedPlanner` | 직접 생성 | rule |

bundle export(`deploy/export.py`의 `_actor_graph`)는 kind별로 actor만 꺼내 `mlp`/`scan_cnn` graph로 기록한다.

PPO는 SB3 같은 외부 라이브러리가 아니라 자체 구현이다(D-005). `GymnasiumWarehouseEnv`와 optional 의존성
`reference-rl = ["gymnasium>=0.29", "stable-baselines3>=2.3"]`(`pyproject.toml`)은 있지만 이 adapter를
import하는 코드·테스트·비교 스크립트는 없고, 개발 환경에 두 패키지가 설치돼 있지 않다. M1 PPO와 PPO v2
모두 **참조 구현과 수치 비교한 적이 없다**([NEXT_STEPS.md](../NEXT_STEPS.md)의 기술 부채).

## 5. 설정

파일은 `packages/amr_lite/configs/`에 있다. `train_rl`·`dagger` CLI는 `--config <이름|경로>`와
`--set a.b.c=<YAML 값>`(`cli/train_rl.py`의 `parse_overrides`)으로 덮어쓰며 리스트는 통째로 교체된다.

### 5.1 `bc.yaml` (읽는 곳: `learning/train_bc.py`의 `train`)

| key | 기본값 | 비고 |
|---|---|---|
| `seed` | 7 | split·shuffle·초기화 |
| `hidden_size` / `batch_size` / `learning_rate` | 256 / 128 / 0.001 | |
| `epochs_smoke` / `epochs_full` | 2 / 30 | `--full`이면 full |
| `validation_fraction` | 0.2 | episode 단위 |
| `risk_weight` / `risk_clearance` | 파일에 없음 → 1.0 / 0.7 | DAgger config에서 2.0 / 0.7 |

### 5.2 `dagger.yaml` (읽는 곳: `learning/dagger_loop.py`의 `run_dagger`)

| key | 기본값 |
|---|---|
| `seed`, `robot_profile`, `rounds` | 7, `opticore_gazebo`, 3 |
| `environment_overrides` | `shield_strategy: scan`, `scenario_randomization.enabled: true`, `procedural.split: train` |
| `teacher.episodes` / `max_steps` | 48 / 1500 |
| `teacher.scenarios` | S0, S1, S2, S3, S4, S5, P-AISLE, P-HALL, P-PINCH, P-JUNCTION, WH-RANDOM |
| `dagger.episodes` / `max_steps` | 32 / 1500 |
| `dagger.beta` / `beta_decay` / `risk_only` | 0.5 / 0.5 / false |
| `dagger.scenarios` | S1, S2, S3, S4, S5, D0, D2, D4, P-MIXED, PD-MIXED, WH-RANDOM |
| `bc.*` | `hidden_size` 256, `batch_size` 128, `learning_rate` 0.001, `epochs_full` 30, `validation_fraction` 0.2, `risk_weight` 2.0, `risk_clearance` 0.7 |
| `evaluation` | `protocol: evaluation_selection_v2`, `episodes_per_scenario` 2, `workers` 4, `backend: subprocess` |

### 5.3 `ppo.yaml` (M1, 읽는 곳: `learning/train_ppo.py`의 `train`, `_warm_start_actor`, `_selection_config`)

| key | 기본값 | key | 기본값 |
|---|---|---|---|
| `seed` / `hidden_size` | 7 / 128 | `learning_rate` | 0.0003 |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | `clip_ratio` | 0.2 |
| `entropy_coef` / `value_coef` | 0.005 / 0.5 | `epochs` / `minibatch_size` | 4 / 64 |
| `rollout_steps_smoke` / `total_timesteps_smoke` | 128 / 256 | `rollout_steps_full` / `total_timesteps_full` | 1024 / 100000 |
| `curriculum_enabled` | true | `curriculum_straight_fraction` / `curriculum_corner_fraction` | 0.30 / 0.60 |
| `warm_start_epochs` / `warm_start_learning_rate` | 15 / 0.001 | `warm_start_log_std` | -1.5 |
| `critic_only_steps` | 10000 | `fine_tune_learning_rate` | 0.00005 |
| `bc_anchor_coef` / `fine_tune_entropy_coef` | 0.5 / 0.0 | `evaluation_interval_updates` | 10 |
| `selection_config` | `evaluation_selection` | `value_normalization` / `target_kl` | true / 0.01 |

warm start batch는 `minibatch_size`(64)를 쓴다. M1 실험 정의는 `m1.yaml`(학습 seed 7, 17, 29, 43, 59)이다.

### 5.4 `ppo_v2.yaml` (읽는 곳: `rl/ppo.py`의 `train`)

| key | 기본값 | 비고 |
|---|---|---|
| `seed`, `robot_profile` | 7, `opticore_gazebo` | |
| `environment_overrides` | `shield_strategy: scan`, `shield_enabled: true`, `scenario_randomization.enabled: true`, `procedural.split: train`, `domain_randomization.enabled: true` | 학습 env |
| `policy.mode` / `residual_scale` | `residual` / 0.5 | |
| `network` | `encoder: mlp`, `hidden_sizes: [256, 256]`, `activation: tanh`, `log_std_init: -1.0`, `cnn: {channels: [16, 32], kernel_sizes: [5, 5], strides: [2, 2], state_hidden: 64}` | `resolve_architecture`가 기본값과 merge |
| `vector_env.num_envs` / `backend` | 8 / `subprocess` | smoke는 ≤ 2개, sync 강제 |
| `rollout_steps` | 256 | env당. smoke는 ≤ 128 |
| `total_timesteps` / `total_timesteps_smoke` | 2000000 / 512 | |
| `gamma` / `gae_lambda` | 0.99 / 0.95 | |
| `learning_rate` → `learning_rate_final` | 0.0003 → 0.00003 | 선형 |
| `clip_ratio` / `epochs` / `minibatch_size` | 0.2 / 10 / 256 | smoke는 epoch ≤ 2 |
| `entropy_coef` → `entropy_coef_final` | 0.001 → 0.0 | 선형 |
| `value_coef` / `max_grad_norm` / `target_kl` | 0.5 / 0.5 / 0.02 | stop 임계는 1.5배 |
| `value_normalization` / `torch_threads` | true / 1 | |
| `lagrangian` | `enabled: false`, `cost_limit: 0.02`, `lambda_init: 0.0`, `lambda_lr: 0.05`, `lambda_max: 10.0` | |
| `warm_start` | `dataset: null`, `epochs: 15`, `learning_rate: 0.001`, `log_std: -1.5`, `anchor_coef: 0.0` | dataset은 direct만, batch 256 하드코딩 |
| `curriculum.stages` | §4.7.3 표 | |
| `evaluation` | `protocol: evaluation_selection_v2`, `interval_updates: 20`, `episodes_per_scenario: 2`, `workers: 4`, `backend: subprocess` | 0이면 학습 중 평가·`best.pt` 없음. smoke는 `S0` 1 episode sync |
| `checkpoint_interval_updates` | 25 | `state.pt` 주기 |

## 6. 불변식과 검증

테스트 경로는 `packages/amr_lite/tests/` 기준이다.

| 불변식 | 보호하는 테스트 |
|---|---|
| 관측 shape·dtype·범위, reset 시 4 frame 동일, step 후 oldest→newest shift | `test_observation.py::test_observation_contract_and_frame_order` |
| 관측 파이프라인이 golden 벡터와 `1e-5` 이내 | `test_observation_golden.py::test_lite_reproduces_golden_observations` |
| 10 Hz scan hold, 행동·관측 지연 | `test_sensor_model.py::test_scan_hold_repeats_frames_between_ten_hertz_updates`, `test_action_delay_executes_the_stop_action_first`, `test_observation_delay_returns_the_previous_observation` |
| 행동 scaling 왕복, 가감속 제한 | `test_action.py::test_action_roundtrip_and_scaling`, `test_acceleration_and_deceleration_limits` |
| 기본(Phase 0/M1) env의 trajectory·reward 불변 | `test_legacy_parity.py::test_default_environment_reproduces_pre_change_rollouts` |
| 같은 seed 결정성 | `test_reproducibility.py::test_deterministic_reset_and_step`, `test_randomized_scenario_same_seed_replays_and_different_seed_changes` |
| v1 selection/test seed 비중첩 (v2 protocol은 검사 안 함) | `test_reproducibility.py::test_selection_and_test_seed_namespaces_do_not_overlap` |
| 학습 episode seed ≥ 1,000,000·고유, curriculum stage 선택 | `test_vector_env.py::test_curriculum_stages_and_training_seed_namespace` |
| sync == subprocess (진행률 변경 포함) | `test_vector_env.py::test_subprocess_and_sync_vector_envs_are_identical` |
| 병렬 평가의 episode 순서·결과 보존 | `test_vector_env.py::test_run_episodes_keeps_episode_order_across_workers`, `test_parallel_evaluator_matches_sequential_and_honours_robot_profile` |
| residual 초기 행동 == rule, `combine_residual` clip | `test_ppo_v2.py::test_residual_policy_starts_as_the_rule_baseline` |
| 네트워크 출력 shape(MLP·CNN), cost head, 잘못된 encoder 거부 | `test_ppo_v2.py::test_network_shapes_and_cost_head` |
| GAE의 episode 경계 처리 | `test_ppo_v2.py::test_gae_handles_episode_boundaries` |
| v2 smoke가 로드 가능한 `best.pt`·`final.pt`·`training.csv`를 만듦(residual MLP, direct CNN+Lagrangian) | `test_ppo_v2.py::test_smoke_training_writes_loadable_checkpoints` |
| direct warm start 실행 | `test_ppo_v2.py::test_direct_warm_start_from_teacher_dataset` (MSE < 1.0만 확인) |
| pause → resume 후 update 번호 연속 | `test_ppo_v2.py::test_paused_training_resumes_to_completion` |
| β=1이면 teacher 실행과 동일, risk weight는 켜졌을 때만 영향 | `test_dagger.py::test_beta_one_executes_the_teacher`, `test_risk_weighting_changes_training_only_when_enabled` |
| DAgger smoke round·β·best 선택 | `test_dagger.py::test_dagger_smoke_selects_a_round` |
| teacher → BC → M1 PPO smoke → 평가 산출물 | `test_smoke_pipeline.py::test_teacher_bc_ppo_and_evaluation_artifacts` |
| BC·M1 stable·v2 residual·v2 CNN(newest)의 NumPy runtime == torch | `test_deploy.py::test_numpy_runtime_reproduces_torch_policy`, `test_bundle_records_mode_profile_and_history` |

**테스트하지 않는 것**

- M1 stable PPO의 실제 갱신 경로(critic warmup, BC anchor, value normalization, KL stop, best 선정).
  테스트는 `warm_start_only`와 32 step 비-stable smoke뿐이라 "동결"은 코드 리뷰로만 지켜진다.
- PPO v2의 truncation bootstrap, λ 갱신·상한, 결합 advantage, value/cost 정규화, KL 조기 중단,
  lr/entropy schedule, best vs final 선정 논리(파일 존재만 확인).
- crash 후 resume(마지막 `state.pt`보다 진행된 상태에서 재개)과 resume 시 config 일치.
- trainer 안의 subprocess backend: `--smoke`가 sync를 강제해 smoke·테스트 모두 이 경로를 타지 않는다.
- `episode_split`의 누수 없음, `history_mode: newest`일 때 env 관측의 4 frame 동일성,
  `collect_dagger`의 `dagger_risk_only` 경로, `GymnasiumWarehouseEnv`.
- 참조 PPO 구현과의 수치 일치, 학습 수렴(PPO v2는 1M step 이상 실행한 적이 없다).

## 7. 확장 방법 (claude 제안사항)

### 7.1 새 encoder (예: attention, 다른 CNN)

1. `rl/networks.py`: `resolve_architecture`의 허용 목록과 `PolicyNetwork.__init__`의 `tower()`.
2. `deploy/export.py`의 `_actor_graph`·`_Graph.ops`: 새 module을 op로 기록. 현재 export 가능한 op는
   `linear`, `conv1d`(padding·dilation·group 없음), `tanh`, `relu`뿐이다.
3. `deploy/runtime.py`의 `PolicyRuntime.policy_actions`·`_run`: 새 graph type 구현. 지금은 `scan_cnn`이
   아닌 type을 모두 MLP로 실행하므로 알 수 없는 type을 **명시적으로 거부**하는 검사도 넣는다. 이전 runtime이
   해석 못 하는 graph면 `format_version`을 올린다.
4. `test_ppo_v2.py::test_network_shapes_and_cost_head`와 `test_deploy.py` fixture에 변형을 추가한다. ROS core는
   runtime만 쓰므로 runtime parity가 통과하면 바꿀 것이 없다.

### 7.2 새 알고리즘 (예: SAC)

- 재사용: `SlotSpec`·`make_vector_env`(curriculum·seed 체계), `combine_residual`, `evaluate_protocol`·
  `selection_score`, `checkpoint_payload`의 필드 관례, `info["safety_cost"]`.
- 작성: `rl/sac.py`(replay buffer, twin Q, target network, 온도). 시간 초과는 PPO v2처럼
  `final_observation`으로 bootstrap한다.
- 연결: 새 checkpoint `kind` → `learning/networks.py`의 `load_policy` dispatch와 `deploy/export.py`의
  `_actor_graph`(결정적 actor가 `tanh(mean)`이면 기존 `mlp`/`scan_cnn` graph 재사용 가능), `experiments/sweep.py`의
  trainer 등록표(`ppo_v2`, `dagger` 항목 형식), 새 CLI.

### 7.3 Recurrent 정책 (GRU/LSTM)

관측 계약(296, frame stack 4)은 그대로 두고 정책만 상태를 가진다.

- 학습: buffer에 hidden state와 episode 시작 flag를 저장하고 minibatch를 시간 순서가 유지되는 sequence
  chunk로 만든다. 현재 `train`은 transition을 무작위 순열로 섞으므로 갱신 loop를 새로 써야 한다.
- 평가: `run_episodes`는 정책에 episode 경계를 알리지 않는다. slot별 reset hook과 상태 보관이 필요하다.
- 배포: `PolicyRuntime`은 stateless다. 상태 API, graph 확장, `format_version` 증가, ROS core의 hidden state
  reset(새 경로·목표·fallback 복귀), sequence 단위 golden 검증이 필요하다.

### 7.4 Reward·cost·curriculum 변경

- reward와 `safety_cost`는 `envs/warehouse_env.py`의 `_reward_components`와 `step`에 있다. 기본값을 바꾸면
  `test_legacy_parity.py`가 깨지므로 새 항은 기본 0인 config 항목으로 추가한다.
- curriculum은 YAML만 바꾸면 된다. 시나리오 ID는 `envs/scenarios.py`의 `make_scenario`가 알아야 한다.

## 8. 주의할 점과 알려진 한계

1. **`step_cost`와 λ 갱신에 truncation bootstrap이 섞인다 (버그, 확신 높음).** `rl/ppo.py`의 `train`은
   `costs[step, index] += gamma * final_costs[...]` **뒤에** `costs.mean()`으로 λ를 갱신하고 `step_cost`를
   기록한다. timeout 1.0 s로 돌린 `/tmp` 확인 실행에서 env 비용 평균은 0.0이었지만 `step_cost`는
   0.0075–0.0096이었다. λ가 실제 비용이 아니라 cost critic 예측을 따라 움직일 수 있다. bootstrap 전 원시 비용
   평균을 따로 써야 한다(E3 전에 수정 권장, claude 제안사항).
2. **crash 후 resume하면 `training.csv` 행이 중복되고 `best.pt`가 후퇴할 수 있다 (확신 높음).** CSV는 매
   update append, `state.pt`는 25 update마다다. 확인 실행에서 update 열이 `1, 2, 2, 3, 4`가 됐다.
   `best_score`도 `state.pt` 시점으로 돌아가므로, 그 뒤 저장된 더 좋은 `best.pt`를 낮은 점수의 checkpoint가
   덮어쓸 수 있다. sweep은 `state.pt`가 있으면 자동으로 `--resume`한다.
3. **resume은 config 일치를 확인하지 않는다.** `state.pt`에 config가 없고 `config.json`은 덮어쓴다.
   direct+warm start는 resume마다 BC warm start를 다시 돌린 뒤 weight를 덮어쓴다. `wall_seconds`도 0부터다.
4. **residual 정책의 권한이 제한된다.** `a_v = clip(base + s·a_pi)`이므로 정지(`a_v = −1`)는 rule의
   `a_v ≤ −1 + s`일 때만 가능하다. `s = 0.5`면 rule 명령이 `v_max`의 25% 이하일 때만 멈출 수 있고, 각속도
   보정은 `±s·w_max`(gazebo에서 ±0.75 rad/s)다. rule이 정면 접근 장애물(D1)에 빠르게 다가갈 때 residual이
   단독으로 멈추지 못할 수 있다(추정). E2가 이 trade-off를 비교한다.
5. **학습은 Shield ON, 선정은 Shield OFF다.** `ppo_v2.yaml`은 scan shield ON으로 rollout하고
   `evaluation_selection_v2`는 `shield_modes: [false]`다. M1도 legacy shield ON 학습, v1 selection(OFF) 선정이다.
   정책이 shield에 기댄 행동을 배웠는지는 선정 점수에 드러나지 않는다(추정).
6. **학습 중 best 선정 seed가 sweep selection seed의 부분집합이다.** 학습 중은 20000–20001,
   `sweep --stage select`는 20000–20004를 쓴다. `best.pt`의 selection 점수는 `final.pt`보다 낙관적으로
   나올 수 있다(추정). test split은 분리돼 있다.
7. **v2에는 timestep 0 평가가 없다.** D-012의 "best = 초기 정책" 상황을 `best.pt`만으로는 알 수 없다. 초기
   정책(residual은 rule, direct+BC는 BC)과 따로 비교해야 한다 (claude 제안사항).
8. **M1에서 `--stable` 없이도 KL stop이 켜진다.** KL 검사는 `cfg.get("target_kl")`만 보며 `ppo.yaml`에
   0.01이 있다. `--stable` help("use separate actor/critic encoders, value normalization, and KL stop")와
   README는 KL stop이 stable 전용처럼 읽힌다. Phase 0 shared 실행 당시 config는 확인할 수 없다. D-012의
   "5 update마다 평가"와 달리 현재 `evaluation_interval_updates`는 10이다.
9. **M1 PPO는 고정 layout 3개로만 학습했다.** randomization OFF, `pose_perturbation` 없음이라 seed가
   episode를 바꾸지 않는다. 선정·test는 randomization ON 8개 시나리오다. M1 일반화 결과 해석 시 고려한다.
10. **`path_error`와 위험 상태 판정이 누적 평균 CTE를 쓴다.** 순간 경로 이탈을 놓치거나, 평균이 0.25를 넘은
    뒤에는 모든 상태가 위험으로 분류된다. clearance·pose 등 metadata도 관측보다 한 step 늦은 값이다. 기본
    DAgger loop(`risk_only: false`)에서는 BC 가중치의 clearance만 영향을 받는다.
11. **teacher·DAgger seed는 1,000,000 namespace 밖이다.** PLAYBOOK §0의 "코드가 강제"는 PPO v2 학습에만
    맞다. 기본 round 수에서는 selection/test seed와 겹치지 않는다.
12. **dataset 디렉터리를 재사용하지 않는다.** 파일 이름이 episode index 기반이라 다시 수집하면 덮어쓰고, 이전
    실행의 남은 파일도 `load_dataset`이 읽는다. 일부 파일에 `minimum_clearance`가 없으면 risk weight가 조용히 꺼진다.
13. **`evaluation.interval_updates: 0`이면 `best.pt`가 없다.** sweep의 `trial_complete`는 `best.pt`를 요구하므로
    이 trial은 완료로 인정되지 않고 매번 다시 학습된다. 이전 실행의 `best.pt`가 남은 디렉터리라면 그대로 남는다.
14. **로그 해석.** `approx_kl`은 중단을 일으킨 값이 아니라 마지막 적용 minibatch 값이라 항상 임계 이하로 보인다.
    `kl_stop`을 함께 본다. `log_std`에는 clamp가 없고 entropy 계수는 0까지 줄어든다.
15. **`python -m amr_lite.cli.train`에는 `--output`이 없다.** `packages/amr_lite/artifacts/` 아래
    (`datasets/teacher`, `checkpoints/*_smoke.pt`, `smoke/`)에 쓴다. `learning/*` CLI도 기본 출력이 같은 위치다.
