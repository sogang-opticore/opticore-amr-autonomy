# Evaluation & Experiments Internals — 평가, 통계, 실험 운영

이 문서는 `packages/amr_lite`의 평가 루프, 통계, sweep runner, G4 gate, manifest, Lite paired 도구가
**코드에서 실제로 어떻게 동작하는지** 설명한다. 마일스톤과 gate 정의는 [ROADMAP.md](../ROADMAP.md),
실행 순서와 예산은 [EXPERIMENT_PLAYBOOK.md](../EXPERIMENT_PLAYBOOK.md), 계약은 [ARCHITECTURE.md](../ARCHITECTURE.md)를 따른다.
기준 코드는 2026-10-14 `docs/spike-handover` 브랜치(`4e83d29`)다.

## 1. 역할과 범위

| 포함 | 제외 (다른 문서) |
|---|---|
| paired 평가 루프, `episodes.csv`와 summary CSV | 환경·시나리오 생성 ([ENVIRONMENT.md](ENVIRONMENT.md)), shield 내부 판정 ([SENSORS_AND_SAFETY.md](SENSORS_AND_SAFETY.md)) |
| 실패 taxonomy가 기록되는 방식 | PPO v2·DAgger·M1 trainer 알고리즘 ([LEARNING.md](LEARNING.md)) |
| Wilson, hierarchical bootstrap, paired 비교 | bundle export·NumPy runtime ([DEPLOY_AND_ROS2.md](DEPLOY_AND_ROS2.md), [ARCHITECTURE.md](../ARCHITECTURE.md) §7) |
| protocol v1/v2 설정과 seed 규약 | ROS 노드·arbiter ([DEPLOY_AND_ROS2.md](DEPLOY_AND_ROS2.md)) |
| sweep runner, G4 gate, manifest, Lite paired·gap report, `experiments/` 구조 | |

이 서브시스템의 소유자는 보고된 숫자가 어떤 행·어떤 shield mode·어떤 seed에서 계산됐는지 설명할 수 있어야
한다 (claude 제안사항). 평가 의미를 바꾸는 변경은 §7.3의 protocol version 규칙을 따른다.

## 2. 파일 지도

경로는 `packages/amr_lite/` 기준이다.

| 파일 | 책임 |
|---|---|
| `amr_lite/evaluation/evaluator.py` | `evaluate`: policy × scenario × seed × shield mode 실행, `episode_row`로 행 생성, 순차(`_sequential_rows`)/병렬(`_parallel_rows`) 경로, CSV·`config.json`·그래프 기록 |
| `amr_lite/evaluation/metrics.py` | `summarize`, `summarize_by_scenario`, `summarize_by_training_seed`, `paired_comparisons`, `wilson_interval`, `_resample_stratified`, `_bootstrap_ci`, `write_csv` |
| `amr_lite/evaluation/failure_types.py` | `FAILURE_TYPES` 집합. 코드 어디서도 참조하지 않는다 |
| `amr_lite/evaluation/plots.py` | `make_plots`(성공률·충돌률·episode time 막대), `plot_training`(M1 trainer 곡선) |
| `amr_lite/envs/warehouse_env.py`의 `_failure_type`, `_info` | 실패 분류와 episode 지표의 실제 계산 위치 |
| `amr_lite/rl/evaluation.py` | `run_episodes`(lockstep 병렬 실행), `evaluate_protocol`·`aggregate`(selection 지표), `selection_score` |
| `amr_lite/experiments/sweep.py` | `expand_trials`, `Sweep`(plan/train/select/test), `rank_variants`, `apply_smoke_budget`, `load_sweep_spec` |
| `amr_lite/experiments/gate.py` | `evaluate_gate`, `write_gate_report` |
| `amr_lite/experiments/manifest.py` | `build_manifest`, `git_snapshot`, `source_snapshot`, `dataset_snapshot`, `sha256_file` |
| `amr_lite/experiments/paired.py` | `run_lite_paired`, `load_policy_spec`, `gap_report`, `write_gap_report`, `PAIRED_COLUMNS` |
| `configs/evaluation.yaml`, `evaluation_selection.yaml`, `evaluation_test.yaml` | dev 평가와 protocol v1 |
| `configs/evaluation_selection_v2.yaml`, `evaluation_test_v2.yaml` | protocol v2 |
| `configs/m1.yaml`, `configs/sweeps/*.yaml`, `configs/gates/g4.yaml`, `configs/paired/warehouse_v1.yaml` | M1 입력, M4 실험 spec, G4 기준, paired set |

### 2.1 CLI 명령 지도

`python -m amr_lite.cli.<모듈>`로 실행한다. 아래 표의 뒤쪽 다섯 개는 다른 문서가 다루므로 요약만 둔다.

| 모듈 | 하는 일 | 주요 인자 | 산출물 |
|---|---|---|---|
| `evaluate` | dev protocol(`evaluation.yaml`)로 rule·BC·PPO(+`bc_dagger` checkpoint가 있으면) paired 평가 | `--bc`, `--ppo`, `--dagger`(기본 smoke checkpoint), `--output`(기본 `artifacts/results`), `--quick`(S0·S3·D0 × 1) | `<output>/` CSV·`config.json`, `<output>/../plots/` |
| `m1` | M1 재현: seed별 M1 PPO 학습 → v1 test 평가 → manifest·보고서 | `--output`, `--smoke`, `--training-seeds`, `--test-episodes`, `--no-resume` | §3.2 |
| `sweep` | sweep `plan`/`train`/`select`/`test`. `all`(기본)은 plan+train+select | `spec`(이름 또는 YAML 경로), `--stage`, `--output`(기본 `artifacts/sweeps/<name>`), `--parallel`, `--smoke`, `--force` | §4.6 |
| `paired lite` | paired set을 Lite에서 실행 | `--policy`(`rule`·`bundle:<dir>`·`checkpoint:<pt>`, 기본 `rule`), `--set`(`warehouse_v1`), `--seeds`(`0 1 2`), `--legs`, `--label`(`lite`), `--robot-profile`(`opticore_gazebo`), `--shield on\|off`(`on`), `--workers`, `--output`(필수) | paired CSV, stdout JSON |
| `paired gap` | paired CSV 두 개 비교 | `--reference`, `--candidate`, `--output` | gap report MD, stdout JSON |
| `train` | Phase 0 smoke 전체(teacher 6 episode → BC → M1 PPO smoke → dev 평가 S0·S3·D0 → S3 데모) | 없음 | `artifacts/datasets/teacher/`, `artifacts/checkpoints/*_smoke.pt`, `artifacts/smoke/` |
| `train_rl` | PPO v2 학습 | `--config`(`ppo_v2`), `--set k=v`, `--output`, `--smoke`, `--resume`, `--max-updates` | `result.json`, `best.pt`, `final.pt`, `training.csv`, `config.json`, 진행 중 `state.pt` |
| `dagger` | BC → DAgger | `--config`(`dagger`), `--set`, `--output`, `--smoke` | `round_<k>/`, `rounds.csv`, `best.pt`, `result.json` |
| `export_policy` / `verify_bundle` | bundle 생성·검사 | `--checkpoint --output --robot-profile --selection` / `bundle --robot-profile` | bundle 디렉터리 / exit 0·1(로드 실패)·2(profile 불일치) |
| `demo` | 1 episode(최대 500 step) 최종 frame | `--policy rule\|bc\|ppo`, `--checkpoint`, `--scenario`(S3), `--seed`(42), `--output`, `--show` | PNG |

## 3. 동작 흐름

### 3.1 M4: train → select → test → gate (`amr_lite/experiments/sweep.py`의 `Sweep`)

```text
configs/sweeps/<name>.yaml ──expand_trials (variants × grid × training_seeds, budget)──┐
                                                                                     ▼
[plan]   plan.json, trials/<variant>-s<seed>/trial_config.json
[train]  ThreadPool(parallel_trials) → 각 trial subprocess:
         python -m <trainer> --config trial_config.json --output <trial> [--smoke] [--resume]
         완료 = result.json status=="complete" AND best checkpoint 존재 → 재실행 시 skip
[select] trial × role(best[, final]) → evaluate_protocol(selection v2, shield OFF)
         → selection_<role>.json(cache) → rank_variants → leaderboard.csv, selection.json(winner), SELECTION.md
[test]   test/DONE 있으면 거부 → rule + winner의 모든 seed × role → evaluate(test v2, shield OFF·ON)
         → paired_comparisons → evaluate_gate(gates/g4) → gate.json, GATE_REPORT.md, DONE
```

### 3.2 M1 오케스트레이션 (`amr_lite/cli/m1.py`의 `run_m1`)

```text
m1.yaml ─┬─ selection(v1) seed ∩ test(v1) seed ≠ ∅ → ValueError
         ├─ resolved_config.json
         ├─ seed마다 learning.train_ppo.train(selection_config=evaluation_selection, warm start=teacher dataset)
         │     → checkpoints/custom_stable_ppo_seed_<s>_final.pt + 학습 중 고른 best
         │     (_can_resume: final.pt의 seed·total_timesteps가 같으면 재학습 없이 metrics JSON 재사용)
         ├─ policies: lite_rule_baseline, bc(있으면), legacy_stable_ppo(있으면), custom_stable_ppo best/final × seed
         ├─ evaluate(policies, test/, evaluation_test) → test/*.csv, plots/
         └─ manifest.json(build_manifest), REPORT.md, run_result.json
```

### 3.3 Lite↔Gazebo paired (`amr_lite/experiments/paired.py`)

```text
configs/paired/warehouse_v1.yaml (map frame goal 8개, agent 4개)
 ├─ Lite:   cli.paired lite → run_lite_paired → WH-warehouse_v1-NN × seeds → lite.csv (PAIRED_COLUMNS)
 ├─ Gazebo: paired_runner.launch.py → goal_sequence_runner + core/recorder.py → gazebo.csv (같은 11열 + 추가 열)
 └─ cli.paired gap --reference A.csv --candidate B.csv → gap_report → GAP_REPORT.md
```

## 4. 핵심 구성요소 상세

### 4.1 Paired 평가 루프 (`evaluate`)

- **env 구성**: config에 `robot_profile`이 있으면 `resolve_env_config(profile, environment_overrides)`(v2),
  없으면 `env.yaml` + `environment_overrides`(v1).
- **seed**: `seeds` 목록이 있으면 그대로, 없으면 `seed + i`(`i < episodes_per_scenario`).
- **paired 설계**: 모든 policy와 shield mode가 같은 (scenario, seed)를 실행한다. instance는 (scenario,
  seed, env 설정)에서 결정적으로 생성되고 `scenario_instance_id`로 기록된다. policy별 env 차이는
  `observation_overrides`(현재 `history_mode`만)뿐이다. paired 비교의 key는 instance id가 아니라
  `(scenario_id, seed, shield_enabled)`이며 instance id 일치는 검사하지 않는다.

| 항목 | 순차 `_sequential_rows` | 병렬 `_parallel_rows` |
|---|---|---|
| 선택 조건 | `workers` ≤ 1 (기본) | `workers` > 1 (sweep test는 4) |
| 순서 | scenario → seed → shield → policy, episode마다 새 `WarehouseEnv` | shield × policy마다 `run_episodes`(lockstep vector env, `backend` 기본 `subprocess`) 후 순차 순서로 재배열 |
| 종료 | env terminated/truncated 또는 `max_steps` | env terminated/truncated만. `max_steps` 무시 |
| 추론 | `policy.predict` 1개씩 | `predict_batch`가 있으면 batch |
| 행 생성 | `episode_row` (공통) | `episode_row` (공통) |

두 경로는 timing 열(`inference_ms`, `simulation_steps_per_second`)을 빼고 같은 행을 내도록 설계됐다. 조건은
env timeout ≤ `max_steps × policy_dt`다. v1은 360 × 0.05 = 18.0 s = `episode_timeout` 18.0이고 360번째
step에서 `elapsed`가 `18.000000000000014`가 되어 truncated된다(실측). v2는 2400 × 0.05 = 120 s이고,
`opticore_gazebo`의 timeout은 `max(18.0, 6.0 + 2.0 × A* 길이)`로 test v2 instance(seed 30000–30019)
최댓값이 90.5 s(`WH-warehouse_v1-07`)였다. 신경망 정책은 batch 추론 때문에 연속 열이 ~1e-8 수준에서
달라질 수 있다(§8.8).

출력: `<output>/episodes.csv`, `summary.csv`, `summary_by_training_seed.csv`, `summary_by_scenario.csv`,
`paired_comparisons.csv`(비교 행이 있을 때만), `config.json`(`evaluation`, `environment`), 그리고
`<output>/../plots/*.png`. 그래프는 출력 디렉터리의 **부모**에 생긴다(sweep은 `<sweep>/plots/`).

### 4.2 `episodes.csv` 열 (47개)

| 묶음 | 열 | 정의 (코드 기준) |
|---|---|---|
| 식별 | `policy`, `policy_family`, `training_seed`, `checkpoint_role`, `checkpoint_sha256` | policy 객체 속성. 없으면 family=dict key, role=`fixed` |
| instance | `scenario_id`, `seed`, `scenario_instance_id`, `evaluation_split` | split은 config `split`, 없으면 `unspecified` |
| 조건 | `shield_strategy`, `shield_enabled` | 실제 적용된 env 값 |
| 결과 | `success`, `collision`, `terminated`, `truncated`, `failure_type` | §4.3 |
| 효율·경로 | `episode_time`, `steps`, `path_length`, `path_length_ratio`, `mean_cross_track_error`, `max_cross_track_error` | ratio = 주행 거리 / A* 길이 |
| 안전 여유 | `minimum_clearance`, `minimum_predicted_ttc`, `safety_cost` | clearance = episode 최소 (정적 거리장, 동적 중심거리−반경). 충돌 시 음수 가능. cost는 step마다 `collision + max(0, safety_distance − clearance)` 합 |
| 행동 | `idle_time`, `angular_oscillation`, `total_abs_rotation`, `action_smoothness` | idle = \|v\|<0.03인 시간, oscillation = \|w\|>0.15에서 부호 전환 수, rotation = Σ\|w\|·dt, smoothness = 연속 행동 차이 제곱합 평균 |
| shield 모드 | `shield_pass`, `shield_clamp`, `shield_stop`, `shield_interventions`(=CLAMP+STOP), `shield_slow`, `shield_evade`, `shield_reverse`, `shield_sustained_time`, `shield_risk_counts`(JSON) | sustained = 최장 연속 non-PASS step × `policy_dt` |
| shield 감사 | `unsafe_without_shield`, `shield_false_positive`, `shield_false_negative`, `shield_actual_false_negative`, `shield_ineffective_intervention`, `tracking_speed_error_mean/max` | `unsafe_without_shield`는 shield OFF에서도 집계된다. 충돌 step의 mode가 PASS면 actual FN, 아니면 ineffective |
| 성능 | `inference_ms`, `simulation_steps_per_second` | 실행마다 달라진다 |

`experiments/phase0_amr_lite/artifacts/*/episodes.csv`는 28–29열의 이전 schema다.

### 4.3 실패 taxonomy

`failure_type`은 `WarehouseEnv._failure_type`이 **매 step** 계산하고 마지막 step 값이 기록된다. 위에서부터
처음 맞는 조건이 이긴다.

| 순위 | 값 | 조건 | 종료 |
|---:|---|---|---|
| 1 | `NO_PATH` | `global_path`가 비어 있음 | terminated |
| 2 | `INVALID_ACTION` | 행동 shape ≠ (2,) 또는 non-finite (0 행동으로 대체) | terminated |
| 3 | `COLLISION_DYNAMIC` / `COLLISION_STATIC` | 충돌. 최종 위치가 어떤 동적 장애물과 `collision_radius + radius` 이내면 DYNAMIC | terminated |
| 4 | `SHIELD_SATURATION` | `shield_counts["STOP"] > max(20, step_count // 2)` | 종료 조건 아님 |
| 5 | `OSCILLATION` | `angular_oscillation > 12` | 〃 |
| 6 | `CIRCLING` | `total_abs_rotation > 4π` | 〃 |
| 7 | `STUCK` | `idle_time > 0.6 × episode_timeout` | 〃 |
| 8 | `TIMEOUT` | `elapsed ≥ episode_timeout` | truncated |
| — | `""` | 해당 없음 (보통 성공) | |

- 4–7은 `success`를 확인하지 않는다. 성공 episode도 이 label을 가질 수 있다(selection v2 rule 실행에서는
  성공 114건 중 0건, 최대 rotation 7.37 rad, 최대 oscillation 10).
- `timeout_rate`는 `truncated` 평균이므로 `TIMEOUT`뿐 아니라 truncated된 4–7을 모두 포함한다.
- `PATH_LOST`는 Lite에서 만들어지지 않는다. `ENVIRONMENT_ERROR`는 info에 키가 없을 때의 기본값이라 실제로
  나오지 않는다. Gazebo recorder는 `TIMEOUT`, `COLLISION_PROXY`, `""`만 쓴다.
- `rl/evaluation.py`의 `aggregate`는 `failure_type or ("SUCCESS" if success else "UNFINISHED")`로
  `failure_counts`를 센다.

### 4.4 통계 (`amr_lite/evaluation/metrics.py`)

| 파일 | group key | CI |
|---|---|---|
| `summary.csv` | (`policy_family`, `checkpoint_role`, `shield_enabled`) — 학습 seed를 합친다 | Wilson + bootstrap |
| `summary_by_scenario.csv` | 위 + `scenario_id` | Wilson + bootstrap |
| `summary_by_training_seed.csv` | 위 + `training_seed` | Wilson, bootstrap 없음(resamples 0 → CI = 점추정) |
| `paired_comparisons.csv` | (`comparison`, `policy`, `checkpoint_role`, `shield_enabled`) | bootstrap |

binary 지표: `success_rate`(success), `collision_rate`(collision), `timeout_rate`(truncated).
continuous: `mean_minimum_clearance`, `mean_episode_time`. 점추정은 group 전체 행의 단순 평균이다.

**Wilson**: `wilson_interval(successes, count, z=1.959963984540054)`. group 전체 episode를 독립으로 본다.

**Scenario-stratified hierarchical bootstrap** (`_bootstrap_ci` + `_resample_stratified`), 반복 B =
`bootstrap_resamples`:

```text
rng = np.random.default_rng(seed)
for b in 1..B:
    T = group 안의 비어 있지 않은 training_seed 집합 (정렬)
    draws = T에서 |T|개 복원추출   (T가 비면 [None] = 전체 행)
    sample = []
    for s in draws:                         # 중복 seed는 중복 포함
        rows_s = training_seed == s 인 행 (None이면 전체)
        for c in sorted(rows_s의 scenario):  # scenario는 고정 층, 재표본하지 않음
            stratum = rows_s 중 scenario == c (n_c개)
            sample += stratum에서 n_c개 균등 복원추출
    est[b] = mean(value(sample))
CI = np.quantile(est, [0.025, 0.975])
```

평가 seed는 별도 층이 아니라 층 안의 episode로 재표본되며 학습 seed 간 짝을 유지하지 않는다. seed는
`bootstrap_seed + crc32("family:role:shield")`(scenario 요약은 `:scenario` 추가)에 지표마다
`crc32(지표명)`을 더한다. 같은 입력이면 CI가 bit 단위로 재현된다.

**Paired 비교** (`paired_comparisons`, reference 기본값 `lite_rule_baseline`):

- `policy_minus_rule`: reference가 아닌 각 행과 같은 `(scenario_id, seed, shield_enabled)`의 reference 행.
  delta = `success`, `collision`, `truncated`(→`timeout_delta`), `minimum_clearance`의 차(후보 − rule).
- `shield_on_minus_off`: 같은 `(family, role, training_seed, scenario_id, seed)`의 ON 행 − OFF 행. rule 포함.
- group별 delta 평균과 위 bootstrap CI(`pairs` = delta 행 수). seed는
  `bootstrap_seed + crc32("comparison:policy:role:shield") + crc32(field)`.

### 4.5 Seed 범위 정책

| 용도 | 범위 | 정의 위치 | 코드 강제 |
|---|---|---|---|
| M1 학습 seed | 7, 17, 29, 43, 59 | `configs/m1.yaml` | 없음. "20000 미만"은 ROADMAP §9 규약 |
| M1 학습 episode | `training_seed + episode_number` | `learning/train_ppo.py`의 `train` | 없음 |
| PPO v2 학습 episode | `1_000_000 + SeedSequence(training_seed, slot, counter) mod (2^31−1−1_000_000)` | `rl/vector_env.py`의 `training_episode_seed` | 구조적으로 강제 |
| teacher·DAgger 수집 | `seed + episode`, `seed + 1000·round + episode` | `learning/collect_teacher.py`, `learning/dagger_loop.py` | 없음. 1,000,000 namespace 밖 |
| dev 평가 | 1000+ (2개) | `evaluation.yaml` | — |
| selection | 20000–20004 | `evaluation_selection*.yaml` | `run_m1`이 test와 겹치면 `ValueError` (v1만) |
| PPO v2 학습 중 selection | 20000–20001 (`episodes_per_scenario: 2`) | `configs/ppo_v2.yaml`의 `evaluation` | — |
| test | 30000–30019 | `evaluation_test*.yaml` | 위와 같음 |
| M1 fallback selection | `training_seed + 50_000` | `train_ppo._selection_config` | `selection_config`가 없을 때만 |
| bootstrap | 20260914(v1), 20261013(v2) | YAML | — |

[EXPERIMENT_PLAYBOOK.md](../EXPERIMENT_PLAYBOOK.md) §0의 "학습 episode seed는 1,000,000부터 … 코드가
강제"는 PPO v2 vector env에만 맞다.

### 4.6 Sweep runner

**Spec** (`load_sweep_spec`: `.yaml` 경로가 있으면 그 파일, 아니면 `configs/sweeps/<name>`):

| 키 | 의미 |
|---|---|
| `name` | 기본 출력 `artifacts/sweeps/<name>` |
| `trainer` | `ppo_v2`(기본, `best.pt`·`final.pt`) 또는 `dagger`(`best.pt`만). 그 외는 `ValueError` |
| `base_config` | trainer 기본 설정 이름 |
| `variants` | `deep_merge(base, overrides)`. 없으면 `{"default": {}}` |
| `grid` | dotted key → 값 목록의 곱. variant 이름에 `__<key 마지막 조각>=<값>`이 붙는다 |
| `training_seeds` | trial마다 `config["seed"]` |
| `budget` | dotted key. variant·grid 뒤에 적용되므로 같은 key면 budget이 이긴다 |
| `parallel_trials` | 동시 trial 수. `--parallel`이 우선 |
| `selection` | `protocol`(기본 `evaluation_selection_v2`), `workers`(4), `backend`(`subprocess`), `evaluate_final`(true), `rule.constraints.{collision_rate_max, timeout_rate_max}`(없으면 1.0) |
| `test` | `protocol`(`evaluation_test_v2`), `workers`(4), `backend`, `include_final`(true), `gate`(`gates/g4`) |

**Trial 디렉터리** `trials/<variant>-s<seed>/`: `trial_config.json`(plan마다 다시 씀), `train.log`(append),
trainer 산출물(`config.json`, `training.csv`, `best.pt`, `final.pt`, `result.json`, 진행 중 `state.pt`),
select 산출물 `selection_best.json`, `selection_final.json`. `plan.json`은 `spec`, `smoke`, trial 목록을 담는다.

**Train**: 미완료 trial만 `ThreadPoolExecutor(parallel)`로 subprocess 실행한다(cwd=`packages/amr_lite`,
`PYTHONPATH`에 ROOT 추가, `MPLCONFIGDIR` 기본값). `state.pt`가 있으면 `--resume`을 붙인다. 하나라도 exit ≠ 0이면
모두 끝난 뒤 `RuntimeError`. trial마다 PPO v2 vector env worker(`num_envs: 8`, subprocess)가 따로 뜬다.

**Select**: 모든 trial이 완료돼야 한다. `selection_<role>.json`이 있으면 재평가 없이 재사용한다.
`evaluate_protocol`은 첫 shield mode의 지표를 반환하므로 v2(`shield_modes: [false]`)에서는 shield OFF다.
smoke는 시나리오 앞 2개 × 1 episode, worker 1, `sync`다.

**Ranking** (`rank_variants`, `best` 행만 사용):

```text
median_success = median(best success_rate)   worst_success = min(...)
median_collision, median_timeout, median_time(mean_time), median_shield_intervention_rate = median(...)
max_best_final_gap = max_seed(best − final success)      # 보고만, 순위에 쓰지 않음
eligible = median_collision ≤ collision_rate_max AND median_timeout ≤ timeout_rate_max
정렬 = (eligible, median_success, worst_success, −median_collision, −median_time) 내림차순
winner = 첫 eligible variant (없으면 None → test 거부)
```

동점은 stable sort라 variant 이름 알파벳 순이 남는다. `failure_counts`는 순위에 쓰지 않으며 CSV에서도
dict라 빠진다. 산출물은 `selection_trials.csv`, `leaderboard.csv`, `selection.json`(`winner`, `protocol`,
`leaderboard`), `SELECTION.md`다.

**Test와 1회 보호**: `test/DONE`이 있고 `--force`가 없으면 `RuntimeError`. rule baseline
(`RuleBasedPlanner(protocol env, baseline.yaml)`)과 winner의 모든 seed × role(`best`, `include_final`이면
`final`)에 `policy_family=winner`, `checkpoint_sha256`을 붙여 `evaluate`한다. smoke는 시나리오 앞 2개(S1,
S2) × 1 episode, bootstrap 50이다. `DONE`(`{"winner", "smoke"}`)은 **마지막에** 쓴다.

### 4.7 G4 gate (`evaluate_gate`)

후보 = winner family의 `best` 행. reference = `g4.yaml`의 `reference` family(role 무관).

| check | shield | 데이터 | 기준 | 계산 |
|---|---|---|---|---|
| `success_delta_ci_low` | OFF | `policy_minus_rule` 비교 행 | ≥ −0.02 | 성공 delta의 bootstrap 95% CI 하한 |
| `collision_delta_ci_high` | OFF | 〃 | ≤ 0.01 | 충돌 delta CI 상한 |
| `improvement_dynamic_success_delta` | OFF | episode 행 | delta ≥ 0.05 **또는** ratio ≤ 0.95 | delta = `dynamic_scenarios` 행의 후보 성공률(전체 seed 합산) − rule 성공률, 비paired 점추정. ratio = 전체 시나리오에서 rule이 성공한 (scenario, seed)에 대해 후보도 성공한 행의 평균 시간 / 같은 행의 rule 평균 시간. value 열은 delta, ratio는 note |
| `worst_seed_gap` | OFF | 후보 best | ≤ 0.05 | seed별 성공률의 median − min. seed ≥ 2일 때만 |
| `best_final_gap` | OFF | 후보 best·final | ≤ 0.10 | seed별 (best − final) 성공률의 최댓값. final 행이 있을 때만 |
| `behaviour_{circling,oscillation,stuck}_rate` | OFF | 후보 best | 각 ≤ 0.02 | `failure_type == 이름` 비율, 분모는 성공 포함 전체 |
| `shield_on_saturation_rate` | ON | 후보 best | ≤ 0.05 | `failure_type == "SHIELD_SATURATION"` 비율. ON 행이 있을 때만 |

paired 비교 행이 없으면 `non_inferiority` check가 nan으로 실패한다. 후보나 reference의 OFF 행이 없으면
`ValueError`. `passed = all(check.passed)`. `GATE_REPORT.md`는 `**Result: PASS|FAIL**`, check 표(값 소수 4자리,
✅/❌, note), episode 수를 쓴다. ROADMAP G4의 profile 확인, bundle `1e-5` parity, shield ON 개입률 보고는
gate 코드에 없다.

### 4.8 Manifest provenance (`build_manifest`)

| 필드 | 내용 |
|---|---|
| `schema_version`, `created_at_utc` | 1, UTC ISO 시각 |
| `experiment` | `m1.yaml` + `mode`(full/smoke) + `training_results` |
| `git` | `commit`, `branch`, `dirty`(저장소 전체 status), `status`, `scope`(`packages/amr_lite`), `scope_dirty`, `scope_status` |
| `source` | `packages/amr_lite`의 `amr_lite/**/*.py`, `configs/**/*.yaml`, `tests/**/*.py`, `tests/golden/*.npz`, `assets/**/*.pgm\|yaml`, `docs/*.md`, `README.md`, `pyproject.toml`의 경로+hash를 정렬해 묶은 `sha256`과 파일 목록 |
| `dataset` | `version`, 절대 `root`, 직계 `*.npz`의 묶음 `sha256`과 목록 |
| `evaluation` | selection·test config |
| `checkpoints` | 존재하는 파일만 절대 경로·`sha256`·`bytes` |
| `runtime` | Python, 실행 파일, platform, 7개 패키지 버전, command |

`build_manifest`를 호출하는 곳은 `cli/m1.py`뿐이다. bundle export는 `git_snapshot`의 commit·branch·dirty만
기록한다. sweep은 manifest를 쓰지 않는다. 보존된 M1 manifest는 이관 전 값(`scope: amr_lite_spike`, branch
`codex/amr-lite-m1-reproducible-baseline`, commit `6a102ee4…`, `dirty: true`, `scope_dirty: false`,
source 62개, checkpoint 12개, Python 3.11.9, torch 2.8.0, numpy 2.3.2)과 원래 위치의 절대 경로를 담는다.

### 4.9 Lite paired runner와 gap report

- `run_lite_paired`: env = `resolve_env_config(robot_profile, {shield_strategy: scan, shield_enabled,
  episode_timeout: goal_timeout_sec(120.0), episode_timeout_per_meter: 0.0} + overrides)`. episode는
  seed × leg, 이름은 `WH-<set>-<NN>`. leg k는 독립 episode이며 leg k−1의 goal에서 A* 경로 끝 방향으로
  시작한다(`warehouse_scenarios._leg_plan`). 실제 도착 pose를 잇는 Gazebo와 다르다.
- 행은 `PAIRED_COLUMNS` 11열. `shield_intervention_rate = (CLAMP+STOP) / episode_length`.
- Gazebo recorder(`integrations/ros2/amr_learned_planner/amr_learned_planner/core/recorder.py`)의
  `collision`은 LiDAR 최근접점이 `robot_radius + collision_clearance` 안에 들어온 proxy이고 episode를
  끝내지 않는다. `seed` 열은 run 번호다.
- `gap_report`: leg(두 CSV의 `scenario_id` 합집합)마다 episode 수, 성공률과 delta, 충돌률, 성공 episode
  평균 시간, 평균 최소 clearance, candidate 실패 유형 수. overall은 pooled 성공·충돌률과
  `time_ratio = mean(leg별 candidate 평균 시간) / mean(leg별 reference 평균 시간)`(양쪽 모두 성공이 있는
  leg만). episode 단위 짝짓기와 CI는 없다.

### 4.10 `experiments/` 디렉터리

| 경로 | 내용 | Git에 없는 것 |
|---|---|---|
| `phase0_amr_lite/` | `README.md`, `RESULTS.md`, `DECISIONS.md`, `KNOWN_LIMITATIONS.md` | |
| `phase0_amr_lite/artifacts/{full,reward_v2,ppo_best,ppo_stable}/` | `results/`(또는 `shield_v2_results/`)의 `config.json`·`episodes.csv`(128행, 이전 schema)·`summary.csv`, `plots/`, `videos/` | smoke, dataset |
| `phase0_amr_lite/artifacts/checkpoints/` | `ppo_stable_full_best.pt`(+metrics, training CSV). `m1.yaml`의 `legacy_ppo_checkpoint` | final checkpoint |
| `m1_reproducibility/` | `README.md`, `RESULTS.md`(공식 수치) | |
| `m1_reproducibility/artifacts/` | `REPORT.md`, `manifest.json`, `resolved_config.json`, `plots/`, `test/{config.json, summary.csv(10행), summary_by_scenario.csv(80), summary_by_training_seed.csv(26), paired_comparisons.csv(13)}` | `episodes.csv`(4,160), checkpoint, dataset, `run_result.json` |

M4 결과는 [EXPERIMENT_PLAYBOOK.md](../EXPERIMENT_PLAYBOOK.md) §6에 따라 `experiments/m4_learned_planner/`처럼
새 디렉터리에 둔다.

## 5. 설정

### 5.1 Protocol 비교

| 항목 | dev `evaluation` | v1 `evaluation_selection` | v1 `evaluation_test` | v2 `evaluation_selection_v2` | v2 `evaluation_test_v2` |
|---|---|---|---|---|---|
| `protocol` / `split` | — / — | — / selection | — / test | v2 / selection | v2 / test |
| `seed` × `episodes_per_scenario` | 1000 × 2 | 20000 × 5 | 30000 × 20 | 20000 × 5 | 30000 × 20 |
| `scenarios` | S0–S5, D0, D1 (8) | 같음 (8) | 같음 (8) | S1–S5, D0–D4, P-AISLE·HALL·PINCH·JUNCTION, PD-MIXED, WH-RANDOM, WHD-RANDOM (17) | S1–S5, D0–D4, P-AISLE·HALL·PINCH·JUNCTION, PD-AISLE·HALL·PINCH·JUNCTION, WH-RANDOM, WHD-RANDOM, WH-warehouse_v1-00…07 (28) |
| `max_steps` | 360 | 360 | 360 | 2400 | 2400 |
| `shield_modes` | [false, true] | [false] | [false, true] | [false] | [false, true] |
| robot profile / shield | `env.yaml`(lite_default) / `legacy` | 같음 | 같음 | `opticore_gazebo` / `scan` | `opticore_gazebo` / `scan` |
| `scenario_randomization` | env 기본(false) | true | true | true | true |
| `procedural.split` | — | — | — | validation | test |
| bootstrap resamples / seed | 2000 / 20260914 (코드 기본) | 2000 / 20260914 | 2000 / 20260914 | 2000 / 20261013 | 2000 / 20261013 |
| `progress_every` | — | 0 | 200 | 0 | 500 |
| policy·mode당 episode | 16 | 40 | 160 | 85 | 560 |

M1 test 4,160 = 160 × 2 mode × 13 policy 행 묶음(rule, BC, legacy, 5 seed × best/final)이다. E6 test는
(rule 1 + 5 seed × 2 role) × 2 mode × 560 = 12,320 episode다(계산). 키 메모: `protocol`은 표기만 하고 어디서도
검증하지 않는다. `max_steps`는 순차 경로만 쓴다. `seeds` 목록이 있으면 `seed`·`episodes_per_scenario`보다
우선한다. `workers`·`backend`는 YAML에 없고 sweep test가 주입한다. bootstrap 기본값은 2000 / 20260914다.

### 5.2 `configs/gates/g4.yaml` (원문)

```yaml
reference: lite_rule_baseline
non_inferiority:              # shield OFF, paired by scenario instance and seed
  success_delta_ci_low_min: -0.02
  collision_delta_ci_high_max: 0.01
improvement:                  # at least one must hold (shield OFF, point estimates)
  dynamic_scenarios: [D0, D1, D2, D3, D4, PD-AISLE, PD-HALL, PD-PINCH, PD-JUNCTION, WHD-RANDOM]
  dynamic_success_delta_min: 0.05
  mean_time_ratio_max: 0.95   # candidate / reference mean episode time on episodes both solved
robustness:                   # across training seeds, shield OFF
  worst_seed_gap_max: 0.05    # median success - worst success
  best_final_gap_max: 0.10    # per seed: best success - final success
behaviour:
  failure_types: [CIRCLING, OSCILLATION, STUCK]
  max_rate: 0.02
shield_on:
  saturation_rate_max: 0.05
```

### 5.3 Sweep spec (`configs/sweeps/`)

모두 `parallel_trials: 2`, selection·test protocol v2, `workers: 4`다.

| spec | trainer | seeds | budget | variants (grid) | 제약 collision / timeout | test |
|---|---|---|---|---|---|---|
| `e1_dagger` | dagger | 7, 17, 29 | — | `bc_only`{rounds 0}, `dagger3`{rounds 3}, `dagger3_unweighted`{rounds 3, bc.risk_weight 1.0} | 0.20 / 0.20, `evaluate_final: false` | `include_final: false` |
| `e2_policy_mode` | ppo_v2 | 7, 17, 29 | 1,000,000 | `residual_0p5`, `residual_0p3`, `direct_scratch`, `direct_bc` | 0.15 / 0.15 | 기본 |
| `e3_lagrangian` | ppo_v2 | 7, 17, 29 | 1,000,000 | `unconstrained`, `cost_0p01`, `cost_0p03` | 0.15 / 0.15 | 기본 |
| `e4_randomization` | ppo_v2 | 7, 17, 29 | 1,000,000 | `randomized_curriculum`, `no_randomization` | 0.15 / 0.15 | 기본 |
| `e5_representation` | ppo_v2 | 7, 17, 29 | 1,000,000 | `default` × `network.encoder` [mlp, cnn] × `environment_overrides.history_mode` [stack, newest] | 0.15 / 0.15 | 기본 |
| `e6_candidate` | ppo_v2 | 7, 17, 29, 43, 59 | 5,000,000 | `candidate: {}` (G3에서 채움) | 0.10 / 0.10 | `include_final: true`, `gate: gates/g4` |

### 5.4 `m1.yaml`, paired set

- `m1.yaml`: `experiment_version: m1-reproducible-baseline-v1`, `training_seeds: [7, 17, 29, 43, 59]`,
  `ppo_config: ppo`, `selection_config: evaluation_selection`, `test_config: evaluation_test`, dataset
  `teacher-phase0-v1`(`artifacts/datasets/teacher`), `bc_checkpoint: artifacts/checkpoints/bc_full.pt`,
  `legacy_ppo_checkpoint: ../../experiments/phase0_amr_lite/artifacts/checkpoints/ppo_stable_full_best.pt`,
  `output: artifacts/m1/m1-reproducible-baseline-v1`. `--smoke`는 selection·test를 S0·S3·D0 × 1(test는
  `--test-episodes`), bootstrap 100, `evaluation_interval_updates: 0`으로 줄인다.
- `paired/warehouse_v1.yaml`: map `opticore_warehouse`, frame `map`, start (0.0, −2.0, yaw 0.0),
  `goal_timeout_sec: 120.0`, leg 8개, agent 4개(person_1 0.6 m/s, forklift_1 0.4 m/s·반경 0.6, person_2·3
  0.4 m/s).

## 6. 불변식과 검증

| 불변식 | 보호 테스트 (`packages/amr_lite/tests/`) |
|---|---|
| 순차·병렬 평가 행이 timing 열 외 동일, robot profile 반영 (rule, S3·D0, `opticore_gazebo`+scan) | `test_vector_env.py::test_parallel_evaluator_matches_sequential_and_honours_robot_profile` |
| `run_episodes`가 worker 수와 무관하게 순서·결과 유지 | `test_vector_env.py::test_run_episodes_keeps_episode_order_across_workers` |
| protocol seed 전개 | `test_vector_env.py::test_protocol_episodes_expand_scenarios_and_seeds` |
| PPO v2 학습 episode seed ≥ 1,000,000 | `test_vector_env.py::test_curriculum_stages_and_training_seed_namespace` |
| v1 selection·test seed 비중첩 | `test_reproducibility.py::test_selection_and_test_seed_namespaces_do_not_overlap` |
| 같은 seed → 같은 instance id, 다른 seed → 다른 instance | `test_reproducibility.py::test_randomized_scenario_same_seed_replays_and_different_seed_changes` |
| manifest scope = `packages/amr_lite`, git 조회 성공 | `test_reproducibility.py::test_manifest_uses_autonomy_package_scope` |
| Wilson 구간이 관측 비율 포함 | `test_metrics.py::test_wilson_interval_contains_observed_rate` |
| bootstrap 재현성, CI가 점추정 포함 | `test_metrics.py::test_summary_has_reproducible_hierarchical_confidence_intervals` |
| gate PASS 경로와 보고서 | `test_gate_and_sweep.py::test_gate_passes_a_candidate_that_solves_dynamic_scenarios` |
| 개선 없음·CIRCLING → FAIL, 없는 family → `ValueError` | `test_gate_and_sweep.py::test_gate_fails_without_improvement_or_with_behaviour_flags` |
| variant × grid × seed 전개, budget, base 불변 | `test_gate_and_sweep.py::test_expand_trials_applies_variants_grid_budget_and_seeds` |
| 제약 → 성공률 순, best-final gap | `test_gate_and_sweep.py::test_rank_variants_applies_constraints_before_success` |
| train → select → test, 두 번째 test 거부 | `test_gate_and_sweep.py::test_sweep_trains_selects_and_tests_only_once` |
| `evaluate`가 `episodes.csv`·`summary.csv` 생성 | `test_smoke_pipeline.py::test_teacher_bc_ppo_and_evaluation_artifacts` |
| paired CSV 열 = Gazebo runner 열 | `test_paired.py::test_lite_paired_rows_match_the_gazebo_runner_schema` |
| gap report 수치 | `test_paired.py::test_gap_report_compares_legs` |
| `failure_type` `""`·`COLLISION_DYNAMIC`(기본 env rollout), `NO_PATH` | `test_legacy_parity.py::test_default_environment_reproduces_pre_change_rollouts`, `test_astar.py::test_no_path_is_classified` |

**테스트하지 않는 것**

- `paired_comparisons`의 delta·CI 값과 `shield_on_minus_off` 행(gate 테스트에서 간접 사용만), 계층
  bootstrap의 재표본 구조 자체, `summarize_by_scenario`·`summarize_by_training_seed`
- `COLLISION_STATIC`, `TIMEOUT`, `STUCK`, `OSCILLATION`, `CIRCLING`, `SHIELD_SATURATION`, `INVALID_ACTION` 판정
- gate의 비열등성 FAIL, `worst_seed_gap`·`best_final_gap`·`shield_on_saturation_rate` FAIL, seed 1개일 때의 생략
- 신경망 정책의 순차·병렬 동일성(rule만), `max_steps` < env timeout일 때의 경로 차이
- sweep 재개(완료 trial skip, `state.pt` → `--resume`), `--parallel` > 1, `--force`, smoke·full 혼용,
  selection cache 무효화
- `run_m1`·`_can_resume`(Runbook Gate 3 smoke로만 확인), `build_manifest`·`source_snapshot` 내용
- v2 selection·test seed 비중첩, `make_plots`, CLI 인자 처리, `gap_report`의 빈 leg·nan

## 7. 확장 방법 (claude 제안사항)

### 7.1 새 sweep spec

1. `configs/sweeps/<name>.yaml`을 §4.6 키로 쓴다. 비교 대상만 `variants`/`grid`에 두고 공통 값은
   `base_config`나 `budget`에 둔다. 같은 마지막 조각을 가진 grid key 두 개는 trial 이름이 모호해진다.
2. `python -m amr_lite.cli.sweep <name> --stage plan --output /tmp/<name>-plan`으로 trial id를 확인한다.
3. smoke는 반드시 별도 `--output`으로 돌린다(§8.3). 실제 실행 전 spec을 커밋한다.
4. 새 trainer는 `TRAINERS`에 `module`과 `checkpoints`(최소 `best`)를 등록한다. 그 CLI는 `--config`,
   `--output`, `--smoke`, (`state.pt`를 쓰면) `--resume`을 받고 완료 시 `result.json`(`status:
   complete`)과 best checkpoint를 남겨야 한다.

### 7.2 새 gate 기준

1. 열린 test에 쓰인 gate 파일은 고치지 않는다(`g4.yaml` 머리 주석). 새 파일(`configs/gates/<new>.yaml`)을 만든다.
2. `evaluate_gate`에 `_check(name, value, threshold, passed, note)`를 추가한다. `passed`는 모든 check의
   AND이므로 추가만 하면 판정에 들어간다. 조건부로 생략되는 check는 생략 사실을 note나 별도 check로 남긴다.
3. `tests/test_gate_and_sweep.py`에 PASS·FAIL 경우를 모두 추가한다.
4. sweep spec의 `test.gate`를 새 파일로 바꾼다.

### 7.3 새 protocol version ("bump" 규칙)

규칙 근거: `evaluation_selection_v2.yaml`("Changing this file requires a protocol version bump"),
`evaluation_test_v2.yaml`, `g4.yaml` 주석, `Sweep.test`의 거부 메시지, [ROADMAP.md](../ROADMAP.md) §6.
결과를 본 뒤 평가 시나리오·seed·shield mode·profile·gate를 바꾸거나 test를 다시 열어야 하면 다음을 따른다.

1. 기존 파일은 그대로 두고 `evaluation_selection_v3.yaml`, `evaluation_test_v3.yaml`(`protocol: v3`)을 만든다.
2. 이미 열린 test instance를 재사용하지 않도록 새 test seed 범위를 쓴다(권장, 코드·문서 규칙은 없음, claude 제안사항).
3. `test_reproducibility.py`에 v3 selection·test 비중첩 테스트를 추가한다(현재 v1만 검사).
4. sweep spec의 `selection.protocol`·`test.protocol`(필요하면 `test.gate`)을 바꾸고 **새 `--output`**을 쓴다.
   `DONE`과 `selection_<role>.json` cache는 디렉터리 단위이고 protocol을 검사하지 않는다.
5. 변경 이유를 결정 문서에 남긴다.

## 8. 주의할 점과 알려진 한계

### 8.1 G4 핵심 기준은 shield OFF, 배포는 shield ON

| 단계 | shield | 근거 |
|---|---|---|
| PPO v2 학습 rollout | ON (`scan`) | `configs/ppo_v2.yaml`의 `environment_overrides.shield_enabled: true` |
| PPO v2 학습 중 best 선택, sweep variant 순위 | OFF | `evaluation_selection_v2.yaml`의 `shield_modes: [false]`, `evaluate_protocol` |
| G4 비열등성·개선·robustness·behaviour | OFF, reference `lite_rule_baseline` OFF | `evaluate_gate`의 `select(..., False)`, `shield_enabled` 거짓인 paired 행 |
| G4의 shield ON 항목 | ON, saturation ≤ 0.05 하나 | `evaluate_gate`의 `shield_on_saturation_rate` |
| ROS 배포 | ON (`scan`) | `integrations/ros2/amr_learned_planner/config/learned_planner.yaml`의 `shield_enabled: true` |

비공식 측정(2026-10-14, selection v2 85 episode, `lite_rule_baseline`, 본 문서 작성 중 재실행해 같은 수치 확인):

| Shield | 성공 | `COLLISION_DYNAMIC` | `TIMEOUT` | `SHIELD_SATURATION` | 시나리오별 |
|---|---:|---:|---:|---:|---|
| OFF | 55 | 30 | 0 | 0 | D1–D4·PD-MIXED 25/25 충돌, D0 1/5, WHD-RANDOM 4/5. 충돌 시 선속도 0.415–1.376 m/s(순차 재실행) |
| ON | 59 | 6 | 9 | 11 (12.9%) | D1 5 충돌, D2 5 timeout, D3·D4 각 5 saturation, PD-MIXED 성공 2·충돌 1·timeout 2, WHD-RANDOM 성공 2·timeout 2·saturation 1 |

`RuleBasedPlanner.predict`는 전방 섹터 scan 최소 거리로 감속(`clearance_slow: 0.75`)하고
`clearance_stop: 0.12` 미만에서 제자리 회전할 뿐 이동 장애물을 예측·회피하지 않는다. 함의:

- 개선 기준(동적 성공률 +5%p)의 비교 대상은 shield 없이 동적 장애물을 피하지 못하는 기준선이다. 같은 측정에서
  `dynamic_scenarios`에 해당하는 시나리오(D0–D4, WHD-RANDOM)의 rule OFF 성공은 5/30이었다. 이 기준은
  낮은 기준선 대비로 판정된다.
- 비열등성의 충돌 상한 +1%p도 충돌률 35.3%(같은 측정) 수준의 reference 대비다.
- 배포 구성(shield ON)의 성공·충돌·timeout은 gate에 없다. 값은 `summary.csv`와
  `paired_comparisons.csv`의 shield ON 행에 계산돼 남는다. 같은 측정에서 rule 자체의 ON saturation은
  12.9%로 G4 상한 5%를 넘었다. saturation은 정책뿐 아니라 shield 동작에도 좌우된다(추정).
- 이 구성은 G3 동결 전 팀 결정 사항이다. 바꾸면 §7.3에 따라 protocol version을 올린다.

### 8.2 Selection split 재사용

PPO v2는 학습 중 seed 20000–20001로 `best.pt`를 고르고(`ppo_v2.yaml` `evaluation`), sweep은 같은
split(20000–20004)으로 variant를 고른다. selection 점수는 낙관적으로 편향될 수 있으며 비편향 추정은 test뿐이다.
`rank_variants`는 `max_best_final_gap`, behaviour label, shield ON을 순위에 쓰지 않는다. ROADMAP §8의 "행동
flag 시 후보 제외"는 selection이 아니라 test 이후 G4에서만 판정된다.

### 8.3 Sweep 운영 위험

- `--smoke`와 실제 실행의 기본 출력이 같다(`artifacts/sweeps/<name>`). smoke trial도 `status:
  complete`를 쓰고 `trial_complete`는 `smoke`나 config 변경을 확인하지 않으므로, 같은 디렉터리의 이후
  실제 `train`이 이를 건너뛴다. `select`도 smoke의 `selection_<role>.json`을 그대로 재사용한다.
- spec을 바꿔도 같은 trial id의 기존 checkpoint는 재학습되지 않는다(`trial_config.json`만 다시 쓴다).
- DAgger(`learning/dagger_loop.py`)는 재개 기능이 없어 중단된 E1 trial은 round 0부터 다시 돈다.
- `DONE`은 test 마지막에 쓰이므로 도중 중단은 `--force` 없이 재실행된다. `--force`는 사실상 완료된 test를
  다시 여는 우회이고, 보호는 출력 디렉터리 단위다(다른 `--output`이면 막지 않는다).
- smoke test는 test v2 앞 2개 시나리오(S1, S2)만 써서 동적 시나리오가 없다. 개선 delta가 nan이 되어
  time ratio ≤ 0.95가 아니면 FAIL이다(Runbook Gate 7 로컬 산출물에서 nan·FAIL 확인).
- `apply_smoke_budget` docstring은 "one variant per grid value"라고 하지만 seed 2개와
  `parallel_trials: 1`만 줄인다.

### 8.4 Provenance 공백

sweep은 manifest를 쓰지 않는다. commit·dirty·source hash·runtime이 sweep 산출물에 없고, checkpoint hash는
test `episodes.csv`의 `checkpoint_sha256`에만 남는다. `source_snapshot`은 `packages/amr_lite`만 hash하므로
`integrations/ros2` 변경은 source hash에 반영되지 않는다(저장소 전체 `dirty`에는 반영).

### 8.5 실패 taxonomy

성공 episode도 4–7 label을 가질 수 있고 G4 behaviour 분모는 성공 포함 전체다(§4.3). `timeout_rate`는
saturation·stuck을 포함한다. `FAILURE_TYPES`는 검증에 쓰이지 않아 새 label 오타를 잡지 못한다.

### 8.6 Gate 구현

- `paired_comparisons`의 reference는 `lite_rule_baseline`으로 고정이고 `g4.yaml`의 `reference`는 개선
  기준의 reference 행 선택에만 쓰인다. `reference`를 바꾸면 비열등성과 개선이 서로 다른 기준선으로 판정된다.
- `worst_seed_gap`(seed 1개)과 `best_final_gap`(final 없음, 예: E1)은 FAIL이 아니라 **생략**된다.
- `improvement_dynamic_success_delta`는 time ratio로 통과해도 value 열에 delta를 보여 준다.
- ROADMAP G4의 "행동 이상 flag가 없고"와 달리 코드는 유형별 2%까지 허용한다.

### 8.7 통계 해석

scenario는 고정 층이라 scenario 집합의 일반화 불확실성은 CI에 없다. 평가 seed는 층 안의 episode로만
재표본된다. Wilson은 학습 seed·scenario 군집을 무시하므로 bootstrap보다 좁을 수 있다(추정). 동적 개선
delta와 gap report에는 CI가 없다. paired key가 instance id를 확인하지 않으므로 policy별 env 차이가 instance를
바꾸면 짝이 조용히 어긋난다(현재 차이는 `history_mode`뿐).

### 8.8 순차·병렬 경로

병렬 경로는 `max_steps`를 무시한다. `max_steps × policy_dt`를 env timeout보다 짧게 잡으면 순차 경로의 행은
success·collision·truncated가 모두 0인 "미완료"가 되어 어떤 비율에도 잡히지 않는다. 신경망 정책은
`predict_batch` 때문에 연속 열이 다를 수 있다. smoke PPO v2 checkpoint로 S3·D0·P-AISLE 12 episode를 비교한
결과 `path_length`·`minimum_clearance` 등이 ~1e-8 차이였고 결과 열(success·collision·truncated·failure·
steps·time)은 같았다. 장기 episode에서 결과가 갈릴 가능성은 배제하지 못한다(추정).

### 8.9 Lite↔Gazebo gap report

Lite 충돌은 episode를 끝내지만 Gazebo recorder는 LiDAR proxy 충돌을 표시만 하고 episode를 끝내지 않는다. Lite leg는 계획된 pose에서
독립적으로 시작하고 Gazebo leg는 실제 도착 pose에서 이어진다. 두 CSV의 `seed`는 의미가 다르다(Lite 난수 seed,
Gazebo run 번호). `time_ratio`는 leg 평균의 평균 비율이며 episode 단위 paired 값이 아니다.
