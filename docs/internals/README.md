# Internals — 코드가 실제로 하는 일

[ARCHITECTURE](../ARCHITECTURE.md)가 Lite와 ROS가 지켜야 할 **계약**을 정의한다면, 이
폴더는 그 계약을 구현한 **코드의 동작**을 서브시스템별로 설명한다. 각 문서는 같은 순서
(역할과 범위 → 파일 지도 → 동작 흐름 → 핵심 구성요소 → 설정 → 불변식과 검증 → 확장 방법 →
주의할 점과 알려진 한계)를 따른다. 기준 commit은 `4e83d29`, 작성일은 2026-10-14다.

내용은 코드를 읽고 작은 실행으로 확인한 사실이다. `(claude 제안사항)` 표시는 Claude가 제안한 규칙·권고·절차이며
각 문서의 "확장 방법" 절이 여기에 해당한다. 표기 규칙은 [handover](../handover/README.md)와 같다.

## 문서와 코드의 대응

| 문서 | 코드 | 담당 역할 (claude 제안사항) |
|---|---|---|
| [ENVIRONMENT](ENVIRONMENT.md) | `amr_lite/envs/`, `amr_lite/config.py`, `configs/env.yaml`, `configs/procedural.yaml`, `configs/robot_profiles/`, `assets/maps/` | R3 안전·환경 |
| [SENSORS_AND_SAFETY](SENSORS_AND_SAFETY.md) | `amr_lite/sensors/`, `amr_lite/planning/`, `configs/baseline.yaml` | R3 안전·환경 |
| [LEARNING](LEARNING.md) | `amr_lite/learning/`, `amr_lite/rl/`, `configs/{bc,dagger,ppo,ppo_v2}.yaml` | R2 학습 |
| [EVALUATION_AND_EXPERIMENTS](EVALUATION_AND_EXPERIMENTS.md) | `amr_lite/evaluation/`, `amr_lite/experiments/`, `amr_lite/cli/`, `configs/evaluation*.yaml`, `configs/sweeps/`, `configs/gates/`, `configs/paired/`, `experiments/` | R1 평가·실험 |
| [DEPLOY_AND_ROS2](DEPLOY_AND_ROS2.md) | `amr_lite/deploy/`, `integrations/ros2/amr_learned_planner/` | R4 ROS2, R5 Gazebo |

`amr_lite/` 경로는 `packages/amr_lite/amr_lite/`, `configs/`는 `packages/amr_lite/configs/`를 뜻한다.
역할 정의는 [향후 방향](../handover/DIRECTION_OPTIONS.md) §5에 있다.

## 모듈 의존 관계

실행 흐름은 `configs → envs(+sensors, planning) → learning/rl → evaluation/experiments → deploy →
ROS2`이지만, import 관계는 이 순서대로 층이 나뉘어 있지 않다. 2026-10-14 기준 subpackage
사이 import(`config` 제외)는 다음과 같다.

| subpackage | module 최상단 import | 함수 안 import(지연) |
|---|---|---|
| `sensors` | `envs` | — |
| `envs` | `learning`, `planning`, `sensors` | — |
| `planning` | `envs`, `learning` | — |
| `learning` | `envs`, `planning`, `rl`, `sensors` | `rl` |
| `rl` | `learning`, `planning` | `envs` |
| `evaluation` | `envs` | `rl` |
| `experiments` | — | `deploy`, `envs`, `evaluation`, `learning`, `planning`, `rl` |
| `deploy` | `experiments`, `learning` | `envs`, `planning`, `rl` |
| `cli` | 대부분 | — |

- `envs`, `sensors`, `planning`, `learning`, `rl`은 서로를 import하는 한 덩어리다. 개별
  submodule 단위로 import해서 순환 오류는 나지 않지만, 한 subpackage만 떼어 내거나 다른
  저장소로 옮기기는 어렵다.
- `deploy.runtime`은 torch를 import하지 않는다(`test_runtime_does_not_import_torch`). ROS
  컨테이너에 `amr_lite`를 `--no-deps`로 설치할 수 있는 이유다.
- ROS 패키지가 쓰는 `amr_lite` 모듈은 `deploy.runtime`, `envs.dynamics`,
  `envs.warehouse_scenarios`, `learning.action`, `learning.observation`, `planning.path_utils`,
  `planning.scan_shield`, `sensors.lidar`다. 복사본이 없으므로 이 모듈의 변경은 곧 ROS 쪽
  변경이다.
- `packages/`는 `integrations/`와 top-level `experiments/` 폴더를 import하지 않는다
  ([REPOSITORY_STRUCTURE](../REPOSITORY_STRUCTURE.md)). `amr_lite.experiments`는 패키지 안의
  실험 운영 모듈로 top-level `experiments/` 폴더와 다르다.

## 문서를 고칠 때 (claude 제안사항)

- 코드를 바꾸는 PR은 해당 internals 문서도 함께 고친다.
- 줄 번호 대신 파일 경로와 심볼 이름으로 가리킨다.
- 추정은 "(추정)"으로 표시한다. 측정값은 날짜와 조건을 함께 쓴다.
