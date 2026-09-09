# Repository Structure

## 목적

실행 가능한 코드, 완료된 실험 기록, 프로젝트 문서를 분리해 Phase가 바뀌어도 과거 결과를 보존한다. 디렉터리는 담당 인원이 아니라 기술적 수명과 의존성 방향을 기준으로 나눈다.

## 현재 구조

```text
opticore-amr-autonomy/
├── packages/
│   └── amr_lite/
│       ├── amr_lite/
│       ├── configs/
│       ├── tests/
│       ├── README.md
│       └── pyproject.toml
├── experiments/
│   └── phase0_amr_lite/
│       ├── README.md
│       ├── RESULTS.md
│       ├── DECISIONS.md
│       ├── KNOWN_LIMITATIONS.md
│       └── artifacts/
├── docs/
│   ├── ROADMAP.md
│   ├── NEXT_STEPS.md
│   └── REPOSITORY_STRUCTURE.md
├── README.md
└── LICENSE
```

## 디렉터리 원칙

### `packages/`

계속 개발하고 다른 구성요소에서 재사용할 실행 코드를 둔다.

- 패키지는 자체 설정, 테스트, 설치 정보를 가진다.
- 패키지 코드는 `experiments/`를 import하지 않는다.
- AMR-Lite의 새 기능과 수정은 `packages/amr_lite/`에서 진행한다.
- 향후 공통 모듈이 명확해지면 별도 패키지로 분리한다.

### `experiments/`

마일스톤별 실험 정의와 검토 가능한 근거를 둔다.

- Phase 0 결과는 `phase0_amr_lite/`에 보존한다.
- 새 마일스톤은 기존 폴더를 덮어쓰지 않고 새 폴더를 만든다.
- 예: `m1_reproducibility/`, `m2_dynamic_safety/`
- 결과표, 사용 config, 대표 그래프와 실패 분석을 함께 남긴다.
- 큰 중간 checkpoint와 재생성 가능한 dataset은 Git에 넣지 않는다.

### `docs/`

특정 실험 하나에 종속되지 않는 프로젝트 수준 문서를 둔다.

- 로드맵과 의사결정 gate
- 저장소·브랜치 운영 방식
- 공통 observation/action 계약
- 향후 ROS2/Gazebo 통합 설계

### 향후 `integrations/`

ROS2/Gazebo 연결을 시작할 때 추가한다.

```text
integrations/
  ros2/
    src/
    launch/
    config/
    tests/
```

ROS2 adapter는 `packages/`의 관측·행동 계약을 사용하되, AMR-Lite core가 ROS2에 의존하게 만들지 않는다.

## 의존성 방향

```text
experiments ──uses──> packages
integrations ─uses──> packages
docs          describes all areas
packages      must not depend on experiments or integrations
```

이 방향을 지키면 Lite 환경과 ROS2 runtime을 독립적으로 테스트할 수 있다.

## 실험 보존 규칙

- 기존 milestone의 원자료를 새 실행으로 덮어쓰지 않는다.
- checkpoint는 best와 final을 구분한다.
- 학습 seed와 평가 seed를 모두 기록한다.
- 결과에는 사용 commit과 config를 연결한다.
- 잘못된 기록을 수정할 때는 원본 삭제 대신 수정 이유를 문서에 남긴다.

## 브랜치 원칙

- `dev`가 기본 통합 브랜치다.
- 새 작업은 `dev`에서 `feature/*`, `fix/*`, `docs/*`로 분기한다.
- 기능 브랜치에는 한 기술 목표만 담는다.
- 병합 전 관련 단위 테스트와 smoke 검증을 실행한다.
- 별도의 안정 release 흐름이 필요해지는 시점에 release branch 정책을 다시 결정한다.
