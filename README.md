# OptiCore AMR Autonomy

OptiCore AMR의 경로 계획, 학습 기반 주행, 안전 제어와 향후 ROS2/Gazebo 연결을 위한 자율주행 저장소다.

현재 기준선은 ROS 없이 실행 가능한 AMR-Lite Phase 0 기술 스파이크다. Rule baseline, BC, DAgger-lite, PPO와 Stable PPO를 같은 관측·행동 계약으로 비교하며, Safety Shield ON/OFF 결과를 함께 기록한다.

## 저장소 구조

```text
packages/
  amr_lite/                 실행 가능한 AMR-Lite 패키지, 설정, 테스트
experiments/
  phase0_amr_lite/          Phase 0 결정·결과·한계와 선별 산출물
docs/
  ROADMAP.md                M0–M5 마일스톤과 의사결정 gate
  NEXT_STEPS.md             후속 기술 항목과 팀 결정 질문
  REPOSITORY_STRUCTURE.md   디렉터리와 의존성 원칙
```

새 기능은 `packages/`에서 개발하고, 완료된 실험의 설정과 결과는 `experiments/<milestone>/`에 분리한다. Phase 0 기록을 새 실험 결과로 덮어쓰지 않는다.

## 빠른 시작

```bash
cd packages/amr_lite
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
pytest -q
```

전체 smoke pipeline:

```bash
cd packages/amr_lite
PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib python -m amr_lite.cli.train
```

세부 학습·평가 명령은 [AMR-Lite README](packages/amr_lite/README.md)를 참고한다.

## 개발 흐름

- `dev`: 기본 통합 개발 브랜치
- `feature/<topic>`: 기술 영역별 기능 개발
- `fix/<topic>`: 결함 수정
- `docs/<topic>`: 문서 전용 변경

기능 브랜치는 `dev`에서 분기하고 검증 후 다시 `dev`로 병합한다. 현재는 별도의 `main` 브랜치를 운영하지 않는다.

## 문서

- [향후 마일스톤](docs/ROADMAP.md)
- [다음 기술 항목](docs/NEXT_STEPS.md)
- [저장소 구조 원칙](docs/REPOSITORY_STRUCTURE.md)
- [Phase 0 결과](experiments/phase0_amr_lite/RESULTS.md)
- [Phase 0 설계 결정](experiments/phase0_amr_lite/DECISIONS.md)
- [Phase 0 알려진 한계](experiments/phase0_amr_lite/KNOWN_LIMITATIONS.md)

## 현재 상태

Phase 0 스파이크는 완료됐다. Stable PPO best는 51,200 timestep에서 선택됐고 제한된 평가에서 Shield OFF 성공률 87.5%를 기록했다. `D1` 정면 접근 충돌, `S4` Shield saturation, 여러 seed에서의 재현성은 Phase 1의 주요 과제다.
