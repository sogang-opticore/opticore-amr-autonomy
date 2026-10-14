# OptiCore AMR Autonomy

OptiCore AMR의 경로 계획, 학습 기반 주행, 안전 제어와 향후 ROS2/Gazebo 연결을 위한 자율주행 저장소다.

목표는 `opticore-amr`의 규칙 기반 local planner(`dwa_node`)를, 경량 2D 환경에서 여러
번의 실험으로 학습한 PPO 중심 정책으로 대체하는 것이다. 학습 정책은 결정적 안전
계층 뒤에서 명령을 제안하고, ROS2에서는 기존 DWA를 fallback으로 둔 채 단계적으로
주행 권한을 넘겨받는다. 마일스톤과 gate는 [ROADMAP](docs/ROADMAP.md)를 따른다.

## 저장소 구조

```text
packages/
  amr_lite/                 Lite 환경, 학습(PPO v2·DAgger), sweep·gate, bundle export/runtime
integrations/
  ros2/amr_learned_planner/ opticore-amr용 ROS2 노드: learned planner, cmd arbiter, recorder
experiments/
  phase0_amr_lite/          Phase 0 결정·결과·한계와 선별 산출물
  m1_reproducibility/       M1 재현성 결과·manifest·요약·그래프
docs/
  ROADMAP.md                DWA → 학습형 local planner 마일스톤과 gate
  ARCHITECTURE.md           목표 스택과 관측·행동·bundle·토픽 계약
  EXPERIMENT_PLAYBOOK.md    학습 실험(E1–E6) 절차와 예산
  RUNBOOK.md                단계별 실행·검증·복구 절차
  NEXT_STEPS.md             후속 기술 항목과 팀 결정 질문
  REPOSITORY_STRUCTURE.md   디렉터리와 의존성 원칙
  internals/                서브시스템별 코드 동작 설명(환경, 센서·안전, 학습, 평가·실험, 배포·ROS2)
  handover/                 spike 회고, 검증 현황, 향후 방향과 2주 계획, 팀 온보딩
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

Smoke 확인:

```bash
cd packages/amr_lite
export PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib
python -m amr_lite.cli.train                      # Phase 0 smoke (teacher → BC → PPO → 평가)
python -m amr_lite.cli.train_rl --smoke --output artifacts/rl/smoke          # PPO v2
python -m amr_lite.cli.sweep e3_lagrangian --smoke --output artifacts/sweeps/smoke
cd ../.. && python -m pytest -q integrations/ros2/amr_learned_planner/test  # ROS 통합 (ROS 없이)
```

세부 명령은 [AMR-Lite README](packages/amr_lite/README.md), 학습 실험은
[실험 playbook](docs/EXPERIMENT_PLAYBOOK.md), opticore-amr 연결은
[amr_learned_planner README](integrations/ros2/amr_learned_planner/README.md)를 따른다.

## 개발 흐름

- `dev`: 기본 통합 개발 브랜치
- `feature/<topic>`: 기술 영역별 기능 개발
- `fix/<topic>`: 결함 수정
- `docs/<topic>`: 문서 전용 변경

기능 브랜치는 `dev`에서 분기하고 검증 후 다시 `dev`로 병합한다. 현재는 별도의 `main` 브랜치를 운영하지 않는다.

## 문서

처음 합류했다면 [handover](docs/handover/README.md)와 [팀 온보딩](docs/handover/ONBOARDING.md)부터 읽는다 (claude 제안사항).
`(claude 제안사항)` 표시는 이번 검토에서 Claude가 제안한 판단·계획이며 팀 합의 전에는 확정이 아니다.

- [Spike 회고](docs/handover/SPIKE_REVIEW.md)
- [검증 현황과 테스트 지도](docs/handover/VERIFICATION_STATUS.md)
- [향후 방향과 2주 계획](docs/handover/DIRECTION_OPTIONS.md) (claude 제안사항)
- [코드 내부 구조](docs/internals/README.md)
- [향후 마일스톤](docs/ROADMAP.md)
- [학습형 planner 아키텍처와 계약](docs/ARCHITECTURE.md)
- [학습 실험 playbook](docs/EXPERIMENT_PLAYBOOK.md)
- [실행·검증 런북](docs/RUNBOOK.md)
- [opticore-amr 통합 패키지](integrations/ros2/amr_learned_planner/README.md)
- [다음 기술 항목](docs/NEXT_STEPS.md)
- [저장소 구조 원칙](docs/REPOSITORY_STRUCTURE.md)
- [Phase 0 결과](experiments/phase0_amr_lite/RESULTS.md)
- [Phase 0 설계 결정](experiments/phase0_amr_lite/DECISIONS.md)
- [Phase 0 알려진 한계](experiments/phase0_amr_lite/KNOWN_LIMITATIONS.md)
- [M1 재현성 결과](experiments/m1_reproducibility/RESULTS.md)

## 현재 상태

- **M1 완료.** 5개 학습 seed와 4,160개 test episode에서 custom Stable PPO best는
  Shield OFF 성공률 85.4%(95% CI 84.0–86.5%)로 Rule baseline 86.2%보다 우월하다는
  증거가 없었다.
- **M2–M6 코드 구현 완료, 학습·실기동 대기.** Gazebo 로봇과 같은 LiDAR·동역학 profile,
  센서 randomization, LiDAR 기반 Safety Shield, 절차적·창고 지도 시나리오, PPO v2와
  DAgger, sweep과 G4 gate, torch 없는 policy bundle, opticore-amr ROS2 통합 패키지가
  들어갔다. 기본 설정(Phase 0/M1)의 결과는 변경 전과 bit 단위로 같다.
- **2026-10-14 인수인계 검토.** 테스트(119 + 26)와 학습·배포 smoke가 다시 통과했다.
  G4 gate가 Shield OFF rule baseline과 비교하도록 정의돼 있어 학습 전에 배포 구성
  (Shield ON) 기준으로 고칠 것을 제안했다 (claude 제안사항). 상세는 [검증 현황](docs/handover/VERIFICATION_STATUS.md) §4.
- 다음 작업은 [2주 계획](docs/handover/DIRECTION_OPTIONS.md)에 따른 결정·G3 동결, E1–E6 학습,
  M5 ROS2 실기동, M6 Gazebo paired 검증이다 (claude 제안사항).
