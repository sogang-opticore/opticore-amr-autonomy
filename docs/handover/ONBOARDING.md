# 팀 온보딩 — 처음 이틀 동안 할 일

spike를 이어받는 팀원이 환경을 갖추고, 자기 영역의 코드를 직접 확인하고, 의심 항목을
기록하기까지의 절차다. 방향과 2주 일정은 [DIRECTION_OPTIONS](DIRECTION_OPTIONS.md),
명령의 세부와 합격 기준은 [RUNBOOK](../RUNBOOK.md)을 따른다. §1, §3–§5와 표시된 문장은 (claude 제안사항)이고,
§2 환경 준비와 §6은 확인된 실행 절차·증상이다.

## 1. 읽는 순서 (claude 제안사항)

| 순서 | 문서 | 시간 | 얻는 것 |
|---|---|---|---|
| 1 | [README](../../README.md) | 5분 | 목표와 저장소 구조 |
| 2 | [SPIKE_REVIEW](SPIKE_REVIEW.md) | 20분 | 지금까지 한 일과 배운 것 |
| 3 | [VERIFICATION_STATUS](VERIFICATION_STATUS.md) | 15분 | 믿어도 되는 것과 아닌 것 |
| 4 | [ROADMAP](../ROADMAP.md) §1–§4, §6 | 20분 | 마일스톤과 gate |
| 5 | [ARCHITECTURE](../ARCHITECTURE.md) | 30분 | 관측·행동·bundle·토픽 계약 |
| 6 | [internals](../internals/README.md) 중 자기 영역 | 1–2시간 | 코드가 실제로 하는 일 |
| 7 | [EXPERIMENT_PLAYBOOK](../EXPERIMENT_PLAYBOOK.md) | 20분 | 학습 실험 절차 (R1·R2 필수) |
| 8 | [amr_learned_planner README](../../integrations/ros2/amr_learned_planner/README.md) | 20분 | ROS 실행 단계 (R4·R5 필수) |

## 2. 환경 준비

```bash
git clone git@github.com:sogang-opticore/opticore-amr-autonomy.git
cd opticore-amr-autonomy/packages/amr_lite
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
export PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib
pytest -q                                   # 119 passed, 약 2분
cd ../.. && python -m pytest -q integrations/ros2/amr_learned_planner/test   # 26 passed
```

- Python 3.10 이상. 확인 환경은 Python 3.11, PyTorch 2.8, NumPy 2.3이다.
- iCloud·Dropbox 같은 동기화 폴더에 clone하지 않는다. git이 느려지고 `.git` 객체가
  동기화 대상이 된다 (claude 제안사항).
- macOS에는 `timeout` 명령이 없다. 긴 실행은 `nohup`이나 별도 터미널을 쓴다.
- ROS2 쪽(R4·R5)은 `opticore-amr`의 Docker 환경(ROS 2 Humble, Ignition Fortress)을 쓴다.
  설치는 [amr_learned_planner README](../../integrations/ros2/amr_learned_planner/README.md) §1을 따른다.

## 3. 첫 smoke (claude 제안사항)

모든 팀원이 한 번씩 돌려 본다. 결과는 성능이 아니라 실행 경로 확인이다.

```bash
cd packages/amr_lite
export PYTHONPATH=. MPLCONFIGDIR=artifacts/.matplotlib
python -m amr_lite.cli.train_rl --smoke --output artifacts/onboarding/ppo-v2
python -m amr_lite.cli.sweep e3_lagrangian --smoke --output artifacts/onboarding/sweep
python -m amr_lite.cli.export_policy --checkpoint artifacts/onboarding/ppo-v2/final.pt \
  --output artifacts/onboarding/bundle
python -m amr_lite.cli.verify_bundle artifacts/onboarding/bundle --robot-profile opticore_gazebo
```

`artifacts/`는 Git에 들어가지 않는다.

## 4. 영역 인수 절차 (claude 제안사항)

자기 영역([DIRECTION_OPTIONS](DIRECTION_OPTIONS.md) §5)을 아래 순서로 인수한다.

1. 해당 internals 문서를 읽고 파일 지도의 모든 파일을 한 번씩 연다.
2. 해당 영역의 테스트를 따로 실행하고, 테스트가 무엇을 확인하고 무엇을 확인하지
   않는지 internals 문서의 "불변식과 검증" 절과 대조한다.
3. 핵심 함수 하나를 골라 작은 입력으로 직접 호출해 본다(`/tmp`의 스크립트로).
4. internals 문서의 "주의할 점과 알려진 한계" 항목을 하나씩 확인하고 결과를 GitHub
   issue로 남긴다. 제목은 `[영역] 요약`, 본문에 재현 명령과 판단을 쓴다.
5. 문서가 코드와 다르면 문서를 고치는 PR을 낸다.

인수가 끝났다는 기준: 자기 영역을 다른 팀원에게 15분 안에 설명할 수 있고, 의심 항목이
issue로 정리돼 있다.

## 5. 작업 규칙 (claude 제안사항)

| 항목 | 규칙 |
|---|---|
| 브랜치 | `dev`에서 `feature/*`, `fix/*`, `docs/*`, `exp/*`로 분기하고 PR로 `dev`에 병합한다 |
| PR | 두 테스트 스위트 통과. 학습·평가 코드를 바꾸면 Gate 7 smoke도 통과 |
| 리뷰 | 다른 역할 1명 이상이 승인한다. protocol·gate 변경은 R1이 반드시 리뷰한다 |
| 실험 | sweep spec을 commit한 뒤 실행한다. 결과는 `experiments/<milestone>/`에 보존하고 기존 폴더를 덮어쓰지 않는다 |
| 결정 | gate 판정과 기준 변경은 `experiments/<milestone>/DECISIONS.md`에 날짜·근거와 함께 남긴다 |
| test split | 후보 확정 후 sweep당 1회. 재실행은 protocol version을 올린다 |
| 수치 인용 | [VERIFICATION_STATUS](VERIFICATION_STATUS.md)의 E4 이상 수준만 결론에 쓴다 |
| 커밋 메시지 | 기존 형식(`feat(amr-lite): ...`, `docs(roadmap): ...`)을 따른다 |

## 6. 자주 막히는 곳

| 증상 | 원인과 대응 |
|---|---|
| `ModuleNotFoundError: amr_lite` | `packages/amr_lite`에서 `PYTHONPATH=.`를 설정하지 않았다 |
| vector env worker가 `spawn` 오류로 종료 | `python - <<EOF`처럼 표준입력 스크립트로 실행했다. `python -m ...`이나 `__main__` 가드가 있는 파일로 실행하거나 `vector_env.backend=sync`를 쓴다([RUNBOOK](../RUNBOOK.md) §13) |
| sweep `--stage test`가 거부됨 | 이미 test를 연 sweep이다. 의도한 재실행이면 protocol version을 올린다 |
| ROS 노드가 주행하지 않음 | `PROFILE_MISMATCH`. bundle profile과 노드 파라미터를 맞추거나 shadow에서만 `profile_check:=warn` |
| smoke 결과가 매번 FAIL | 정상이다. smoke budget은 성능을 내지 못한다 |
