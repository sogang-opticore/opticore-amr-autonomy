# Handover — spike에서 팀 개발로

2026-09-06부터 2026-10-13까지 AI에 상당 부분을 위임해 진행한 탐색 spike를 5인 팀이
이어받기 위한 문서 묶음이다. 기준 commit은 `4e83d29`(`dev`), 작성일은 2026-10-14다.

**표기**: `(claude 제안사항)`가 붙은 제목·문장은 이번 검토에서 Claude가 제안한 판단·계획·규칙이다. 표시가 없는
내용은 코드, 측정, 기존 기록에서 확인한 사실이다. 제안은 팀 합의 전까지 확정된 결정이 아니다.

| 문서 | 질문 |
|---|---|
| [SPIKE_REVIEW](SPIKE_REVIEW.md) | 무엇을 만들었고, 무엇을 배웠고, 무엇을 의심해야 하나 |
| [VERIFICATION_STATUS](VERIFICATION_STATUS.md) | 구성요소별로 어떤 근거가 실제로 있나. 테스트는 무엇을 지키나 |
| [DIRECTION_OPTIONS](DIRECTION_OPTIONS.md) (claude 제안사항) | 따라가기·다시 만들기·새 방향 중 무엇을 하나. 2주 동안 누가 무엇을 하나 |
| [ONBOARDING](ONBOARDING.md) (claude 제안사항) | 처음 이틀 동안 무엇을 읽고, 무엇을 실행하고, 어떻게 인수하나 |

코드가 실제로 어떻게 동작하는지는 [internals](../internals/README.md)에 서브시스템별로
정리돼 있다. 계약은 [ARCHITECTURE](../ARCHITECTURE.md), 마일스톤과 gate는
[ROADMAP](../ROADMAP.md)가 기준이다.

## 핵심 세 줄

1. 코드는 M6까지 있지만 근거는 M1까지만 있다. 학습 sweep과 ROS2/Gazebo 실행은 한 번도 하지 않았다.
2. 유일한 공식 결과(M1)에서 학습 정책은 rule baseline보다 낫다는 증거가 없었다.
3. G4 gate와 선정은 shield 없는 rule baseline과 비교하도록 정의돼 있고, 학습·평가·Gazebo 기록을 오염시키는
   결함이 16개 있다([DIRECTION_OPTIONS](DIRECTION_OPTIONS.md) §4). 학습과 실기동 전에 고친다 (claude 제안사항).
