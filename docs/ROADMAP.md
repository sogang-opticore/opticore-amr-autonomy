# OptiCore AMR Autonomy Roadmap

## 목적

Phase 0에서 확인한 AMR-Lite의 실행 가능성을 반복 가능한 실험, 동적 장애물 안전성, 안정적인 학습 정책, ROS2/Gazebo 연결 순으로 확장한다. 이 문서는 인원별 업무가 아니라 기술 영역과 마일스톤의 완료 조건을 정의한다.

일정은 팀 가용 인원에 따라 달라질 수 있으므로 날짜 대신 sprint 단위의 상대 규모를 사용한다. 아래 수치 기준은 최초 제안값이며 M1 시작 시 팀 합의로 확정한다.

## 현재 기준선 — M0 완료

Phase 0 기술 스파이크는 완료 상태다.

- ROS 없이 실행되는 2D differential-drive 환경
- 정적·동적 시나리오와 4-frame LiDAR 관측
- A*와 Rule baseline
- BC, DAgger-lite, PPO, Stable PPO
- 공통 Safety Shield와 ON/OFF paired evaluation
- 20개 자동 테스트
- Stable PPO best: 51,200 timestep, Shield OFF 성공률 87.5%
- 주요 잔여 위험: `D1` 정면 접근 충돌, `S4` Shield saturation, final checkpoint 성능 저하

현재 수치는 8개 시나리오와 각 2개 평가 seed에서 나온 결과이므로 일반화 성능으로 해석하지 않는다.

## M1 — 재현 가능한 평가 기준선

예상 규모: 1 sprint

목표:

- 모델 개선 전에 평가 방식과 checkpoint 선택 규칙을 고정한다.
- 단일 seed 결과를 여러 학습·평가 seed 분포로 확장한다.

주요 작업:

- Stable PPO를 최소 5개 학습 seed로 반복
- 모델·Shield 조건별로 시나리오당 최소 5개 평가 seed 실행
- 평균, 중앙값, 표준편차, 95% confidence interval 산출
- best checkpoint 선택용 시나리오와 최종 test 시나리오 분리
- 실험 config, Git commit, dataset version, seed를 manifest에 기록
- 현재 최소 PPO와 검증된 PPO 구현의 smoke 수준 수치 비교
- 전체 평가와 결과표 생성을 하나의 명령으로 자동화

산출물:

- 고정 evaluation protocol
- seed별 원자료와 통계 보고서
- checkpoint selection 규칙
- 자동 실행 명령과 실험 manifest

완료 기준:

- 동일 commit과 config에서 평가 결과를 재생성할 수 있다.
- best와 final checkpoint가 분리 보고된다.
- 주요 성능 지표에 95% confidence interval이 포함된다.
- 모든 자동 테스트와 smoke pipeline이 통과한다.

## M2 — 동적 장애물 인식과 안전 제어

예상 규모: 1–2 sprints

선행조건: M1 평가 protocol 고정

목표:

- `D1` 정면 접근 충돌과 `S4` Shield saturation을 재현하고 직접 해결한다.
- 단순 거리 임계값을 넘어 장애물의 상대 운동을 안전 판단에 사용한다.

주요 작업:

- frame stack과 explicit velocity tracker 비교
- CPA/TTC feature 및 예측 오차 계측
- Shield horizon, margin, action projection 조정
- 제한적 후진과 bounded evasive trajectory 구현
- PASS/CLAMP/STOP의 false-positive·false-negative 분석
- D0–D4 속도, 시작 위치, 정지 시점 randomization
- 정지 로봇으로 접근하는 장애물의 충돌 책임 정의

산출물:

- 동적 장애물 tracker 또는 선택된 temporal representation
- Safety Shield v3와 recovery policy
- D0–D4 stress-test 결과
- Shield 개입 원인과 실패 사례 보고서

완료 기준 제안:

- `D1` 충돌률 10% 이하: 최소 20 episode 기준
- `S4` Shield saturation/timeout 비율 5% 이하
- Shield ON이 정적 시나리오 성공률을 OFF 대비 5%p 넘게 낮추지 않는다.
- 모든 intervention에 원인과 지속 시간이 기록된다.

## M3 — 학습 정책 안정화와 일반화

예상 규모: 2 sprints

선행조건: M1 완료, M2의 관측·안전 인터페이스 확정

목표:

- 특정 seed나 중간 checkpoint에 의존하지 않는 정책을 선택한다.
- 정적 환경 중심 학습에서 동적 장애물과 perturbation을 포함한 학습으로 확장한다.

주요 작업:

- teacher 데이터의 scenario-family train/validation 분리
- DAgger 위험 상태 sampling 비중 조정
- Stable PPO의 learning rate, target KL, BC anchor sweep
- 동적 장애물 curriculum과 domain randomization
- BC, DAgger, Stable PPO 및 필요시 SAC 비교
- reward hacking 자동 탐지와 회전·정지 행동 분석
- 여러 학습 seed의 median과 worst-case 성능 비교

산출물:

- versioned teacher/DAgger dataset 명세
- hyperparameter 비교표
- seed별 학습 곡선과 failure breakdown
- 배포 후보 policy와 선정 근거

완료 기준 제안:

- 전체 평가 성공률 중앙값 85% 이상
- 전체 충돌률 중앙값 10% 이하
- final checkpoint가 해당 seed의 best 성공률보다 10%p 넘게 저하되지 않는다.
- 단방향 선회, 무한 정지 등 알려진 reward hacking이 자동 평가에서 검출되지 않는다.
- 채택 정책이 Rule baseline 대비 장점 또는 유지할 명확한 근거를 가진다.

## M4 — ROS2/Gazebo 통합 검증

예상 규모: 1–2 sprints

선행조건: M2 인터페이스 확정, M3 배포 후보 선택

목표:

- Lite에서 선택한 정책을 ROS2/Gazebo에서 같은 관측·행동 계약으로 실행한다.
- Lite 성능과 Gazebo 성능의 차이를 측정한다.

주요 작업:

- LaserScan, odometry, TF 기반 observation adapter
- normalized action에서 `cmd_vel`로의 scaling과 acceleration limit 일치
- footprint, sensor frame, 제어주기 정합성 검증
- Lite와 Gazebo의 동일 경로·장애물 paired scenario 구성
- inference latency와 20 Hz deadline 측정
- policy 오류 시 정지 및 기존 planner fallback 정의

산출물:

- ROS2 inference node와 launch/config
- Lite ↔ Gazebo 대응 시나리오
- 기능·성능 parity 보고서
- fallback과 운영 안전 절차

완료 기준:

- 동일 observation/action 계약에 대한 자동 검증이 통과한다.
- 20 Hz 제어주기에서 deadline miss가 측정·보고된다.
- 모든 paired scenario의 성공·충돌·clearance 차이가 기록된다.
- 모델 로드 실패, NaN action, deadline miss 시 안전 정지가 작동한다.

## M5 — Autonomy release candidate

예상 규모: 1 sprint

선행조건: M1–M4 gate 통과

목표:

- 반복 가능한 배포 후보와 운영 문서를 제공한다.

주요 작업:

- config와 checkpoint version 고정
- CI에 단위 테스트, smoke, artifact schema 검사 추가
- 모델 checksum과 release note 생성
- 설치, 실행, 평가, rollback 문서 정리
- 알려진 한계와 허용 운용 범위 명시
- 대표 정적·동적 시나리오 데모 생성

완료 기준:

- 깨끗한 환경에서 문서만으로 설치와 smoke 실행이 가능하다.
- release checkpoint와 config를 checksum으로 식별할 수 있다.
- 평가 보고서와 알려진 한계가 release에 연결된다.
- ROS2/Gazebo에서 성공·실패·fallback 데모가 재현된다.

## 기술 영역

마일스톤은 다음 7개 기술 영역을 조합해 진행한다.

1. 시뮬레이션 환경·시나리오
2. 관측·동적 장애물 인식
3. 경로계획·Rule baseline·Teacher
4. BC·DAgger·PPO 학습 정책
5. Reward·Safety Shield·Recovery
6. 평가·실험 자동화·통계
7. ROS2·Gazebo·시스템 연결

팀원 배치는 이 문서에서 고정하지 않는다. 각 영역은 담당 인원과 무관하게 입력, 출력, 설정, 테스트, 결과를 남겨야 한다.

## 의사결정 Gate

### G1 — 평가 기준 승인

M1 종료 시 평가 seed 수, 통계 방식, checkpoint 선택 규칙을 승인한다. 이후 정책 결과에 맞춰 기준을 변경하지 않는다.

### G2 — 동적 장애물 전략 승인

M2 종료 시 frame stack만 유지할지 tracker/CPA/TTC를 추가할지, Shield가 후진과 회피 조향을 허용할지 결정한다.

### G3 — 배포 후보 정책 선정

M3 종료 시 Rule, BC/DAgger, Stable PPO 중 ROS2/Gazebo로 가져갈 주 정책과 fallback을 선정한다. PPO 채택은 필수가 아니다.

### G4 — Release 승인

M4 종료 시 성능, 실시간성, 안전 정지 결과를 검토해 release candidate 진입 여부를 결정한다.

## 공통 보고 원칙

- 성공률만 보지 않고 충돌률, timeout, minimum clearance를 함께 보고한다.
- 학습 seed와 평가 seed를 구분한다.
- Shield OFF와 ON 결과를 분리한다.
- best와 final checkpoint를 모두 보고한다.
- 실패한 실험도 설정과 원인을 기록한다.
- 목표 수치가 바뀌면 결과가 아니라 결정 문서에 변경 이유를 남긴다.
