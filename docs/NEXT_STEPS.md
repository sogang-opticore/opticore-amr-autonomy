# Phase 1 Discussion and Next Steps

## 1. 시뮬레이터와 관측

- NumPy/vectorized ray casting 또는 spatial index로 병목 측정 후 최적화
- Gymnasium 정식 adapter와 vector environment
- 기존 `warehouse_map.pgm` 선택적 importer
- LiDAR noise, latency, dropout, domain randomization
- 1-frame 대 4-frame paired ablation

## 2. 동적 장애물

- Safety Shield에 추가한 obstacle velocity extrapolation의 horizon·margin을 시나리오별 계측·조정
- frame stack과 explicit tracker/CPA feature 비교
- 횡단, 정면 접근, 추종, 정지 패턴별 seed와 속도 범위 확대
- `D1` 정면 접근 실패를 대상으로 제한적 후진·시간 확장 회피 trajectory 비교
- 정지 로봇으로 계속 접근하는 장애물의 책임/충돌 정의 명확화

## 3. 학습

- teacher episode와 perturbation 확대, scenario-family validation split
- DAgger 정상/위험 상태 sampling weight 조정 및 반복 횟수 비교
- Stable PPO의 checkpoint 의존성 검증: 여러 학습 seed, KL·anchor·learning-rate sweep, 평가 seed 확대
- 검증된 PPO 라이브러리와 최소 구현의 수치·성능 비교
- Stable best와 final checkpoint를 항상 분리 보고하고 선택 규칙을 사전 고정
- PPO 유지 여부 및 SAC 비교 결정

## 4. 안전과 평가

- 단순 CLAMP/STOP을 넘어 안전 action projection 검토
- shield 예측의 false-positive/false-negative 계측
- shield saturation recovery와 제한적 후진 정책
- 모델·시나리오당 seed 수 확대와 confidence interval
- failure taxonomy 자동 판정 정확도 검토

## 5. ROS2/Gazebo 연결

- observation builder와 action scaling을 재사용하는 ROS adapter
- Lite baseline과 기존 hybrid planner의 기능/파라미터 대응표
- 동일 warehouse 경로와 장애물 배치의 Lite ↔ Gazebo paired test
- timing, sensor frame, footprint 차이 계측 후 Sim-to-Real 범위 결정

## 6. 기술 영역

1. 시뮬레이션 환경·시나리오
2. 관측·동적 장애물 인식
3. 경로계획·Rule baseline·Teacher
4. BC·DAgger·PPO 학습 정책
5. Reward·Safety Shield·Recovery
6. 평가·실험 자동화·통계
7. ROS2·Gazebo·시스템 연결

인원 배치는 고정하지 않는다. 각 영역의 입력·출력 계약과 완료 기준을 먼저 고정하고, 팀 상황에 맞춰 한 사람이 여러 영역을 맡거나 여러 사람이 같은 영역을 나눠 맡는다. 실행 순서와 gate는 [ROADMAP.md](ROADMAP.md)를 따른다.

## 팀 결정 질문

1. 이 Lite 환경을 정식 공통 기반으로 사용할 것인가?
2. 기존 ROS planner 중 어느 상태 머신까지 baseline에 반영할 것인가?
3. 동적 장애물 대응을 최종 완료 조건에 포함할 것인가?
4. frame stack과 tracker feature 중 무엇을 주 입력으로 둘 것인가?
5. Stable PPO seed sweep과 BC/DAgger 데이터 품질 개선 중 무엇을 먼저 할 것인가?
6. Safety Shield가 정지 외에 제한적 회피/후진을 허용해야 하는가?
7. Gazebo paired validation을 어느 gate부터 필수로 둘 것인가?
