# M2 · 창고 주행 문제 정의와 평가 기반

같은 단일 로봇·창고·시간 제한에서 기존 hybrid 주행기가 어떤 임무를 수행하고 어디서 정체하거나 접촉하는지 측정한다. 기준선은 `opticore-amr`의 실제 위치추정·A*·local control·Gazebo 조합이다. Lite는 준비 비용과 환경 차이를 진단하는 데 사용한다.

## 범위와 인수 조건

| 확인할 것 | 다음 단계로 넘길 증거 |
|---|---|
| 창고와 로봇 | 지도·세계·센서 좌표, 차체와 footprint, 동역학과 지연의 설정/실측 구분 |
| 실제 기준선 | 같은 목표·장애물 일정으로 반복 실행한 B0/B1, 실패 재현 및 원인별 원시 기록 |
| 계측 신뢰성 | 목표 교체 시 이전 도착 신호 배제, 좌표 변환, 시간 제한, 실제 접촉과 접촉 데이터 누락을 구분하는 검증 |
| 비교와 학습 경계 | 개발·선정·독립 시험 분할, 고정할 관측·행동·학습 출발 설정, 운용 지표와 측정 예산 |
| 진행 판정 | 실행된 기준선 결과, 대표 문제, 계측 검증 결과가 갖춰지면 M3로 진입 |

우선 단일 로봇, 정적 운송·횡단·정면 접근·일시 차단을 다룬다. 동적 상황에서 재현되지 않은 문제는 가설로 기록한다. 이후 단계에서 정책 학습, 반복 임무 복구, 두 대의 fleet 호환 및 독립 시험으로 범위를 넓힌다. 실물 로봇과 적재·하역은 이번 검증 범위 밖이다.

## M1 보존과 재사용

- M1 이관 기준 `c44806c3fe2d6e12974b327012ed4570901fabe4`, 현재 개발 기반 `d81ccb6e579469cbdf9c22afa18943544fb254b3`. 두 이력에서 `experiments/m1_reproducibility`의 Git tree는 `68fb03585146f8ef5555fc39a527a2cc3849f5f0`이다.
- 보존한 Lite 평가에서 Shield OFF 성공률은 Rule 86.2%, PPO 5-seed best 85.4%다. D1 정면 접근은 PPO가 모두 충돌했다. 이 관측은 창고 재현 실험의 동기다. [M1 결과](../../experiments/m1_reproducibility/RESULTS.md)
- 원래 `opticore-amr/amr_lite_spike` 보관본을 manifest와 바이트 단위로 대조했다. 체크포인트 12개가 일치했다. 확인한 teacher NPZ 6개는 이름이 일치하고 SHA-256이 모두 다르다. 같은 teacher 데이터로 재학습을 재현하려면 manifest에 맞는 사본을 확보해야 한다. [파일별 해시 대조표](../../experiments/m2/m1-asset-audit.json)
- 재검증 명령은 `python3 scripts/m2/audit_m1_assets.py /path/to/opticore-amr/amr_lite_spike`다. 파일 누락이나 해시 불일치가 있으면 개별 결과를 출력하고 종료 코드 1을 반환한다. M1 모델·데이터·요약 CSV는 보존하며 M2의 기준선 시행 수에는 포함하지 않는다.

## 구현과 실행 증거

**현재 판정은 보류다.** 2026-11-22에 개발용 정적 운송 임무를 B0와 B1에서 기동했다. 이 호스트에 ROS 2 Humble의 `ament_index_python`과 Fortress 실행 파일이 없어 두 시도 모두 목표 전송 전에 끝났다. 확보한 Gazebo 주행은 0회다. 직접 시도한 Docker 엔진 접근과 기존 원격 실행 호스트 연결도 이 작업 환경에서 허용되지 않았다.

| 개발 조건 | B0 직접 제어 | B1 중계 제어 |
|---|---|---|
| 정적 station-to-station | 환경 기동 실패, 주행 0회 | 환경 기동 실패, 주행 0회 |
| 횡단·정면 접근·일시 차단 | 실행 대기 | 실행 대기 |
| 무충돌 완료율·충돌률·운송시간·실측 RTF | 미측정 | 미측정 |

개발 입력은 4종 × 3분할로 등록했다. 독립 시험 결과는 열지 않았다. M2 전용 Python 회귀 테스트 44개와 네이티브 C++ 접촉 분류 테스트를 로컬에서 통과했다. Python 테스트에는 실제 기록기 코어와 가짜 ROS 메시지를 사용한 경우가 포함된다. Gazebo 플러그인 빌드, 실접촉 보정, 단일 로봇 폐루프 주행, 대표 실패 재현, B0/B1 비교 및 실측 예산 확정이 M2의 잔여 항목이다.

- [창고·차체 기준](warehouse.md) · [실험 장면](scenarios.md) · [임무 입력과 산출물](missions.md)
- [시스템 경계](baselines.md) · [ROS·fleet 계약](ros-fleet.md) · [빌드와 실제 주행기 기동](baseline-launch.md)
- [에피소드·충돌 계측](episode-schema.md) · [관측·행동](learning-io.md) · [PPO·보상](learning-ppo.md)
- [평가 절차와 예산](evaluation.md) · [B0/B1 시도별 결과](../../experiments/m2/baseline-inventory.json) · [Lite 처리량 측정](../../experiments/m2/lite-throughput.json)

설치와 접촉 보정 후 첫 개발 시행 및 집계 예시는 아래와 같다. 명령은 저장소 루트에서 실행한다.

```sh
python3 experiments/m2/run_mission.py train-static --planner b0 --seed 0 --out /tmp/m2-b0-static-01
python3 experiments/m2/run_mission.py train-static --planner b1 --seed 0 --out /tmp/m2-b1-static-01
python3 scripts/m2/summarize_baseline.py --split train --seed 0 /tmp/m2-b0-static-01 /tmp/m2-b1-static-01
```
