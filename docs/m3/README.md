# M3 · 학습 경로와 단일 로봇 폐루프

**Post-M1 계획의 M3 완료.** seed 7의 SB3 PPO를 Lite에서 학습·재개하고, 같은 actor를 NumPy 번들로 Humble에 올렸다. Gazebo의 단일 AMR은 4m 정적 진단 경로에서 학습 명령으로 이동했고, 의도적으로 명령을 차단하자 정지한 뒤 새 관측을 받아 이동을 재개했다. [학습·재개 기록](../../experiments/m3/training-provenance.json)

## Lite 학습

M2 기준의 SB3 2.4.1, Gymnasium 1.0.0, seed 7, 네 환경, 20Hz 제어를 사용했다. 각 rollout은 1,024전이다. 동적 장애물은 개발 임무 manifest의 궤적을 따르며, Lite는 지도와 timed box의 평면 충돌을 계산한다. Lite의 정책 입력과 ROS 입력은 네 개의 서로 다른 LiDAR 획득 시각으로 같은 296개 float32 특성을 만든다. 지상진실과 접촉 데이터는 정책 입력에서 제외한다. [관측·명령 계약](policy-io.md) · [센서·동역학 실측](fortress-parity.md)

| 항목 | 실측 |
|---|---|
| 기본 학습 | 65,536전이, CPU 1스레드, 52.46초, 1,249전이/s |
| 수집 중 완료 에피소드 | 시간 초과 24, 충돌 1, 손상 전이 0 |
| 저장·재개 | 66,560전이까지 진행, PPO 갱신 640→650회, Adam 단계 10,240→10,400 |
| SB3↔NumPy | 골든 관측 64개, 결정론적 행동 최대 절대차 1.42×10⁻⁷ |

학습은 정적 개발 임무에서 수행했다. 고정 seed 100–103의 개발 진단은 학습 전후의 차이를 보여 준다. 목표 최근접 거리는 경로 추종을, 종료 판정은 무충돌 임무 완료를 평가한다. [에피소드별 보상과 학습 곡선 데이터](../../experiments/m3/training-diagnostic.json)

| 개발 조건 | 학습 전 | 학습 후 | 최근접 목표 오차, 전 → 후 |
|---|---|---|---:|
| 정적 운송 | 시간 초과 | 시간 초과 | 4.493 → 0.871 m |
| 횡단 | 시간 초과 | 시간 초과 | 4.478 → 0.871 m |
| 정면 접근 | 충돌 | 무접촉 시간 초과 | 39.347 → 0.873 m |
| 일시 차단 | 시간 초과 | 시간 초과 | 14.500 → 14.351 m |

정적·횡단·정면 접근의 학습 후 진단에서는 경로 투영점이 끝점에 도달해 진척 보상 100을 얻었지만, 횡방향 오차 때문에 목표 0.4m 조건을 만족하지 못했다. 차단 장면은 시작점에서 목표 방향으로 거의 진행하지 못했다. 동일 actor에 Gazebo 실측으로 정한 노이즈·지연을 적용한 별도 Lite 민감도 확인에서는 정적 운송 1건이 종료 직전 충돌로 바뀌었고, 정면 접근의 최종 목표 오차는 4.81m에서 35.52m로 커졌다. 해당 비교는 조건당 고정 seed 한 쌍이다. [민감도 시행별 결과](../../experiments/m3/fortress-variation.json)

## Humble·Fortress 통합

고정된 원본 창고, Dock 1 시작점 `map(2, 22)`, 진단 목표 `map(6, 22)`를 사용했다. 65,536전이 번들 하나를 두 구성에 적용했다. Shadow는 기존 DWA가 최종 명령을 내렸고 학습 후보도 별도로 기록했다. Learned에서는 중계기가 학습 후보를 선택하며, 상태 이상이나 진단용 hold에서 물리적 0을 발행한다. 속도 제한은 종방향 0.45 m/s, 각속도 0.6 rad/s다.

| 직렬 진단 | 결과 | 시뮬레이션 시간 | 목표 오차 | 최종 이동 명령의 출처 |
|---|---|---:|---:|---|
| Shadow | 무충돌 도착 | 7.276초 | 0.365m | 119/119 DWA |
| Learned | 무충돌 도착 | 12.608초 | 0.389m | 217/217 학습 후보 |

Learned는 0.150m를 이동한 뒤 hold를 받았다. 드라이버는 0 명령과 정지 상태를 0.400초 동안 확인했고, rosbag 위치·속도로 0.360초의 정지 구간을 독립 확인했다. Hold를 해제한 뒤 0.103m 이동했을 때 복귀 조건을 충족했고 이후 목표에 도착했다. Learned·shadow의 유효 접촉 샘플은 각각 252개·150개였다.

두 기록에 대해 Humble에서 관측 381개와 결정론적 행동 344개를 오차 0으로 재생했다. 정책이 사용한 서로 다른 LiDAR 획득 182개의 시각과 값을 rosbag과 대조했다. 같은 입력을 macOS에서 재생했을 때 행동의 최대 절대차는 1.79×10⁻⁷이었다. 초기 동시 실행에서는 learned 100/441, shadow 45/115의 추론이 25ms를 넘겼다. 직렬 실행과 BLAS 스레드 1개를 함께 적용한 반복에서는 344회 중 초과가 0회였다. Shadow와 learned의 운송시간에는 속도 상한과 hold의 운영 차이가 포함된다. [실행별 기록과 원시 파일 SHA-256](../../experiments/m3/warehouse-verification.json)

ROS 노드는 실제 TF·EKF·A* 경로를 사용한다. Lite는 평면 센서·차체·구동 모델과 Lite A* 경로를 사용하며, Gazebo에서만 제한 진단 속도와 중계기 상태 전이가 적용된다. 이 결과로 확인한 범위는 관측·행동 전달과 제한된 정적 폐루프다. 정책 비교는 별도의 실행 조건과 표본으로 판정한다.

## 재현

로컬 작업본의 원본은 `.cowork/transfer/local-m3.WpEZAAME/`에 보관했다. 학습 결과에는 resolved 설정, SB3 ZIP과 옵티마이저 상태, NumPy actor 및 골든 벡터, rollout·진단 로그가 포함된다. Gazebo 결과에는 시뮬레이션 시계, 원시 rosbag, 접촉 기록, 후보·최종 명령, 정책 replay, 실행 SHA와 파라미터 해시가 있다.

```sh
export PYTHONPATH="$PWD/packages/amr_lite:${PYTHONPATH:-}"
MPLBACKEND=Agg python -m amr_lite.cli.train_m3 --steps 65536 --out /tmp/m3-train
MPLBACKEND=Agg python -m amr_lite.cli.train_m3 --steps 1024 --resume /tmp/m3-train/final.zip --out /tmp/m3-resume
python scripts/m3/check_variation.py /tmp/m3-train/bundle
```

Gazebo는 소스 해시를 확인한 Humble·Fortress 작업공간과 [M2에서 고정한 로봇 스택](../m2/baselines.md)을 사용한다. 해당 작업공간을 source하고 헤드리스 Docker에서는 Xvfb로 실행한다.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/m3/run_warehouse.py --mode shadow --bundle /tmp/m3-train/bundle --out /tmp/m3-shadow --domain 86
python3 experiments/m3/run_warehouse.py --mode learned --bundle /tmp/m3-train/bundle --out /tmp/m3-learned --domain 87
python3 scripts/m3/verify_warehouse.py /tmp/m3-learned --bundle /tmp/m3-train/bundle
python -m amr_lite.learning.stamped_replay /tmp/m3-learned/replay.jsonl --bundle /tmp/m3-train/bundle
```

이번 평가는 train 분할의 단일 로봇과 제한된 정적 진단 구간을 사용했다.
