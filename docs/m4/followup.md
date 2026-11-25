# M4 후속 개발 실험 · 동역학 혼합 학습

기존 M4는 명목 프로파일 전용 학습과 실측 범위 전용 학습에서 서로 다른 실패를 보였다. 같은 학습 예산에서 두 프로파일을 경험하면 정적 경로 추종과 동적 무충돌 완료를 함께 유지할 수 있는지 검증한다. 모델 고정 전 개발 실험에 한정한다.

- 전체 구성에서 에피소드마다 명목 또는 실측 범위 프로파일을 각각 확률 0.5로 뽑는다. 별도 난수로 배정하고 기존 센서 seed, 실측 프로파일 난수, 임무 순서, 4프레임 관측, 보상, 보호 기능을 유지한다.
- seed 7·19·31, 단계별 65,536전이, 총 589,824전이를 사용한다. 각 seed의 마지막 체크포인트를 사용한다.
- 각 actor를 기존 16개 기본 조건과 8개 추가 조건에 적용하고, 같은 actor로 보호 기능을 끈 Lite 진단도 수행한다. 기존 M3 기준과 [M4 판정 조건](protocol.md)을 그대로 적용한다. 독립 평가 조건은 열지 않는다.
- 실행 시각과 소스·모델 해시는 실험 기록으로 보관한다. 결과에는 기존 전체 구성과의 차이, seed별 동적 완료, 정면 접근 실패 유형을 함께 기록한다.

## 2026-11-28 개발 판정

**혼합 학습 구성을 M5 단일 로봇 운용 검증 후보로 선정한다.** 세 seed 모두 동적 임무에서 무충돌로 한 번 이상 도착했고, M3 대비 동적 완료율이 38.9%p 증가했다. 전체 접촉률, 정적 완료율, 정적 시간 초과율도 사전 기준을 충족했다. M5에는 세 actor를 함께 넘긴다. 배포 모델 선정과 독립 시험은 M6에 남아 있다.

| 기본 평가 | M3 기준 | 기존 M4 전체 구성 | M4 혼합 학습 |
|---|---:|---:|---:|
| 전체 무충돌 완료 | 0/16 | 12/48 | 24/48 |
| 동적 무충돌 완료 | 0/12 | 8/36 | 14/36 |
| 정적 무충돌 완료 | 0/4 | 4/12 | 10/12 |
| 정적 시간 초과 | 2/4 | 8/12 | 2/12 |
| 전체 접촉 | 5/16 | 6/48 | 9/48 |

이전 전체 구성과 비교하면 접촉이 세 건 늘었다. 혼합 구성에서 횡단은 8/12회 완료했고 정면 접근은 6/12회 완료했다. 정면 접근의 나머지 6회는 이동 장애물에 접촉했다. 일시 차단은 3회 접촉·9회 시간 초과였다. 추가 진단에서 복수 장애물은 0/12, 경로 복귀 장면은 4/12회 임무를 완료했다. 같은 actor의 보호 기능을 끈 Lite 기본 평가에서는 접촉이 9/48에서 30/48로 늘었다. [개발 조건별 원시 결과와 판정](../../experiments/m4/followup/mixed.json) · [혼합 구성 재생](../../experiments/m4/followup/mixed-replay.json)

| 학습 seed | 동적 무충돌 완료 | 최종 NumPy 번들 SHA-256 앞 12자리 |
|---|---:|---|
| 7 | 8/12 | `f38425b2143d` |
| 19 | 4/12 | `93316698a120` |
| 31 | 2/12 | `3e1c49176569` |

## 실패 재현과 학습 비용

기존 전체 구성의 정적 임무 12회와 환경 변동 제거 구성의 정면 접근 12회를 다시 실행했다. 원래의 종료 사유·시각·최근접 거리를 대조했다. 정적 실패 중 seed 7은 목표에서 8.43~13.66m 떨어진 곳까지만 접근했다. seed 31은 목표를 0.87m까지 스쳐 지난 뒤 약 4.95m 떨어진 곳에서 보호 정지 상태로 끝났다. 실측 프로파일의 정면 접근 충돌 6건 중 4건은 이동 장애물, 2건은 고정 지도 접촉이었다. seed 7·센서 seed 100의 실측 정면 접근에서 장애물이 8m 이내의 원시 LiDAR 광선을 처음 가린 시각은 8.30초였다. 감속 명령은 10.20초, 실제 속도 감소는 10.35초, 접촉은 12.26초에 기록됐다. 광선 가림은 시뮬레이터의 사후 계측이다. [원래 actor의 재생 기록](../../experiments/m4/followup/replay.json)

소스 `f252685`에서 세 학습을 로컬 CPU로 병렬 실행했다. 실제 학습 구간은 11월 28일 10:58:57~11:01:29 KST로 약 2분 32초였다. 최종 모델을 저장·재로드했고, 9개 단계 체크포인트와 학습 로그의 해시, 번들별 골든 입력 32개를 확인했다. SB3와 NumPy 간 최대 행동 오차는 2.39×10⁻⁷ 이하였다. 후속 평가는 새 시행 144회와 기존 대조군 168회를 사용했다. 표의 48회는 보호 기능을 켠 혼합 구성의 기본 임무만 집계한다.

## 재현

저장소 루트에서 M3 Python 환경을 활성화한다. 원본 M4 학습 모델을 사용하는 재생과 혼합 학습·평가를 아래 순서로 재현할 수 있다. 마지막 평가 명령은 학습 당시의 깨끗한 소스 작업본, 새 결과 파일 경로를 받는다.

```sh
export PYTHONPATH="$PWD/packages/amr_lite"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MPLBACKEND=Agg
python scripts/m4/replay_failures.py "$ORIGINAL_M4_RUNS" /tmp/m4-original-replay.json
git worktree add --detach .cowork/mixed-source f252685
cd .cowork/mixed-source
export PYTHONPATH="$PWD/packages/amr_lite"
for seed in 7 19 31; do
  python -m amr_lite.cli.train_m4 --variant full --seed "$seed" --mix-profiles --out "$MIXED_RUNS/$seed"
done
cd ../..
export PYTHONPATH="$PWD/packages/amr_lite"
python scripts/m4/score_followup.py "$MIXED_RUNS" .cowork/mixed-source /tmp/m4-mixed.json
python scripts/m4/replay_failures.py "$MIXED_RUNS" /tmp/m4-mixed-replay.json --mixed-score /tmp/m4-mixed.json
```

`ORIGINAL_M4_RUNS`와 `MIXED_RUNS`는 각각 원본 학습 결과 디렉터리와 새 학습 결과 디렉터리의 절대경로다.
