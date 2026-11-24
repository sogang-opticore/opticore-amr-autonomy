# M5-R1 · 단일 로봇 명령 전환과 복구

M2 창고 launch의 `planner:=b1 command:=operational` 구성이 `operational_arbiter`를 로봇별 최종 명령·상태 발행자로 사용한다. 기존 M3 번들 또는 같은 규약으로 내보낸 SB3 actor, 새 replay 파일 경로를 지정한다. 아래 명령은 Humble·Fortress, 빌드된 ROS workspace와 로컬 번들이 준비된 환경용이다.

```sh
ros2 launch amr_learned_planner m2_baseline.launch.py \
  planner:=b1 command:=operational robot:=amr1 \
  m3_bundle:="$BUNDLE_DIR" m3_trace:="$NEW_TRACE_PATH"
```

| 입력·사건 | 최종 동작과 증거 |
|---|---|
| 새 `goal_pose` | 즉시 정지하고 이전 명령을 폐기한다. 현재 목표 ID, 지도 좌표와 경로 끝점, odometry 획득 시각의 TF 및 센서 조건이 맞아야 주행한다. |
| `cancel_goal` | 로봇 namespace의 `std_msgs/String`에 활성 목표의 `map:<sec>.<9자리 nanosec>`를 담는다. 일치하면 즉시 정지하고 새 목표가 올 때까지 `CANCELLED`를 유지한다. |
| 입력 누락·지연, TF 불일치, 진단 hold | 두 주행기의 명령을 차단한다. 20 Hz로 현재 목표의 정책 보고(최대 0.20초), scan·odom(각 최대 0.50초), 유효 경로를 확인한다. |
| 정책 추론 시간 초과·비정상 행동 | 동일 입력이 정상인 경우만 0.30초간 정지 후 새로운 DWA 명령(최대 0.25초)을 받는다. 정상 정책 보고가 1초간 지속되면 learned로 복귀한다. 두 소스 모두 0.45 m/s와 0.6 rad/s의 제한을 적용한다. |
| 일시 정체 | 유효 관측으로 계속 주행을 시도하며 장애물이 움직이면 정상 주행을 재개한다. |
| 지속 정체·경로 누락 | odom 기준 0.12m 미만 이동이 3초간 지속되거나 경로가 3초간 비면 정지하고 `dwa/status=DYNAMIC_BLOCKED`를 발행한다. 기존 A*가 재계획한다. 요청 시각 이후 같은 목표에 대응하는 경로가 와야 다시 진행한다. 15초 정체는 `BLOCKED_TIMEOUT`, 15초 경로 미수신은 `PATH_TIMEOUT`으로 기록한다. |
| 결과 계측 | `cmd_arbiter/state`에 목표 ID, 시각, 상태, 명령 출처를 기록한다. recorder는 같은 목표의 취소·차단 종료 및 정책 이상·추론 지연을 집계한다. |

위 시간·거리 값은 개발용 기본 설정이다. M5-R1 검증은 ROS 메시지 콜백과 기록기를 통과하는 오프라인 회귀시험이다. 실제 Gazebo 임무에서 감속·정지 거리, DWA 전환, 경로 재수신, 2대 호환을 측정하는 작업은 M5-R2/R3에 남아 있다.

```sh
python -m pytest -q integrations/ros2/amr_learned_planner/test/test_m5_operations.py
```
