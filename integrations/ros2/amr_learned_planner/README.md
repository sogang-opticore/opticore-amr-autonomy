# amr_learned_planner — opticore-amr 통합 패키지

AMR-Lite에서 학습한 local planner를 `opticore-amr`의 ROS2 Humble/Gazebo 스택에서
실행하는 ament_python 패키지다. 기존 `amr_navigation`(A*, `dwa_node`)은 **수정하지
않는다**. DWA는 fallback으로 계속 돌고, `cmd_arbiter`가 `/cmd_vel`과 A*가 듣는
`/dwa/status`의 유일한 발행자가 된다. 계약과 상태 전이는
[ARCHITECTURE.md](../../../docs/ARCHITECTURE.md), 단계와 합격 기준은
[ROADMAP.md](../../../docs/ROADMAP.md) M5·M6을 따른다.

## 구성

| 실행 파일 | 역할 |
|---|---|
| `learned_planner` | scan 재표본화 → 관측 → bundle 추론(NumPy) → scan shield → 가감속 제한. `learned_planner/{cmd_vel,status,health}` 발행 |
| `cmd_arbiter` | 모드 `dwa`/`shadow`/`learned`. `cmd_vel`, `dwa/status`, `cmd_arbiter/state` 발행 |
| `nav_episode_recorder` | `/goal_pose`마다 한 행씩 결과 CSV 기록(수동 주행·데모용) |
| `goal_sequence_runner` | paired set(`packages/amr_lite/configs/paired/*.yaml`)의 goal을 차례로 보내고 Lite와 같은 CSV 형식으로 기록 |

판단 로직은 모두 `amr_learned_planner/core/`(ROS 의존 없음)에 있고, 관측·재표본화·
shield·정책 runtime은 `amr_lite`의 학습 코드와 **같은 함수**를 import한다.

```text
/global_path ─┬─► dwa_node ── dwa_fallback/cmd_vel, dwa_fallback/status ─┐
/lidar, odom ─┴─► learned_planner ── learned_planner/cmd_vel|status|health ─┤
                                                                          ▼
                              A* ◄── dwa/status ── cmd_arbiter ──► cmd_vel
```

## 1. 설치 (opticore-amr 컨테이너)

두 저장소가 `/workspace` 아래에 나란히 있다고 가정한다.

```bash
# AMR-Lite: torch 없이 NumPy·PyYAML만으로 동작하도록 --no-deps
python3 -m pip install --no-deps -e /workspace/opticore-amr-autonomy/packages/amr_lite
python3 -c "import numpy, yaml, amr_lite.deploy.runtime; print('amr_lite runtime OK')"

# 패키지를 opticore-amr 워크스페이스에 연결하고 빌드
ln -s /workspace/opticore-amr-autonomy/integrations/ros2/amr_learned_planner \
      /workspace/opticore-amr/ros2_ws/src/amr_learned_planner
cd /workspace/opticore-amr/ros2_ws
colcon build --packages-select amr_navigation amr_learned_planner
source install/setup.bash
```

`--symlink-install`은 쓰지 않는다(opticore-amr 빌드 관례).

## 2. Policy bundle 준비

학습 checkpoint를 bundle로 export하고 배포 로봇 profile과 대조한다(학습 장비에서
실행, torch 필요).

```bash
cd /workspace/opticore-amr-autonomy/packages/amr_lite
PYTHONPATH=. python -m amr_lite.cli.export_policy \
  --checkpoint artifacts/sweeps/e6_candidate/trials/candidate-s7/best.pt \
  --output /workspace/bundles/e6-candidate-s7
PYTHONPATH=. python -m amr_lite.cli.verify_bundle /workspace/bundles/e6-candidate-s7 \
  --robot-profile opticore_gazebo          # exit 0이어야 learned 모드 후보
```

학습 전 파이프라인만 점검하려면 Phase 0 Stable PPO checkpoint를 export해 shadow로만
쓴다. 이 bundle은 `lite_default` profile이므로 `profile_check:=warn`이 필요하고
`learned` 모드에 쓰면 안 된다.

```bash
PYTHONPATH=. python -m amr_lite.cli.export_policy \
  --checkpoint ../../experiments/phase0_amr_lite/artifacts/checkpoints/ppo_stable_full_best.pt \
  --output /workspace/bundles/phase0-stable-best
```

## 3. 실행 단계 (M5)

시뮬레이션과 localization은 `astar_dwa.launch.py`를 쓸 때와 같이 먼저 띄운다.
`learned_nav.launch.py`가 A*와 DWA를 함께 띄우므로 `astar_dwa.launch.py`는 실행하지
않는다.

```bash
# M5.1 기존 동작 확인 (rollback 모드): DWA가 그대로 주행
ros2 launch amr_learned_planner learned_nav.launch.py mode:=dwa bundle:=/workspace/bundles/e6-candidate-s7

# M5.2 shadow: DWA가 주행, 학습 planner는 계산·기록만
ros2 launch amr_learned_planner learned_nav.launch.py mode:=shadow \
  bundle:=/workspace/bundles/e6-candidate-s7 record:=true record_csv:=/workspace/results/shadow.csv

# M5.3 learned: 학습 planner 주행, 고장 시 0.3 s 정지 후 DWA
ros2 launch amr_learned_planner learned_nav.launch.py mode:=learned \
  bundle:=/workspace/bundles/e6-candidate-s7 record:=true record_csv:=/workspace/results/learned.csv
```

확인할 토픽:

```bash
ros2 topic echo /cmd_arbiter/state     # mode, state(LEARNED/FALLBACK...), 전환 이유, fallback 횟수
ros2 topic echo /learned_planner/health # healthy, faults, inference_ms, deadline_misses, shield_mode
ros2 topic echo /dwa/status            # A*가 보는 effective 상태
```

### Fault injection (G5 확인 항목)

```bash
ros2 param set /learned_planner fault_injection nan_action      # NON_FINITE_ACTION
ros2 param set /learned_planner fault_injection slow_inference  # DEADLINE_MISS (3회 연속)
ros2 param set /learned_planner fault_injection stale_scan      # SCAN_STALE
ros2 param set /learned_planner fault_injection none            # 복구 → 2 s 정상 후 LEARNED 복귀
```

`mode:=learned`에서 각 fault는 `cmd_arbiter/state`에 `LEARNED → FALLBACK_HOLD →
FALLBACK` 전환과 이유로 남아야 한다. 존재하지 않는 `bundle:` 경로로 띄우면
`BUNDLE_LOAD_FAILED`로 처음부터 DWA가 주행한다.

## 4. 다중 로봇

`amr_bringup/launch/multi_robot.launch.py`의 로봇별 A*/DWA `TimerAction` 두 개를
아래 노드 묶음으로 바꾸면 된다(opticore-amr 쪽 변경은 팀 결정 후 진행).

```python
from amr_learned_planner.launch_helpers import make_learned_nav_nodes

TimerAction(period=20.0, actions=make_learned_nav_nodes(
    mode="shadow", bundle_path="/workspace/bundles/e6-candidate-s7", namespace=rn))
```

namespace를 주면 입력 토픽은 상대 이름(`odometry/filtered`, `lidar`, ...), frame은
`<rn>/odom_filtered`로 바뀌고 DWA의 출력은 `/<rn>/dwa_fallback/*`로 remap된다.

## 5. Gazebo paired 평가 (M6)

같은 goal 순서를 Gazebo와 Lite에서 실행하고 비교한다.

```bash
# Gazebo: DWA와 학습 planner를 각각 (planner 모드만 바꿔서) 실행
ros2 launch amr_learned_planner paired_runner.launch.py paired_set:=warehouse_v1 runs:=3 \
  label:=gazebo_learned output_csv:=/workspace/results/gazebo_learned.csv

# Lite: 같은 다리(leg)를 여러 seed로
cd /workspace/opticore-amr-autonomy/packages/amr_lite
PYTHONPATH=. python -m amr_lite.cli.paired lite --policy bundle:/workspace/bundles/e6-candidate-s7 \
  --set warehouse_v1 --seeds 0 1 2 3 4 --label lite_learned --output /workspace/results/lite_learned.csv

# 차이 보고서 (reference 대비 candidate)
PYTHONPATH=. python -m amr_lite.cli.paired gap --reference /workspace/results/lite_learned.csv \
  --candidate /workspace/results/gazebo_learned.csv --output /workspace/results/GAP_REPORT.md
```

같은 `gap` 명령으로 Gazebo DWA 대비 Gazebo 학습 planner도 비교한다.

## 6. 테스트

ROS 없이도 core와 노드 배선을 검사한다(rclpy가 없으면 `test/fake_ros.py`의 가짜
rclpy를 쓴다).

```bash
cd /workspace/opticore-amr-autonomy
python -m pytest -q integrations/ros2/amr_learned_planner/test
# ROS 환경
colcon test --packages-select amr_learned_planner && colcon test-result --verbose
```

`test_core_observation_matches_lite_environment`와 golden vector 테스트는 같은 raw
scan·odom·path에서 ROS core 관측이 Lite 관측과 `1e-6` 안에서 같은지 확인한다.

## 7. 알려진 한계

- Gazebo에는 접촉 센서가 없어 recorder의 `collision`은 LiDAR 점이 0.36 m 원형
  footprint 안으로 들어온 경우를 세는 proxy다(실제 차체는 0.60 × 0.40 m 직사각형).
- LiDAR 장착값은 파라미터(`lidar_mount_*`, URDF 기준 0.25 m)이며 TF에서 읽지 않는다.
- 동적 장애물 no-go layer(`/dynamic_obstacle_layer`)는 계속 DWA가 발행한다. 학습
  planner는 `EMERGENCY`/`PATH_LOST`로만 A* 재계획을 요청하고 DWA의 동적 상태
  (`DYNAMIC_BLOCKED` 등)는 내지 않는다.
- goal의 최종 yaw는 추종하지 않는다(기존 DWA와 같음). 후진은 사용하지 않는다.
- 실제 ROS 실행 검증은 아직 수행하지 않았다. 이 패키지는 가짜 rclpy 위의 배선
  테스트와 Lite 폐루프 테스트까지만 통과한 상태다.
