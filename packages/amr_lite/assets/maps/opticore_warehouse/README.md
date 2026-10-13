# opticore-amr warehouse map

`opticore-amr`의 `ros2_ws/src/amr_bringup/maps/`에서 그대로 복사한 ROS
`map_server` 지도다. Lite의 warehouse 시나리오와 Lite↔Gazebo paired 평가가 이
파일을 사용한다. 파일을 고치지 말고, 원본 지도가 바뀌면 새 디렉터리로 추가한다.

| 항목 | 값 |
|---|---|
| 원본 commit | `opticore-amr` `96bced0` (Replace map with 2.0m pinch) |
| 크기 | 1207 × 840 cell, 0.05 m/cell (60.35 × 42.0 m) |
| map 원점 | (−3.18, −16.9) |
| `warehouse_map.pgm` SHA-256 | `9c05eb3ca202e7d370830ef1abf9c86a8757a12e4606bb4b3096a289d5ddc7ea` |
| `warehouse_map.yaml` SHA-256 | `6c8c41a2579f6c84284d5556d6ff1ff195cfd97e81a02f55b2ff67667682c363` |

## 해석 주의

`free_thresh: 0.25`이므로 `map_server`는 미탐색 표시값 205(점유 확률 0.196)를
**free**로 읽는다. 선반 내부와 외벽 바깥이 이 값이다. Lite는 물리 세계를 재현하기
위해 205를 점유로 취급한다(`amr_lite.envs.grid_map.load_ros_map`의
`unknown_values`).

Lite 좌표는 지도 왼쪽 아래 모서리를 원점으로 한다.
`x_lite = x_map + 3.18`, `y_lite = y_map + 16.9`.
