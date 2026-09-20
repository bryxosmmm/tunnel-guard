# ROS 2 Humble live node

The repository now contains a live ROS 2 adapter in `tunnel_guard/ros_node.py` and an
`ament_python` package under `ros2_ws/src/tunnel_guard_ros`. It subscribes to one
`sensor_msgs/msg/PointCloud2` stream, uses the same padded/endianness-aware decoder as the
offline bag runner, and keeps one stateful `Detector` for the lifetime of the node.

The node publishes:

- `/tunnel_guard/result` (`std_msgs/msg/String`): the complete JSON result, including status,
  health, objects, range support and timing;
- `/tunnel_guard/obstacle` (`std_msgs/msg/Bool`): `true` for both `obstacle` and
  `unresolved_obstacle`, so an alert consumer cannot silently ignore an unresolved hazard;
- `/tunnel_guard/nearest_obstacle_m` (`std_msgs/msg/Float32`): nearest confirmed hazard distance
  in the configured processing frame, or NaN when none is reported;
- `/tunnel_guard/points` (`sensor_msgs/msg/PointCloud2`): transformed points in
  `tunnel_guard_local`, sampled only for visualization;
- `/tunnel_guard/markers` (`visualization_msgs/msg/MarkerArray`): observed-support boxes and
  status text for RViz2.

The boolean topic is an alert transport, not a calibrated collision probability. Consumers
must inspect `result.status` and `result.health`; `no_obstacle_observed` does not establish a
clear route, and `unresolved_obstacle` is not a proven collision.

## Docker build and run

The root `Dockerfile` uses Ubuntu 22.04 / ROS 2 Humble, installs the pinned Python detector,
builds the ROS package with `colcon`, and includes RViz2. Build on an x86_64 host with network
access to the ROS and Python package registries:

```sh
docker build -t tunnel-guard:humble .
docker run --rm -it --net=host --ipc=host \
  tunnel-guard:humble
```

The default command launches `tunnel_guard.launch.py` and subscribes to `/lidar_points`. For a
different topic or configuration:

```sh
docker run --rm -it --net=host --ipc=host tunnel-guard:humble \
  ros2 launch tunnel_guard_ros tunnel_guard.launch.py \
  input_topic:=/hesai/pandar_points \
  config_path:=/opt/tunnel-guard/configs/detector.json
```

To replay a ROS 2 bag from the host, mount it read-only and run the node and playback in
separate terminals (the container uses host networking):

```sh
docker run --rm -it --net=host --ipc=host \
  -v "$PWD/data:/data:ro" tunnel-guard:humble
ros2 bag play /path/to/bag --clock
```

RViz2 can use `rviz/tunnel_guard_live.rviz` directly; the older `rviz/tunnel_guard.rviz` remains
the recorded-result replay configuration. The ROS 2 node and Docker image are source deliverables;
target-host latency, QoS compatibility with the actual driver, and GUI operation still need
to be demonstrated on the Intel/Ubuntu target.
