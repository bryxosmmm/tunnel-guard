# ROS 2 Humble release path

The supported live adapter is `ros2 launch tunnel_guard_ros
tunnel_guard.launch.py`. It is the only live ROS contract in this repository;
the compatibility module `tunnel_guard.ros_node` invokes the same adapter.

## Build and run

Build on the intended Intel/Ubuntu target. An Apple Silicon host can build and
run `linux/amd64` under emulation, but that is only a compatibility check and
cannot establish target runtime.

```sh
docker build --platform linux/amd64 -t tunnel-guard:humble .
docker run --rm -it --name tunnel-guard --ipc=host \
  -v "$PWD/data:/data:ro" tunnel-guard:humble \
  ros2 launch tunnel_guard_ros tunnel_guard.launch.py \
  input_topic:=/lidar_points input_timeout_s:=3.0
```

Find the point-cloud topic in the bag before starting the node. The supplied
recordings do not all use the same input topic.

```sh
docker exec tunnel-guard bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag info /data/<bag>'
docker exec tunnel-guard bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /data/<bag> --rate 1.0'
```

For another source topic, restart the node with `input_topic:=<topic>`. For a
best-effort publisher, use `input_reliability:=best_effort`; recorded ROS bags
normally offer reliable reliability. The subscription deliberately has
`KEEP_LAST`, depth 1. It prevents a slow detector from reporting an unbounded
backlog as current observations, but it also means DDS loss cannot be counted
without a source sequence counter.

Run RViz in a third terminal in the same container (or with an equivalent
properly configured GUI environment):

```sh
docker exec -it tunnel-guard bash -lc \
  'source /opt/ros/humble/setup.bash && rviz2 -d /opt/tunnel-guard/rviz/tunnel_guard_live.rviz'
```

The same RViz topic names are used by recorded result exports. A recorded
export is a review artifact, not live inference.

## Live topic contract

| Topic | Type | QoS | Meaning |
|---|---|---|---|
| configured `input_topic` (default `/lidar_points`) | `sensor_msgs/msg/PointCloud2` | `KEEP_LAST(1)`, selected reliable or best-effort | source cloud; its header stamp is the acquisition timestamp |
| `/tunnel_guard/result` | `std_msgs/msg/String` | reliable, `KEEP_LAST(1)` | complete JSON result, including state, evidence, counters and timing scope |
| `/tunnel_guard/obstacle` | `std_msgs/msg/Bool` | reliable, `KEEP_LAST(1)` | attention-required flag; true for anything except `no_obstacle_observed`, including `unknown`; **not** a braking command |
| `/tunnel_guard/nearest_obstacle_m` | `std_msgs/msg/Float32` | reliable, `KEEP_LAST(1)` | nearest confirmed reported hazard in the processing frame, or NaN |
| `/tunnel_guard/points` | `sensor_msgs/msg/PointCloud2` | reliable, `KEEP_LAST(1)` | sampled processing-frame display cloud |
| `/tunnel_guard/markers` | `visualization_msgs/msg/MarkerArray` | reliable, `KEEP_LAST(1)` | reference contour where supported, observed-support boxes, state and distance text |

The result has five detector states: `obstacle`, `unresolved_obstacle`,
`candidate`, `no_obstacle_observed`, and `unknown`. It separately reports
`nearest_obstacle_m` (confirmed), `nearest_candidate_m` (not confirmed), and
`nearest_unresolved_range_m` (geometry unresolved). `no_obstacle_observed`
does not mean the route is clear. Neither the states nor the Bool topic is a
validated command to brake or continue.

Input cloud header time drives detector temporal evidence and is copied to
per-frame display messages. The node reports local monotonic
`callback_to_publish_s`; it deliberately leaves `result_age_s` null because the
sensor/bag clock and the host clock have not been proven comparable. Inventing
an age from those two clocks would be false telemetry.

All detector outputs use `tunnel_guard_local`, the configured processing-frame
coordinates. The node publishes no TF and no identity transform. In particular,
the frame name does not assert surveyed sensor mounting, vehicle extrinsics, or
calibration.

## Failure and replay behavior

Each valid result includes received/processed counts, invalid, duplicate and
out-of-order drops, and an explicit `input_loss_count: null` explanation when
DDS loss is unknowable. Duplicate acquisition timestamps never create temporal
evidence. Backward timestamps or a changed source frame publish `unknown`; a
new bag epoch must start a new node.

After `input_timeout_s` of silence, the watchdog resets detector state,
publishes `unknown`, sets the attention flag, emits an empty cloud and a
`DELETEALL` marker. Thus stale boxes and points are removed rather than
continuing to look current. The timeout is wall-clock time and is independent
of bag playback time. For deliberately slow playback, set it longer than the
inter-message wall interval.

This repository does not contain target-hardware Docker, full-bag live DDS, or
RViz GUI evidence. Those must be run and saved on the target as part of release
acceptance; the commands above are the entire required command path.

## Bounded Ubuntu server integration run (2026-09-27)

An Ubuntu 22.04 x86-64 server with eight virtual CPUs and 15 GiB RAM built the
image with `ROS_APT_MIRROR=https://mirror.umd.edu/packages.ros.org/ros2/ubuntu`.
The base image's signed ROS repository configuration remained in place. The
container compiled the native C++ module and the ROS package. The entrypoint
sources ROS setup scripts before enabling Bash `nounset` because the ROS setup
scripts read unset shell variables.

A derivative bag containing the first ten original serialized PointCloud2
messages of `roundT_doubleT` was replayed. The offline native detector emitted
ten rows (one candidate, nine unresolved obstacles), with median processing
427 ms on this server. Live replay at 0.25x did not process all ten messages.
At 0.1x, live ROS published ten results with ten distinct original acquisition
timestamps and identical frame statuses to the offline run, then emitted
`unknown` after input stopped. This does not establish ten-hertz operation,
full-bag quality, or sensor-to-display latency. The cloud was also inspected
in the browser 3D viewer served through an SSH tunnel; RViz GUI was not run.
