# ROS 2 Humble release contract

The supported live adapter is `python3 -m tunnel_guard.ros_node` (normally via
`ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py`). It uses one
topic namespace, `/perception/...`.

```sh
docker build --platform linux/amd64 -t tunnel-guard:humble .
docker run --platform linux/amd64 --rm -it --name tunnel-guard \
  -v "$PWD/data:/data:ro" tunnel-guard:humble \
  ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
  input_topic:=/lidar_points input_timeout_s:=3.0
docker exec tunnel-guard bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /data/<bag> --rate 1.0'
```

Use `ros2 bag info /data/<bag>` before launch and set `input_topic` to its real
PointCloud2 topic. The subscription is `KEEP_LAST(1)` and configurable as
`reliable` (default) or `best_effort`; this bounds obsolete queued work, but
DDS loss cannot be counted without a source sequence number.

| Topic | Type | Meaning |
|---|---|---|
| `input_topic` | `sensor_msgs/msg/PointCloud2` | source cloud; header time drives detector evidence |
| `/perception/status` | `std_msgs/msg/String` | full result JSON |
| `/perception/attention_required` | `std_msgs/msg/Bool` | true except for `no_obstacle_observed`; not a braking command |
| `/perception/nearest_obstacle_m` | `std_msgs/msg/Float32` | nearest confirmed reported hazard, or NaN |
| `/perception/points_display` | `sensor_msgs/msg/PointCloud2` | sampled display cloud |
| `/perception/debug_markers` | `visualization_msgs/msg/MarkerArray` | supported contour, observed support and status |

All output publishers are reliable `KEEP_LAST(1)`; recorded RViz export writes
the same five-topic set. `status` distinguishes
confirmed, candidate and unresolved ranges; `no_obstacle_observed` never means
the route is clear. The `attention_required` topic is only a fail-visible review
signal, not a validated train-control decision.

If no fresh, successfully processed acquisition arrives within `input_timeout_s`,
the watchdog resets detector state and publishes `unknown` with an empty cloud,
`DELETEALL` markers, attention true and a NaN obstacle distance. Duplicate messages
do not renew freshness. The timer uses a steady clock, including when
`use_sim_time=true` and `/clock` stops. The acquisition watermark survives resets.
Received, processed, invalid, duplicate and out-of-order counters remain separate;
out-of-order rejection does not also count as invalid. `result_age_s` stays null:
acquisition and host clocks have not been proven comparable.
`callback_processing_s` is local elapsed time, not sensor-to-warning latency.
The single-threaded executor cannot interrupt a blocked inference callback; the
watchdog is not a hard real-time deadline.

All outputs use `tunnel_guard_local`, detector-local processing coordinates. No
TF, identity transform, surveyed mounting, or vehicle extrinsic is asserted.

Run RViz separately in a properly configured GUI environment:

```sh
rviz2 -d /opt/tunnel-guard/rviz/tunnel_guard.rviz
```

Apple Silicon emulation is a compatibility check only. Full-bag runtime,
watchdog/restart, RViz and video acceptance must still be captured on the target
Ubuntu 22.04 / Humble deployment machine.

Use the [target deployment acceptance record](TARGET_ACCEPTANCE.md) to capture
that evidence, including the image identity, input/output bags, watchdog
transition and RViz session.

## Measured delivery evidence — 2026-09-25

The linux/amd64 image built and ran under Apple Silicon emulation. Actual DDS
replays reproduced the original duplicate-stream and stopped-clock timeout failures,
then exercised the corrected adapter. The accepted normal-clock recording contains
33 messages on each of the five topics; the frozen-clock recording contains eight
on each. Unknown outputs clear display support, require attention, carry NaN
distance and invent neither an acquisition timestamp nor result age. Strictly newer
acquisitions restore processing. No TF topics were published.

These are delivery experiments, not complete-input evaluation: the normal-clock
run processed 29 acquisitions from 32 offered valid scans under `KEEP_LAST(1)`.
An earlier recording missed the first status message and remains in the evidence;
the complete recording waits for output discovery before replay. The five topics
are not an atomic multi-topic snapshot; use the stamped full status as authoritative.

The offline 30-scan export preserved all inspected detector objects, geometry,
poses, acquisition identities and range coverage against the verified best run.
See [portable evidence](../results/issue33-delivery-20260925.json) and its ZIP
for recipes, replay/recorder code, logs, image identities and separate evaluation.
Full output bags remain under `build/issue33-live/`. RViz GUI, full-bag timing
and deployment-host acceptance are not established by this emulated run.
