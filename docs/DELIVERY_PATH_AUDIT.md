# Delivery path audit, 2026-09-19

What stands between this repository and an evaluator seeing the detector work: `docker build` → `docker run` →
`ros2 bag play` → the result. Everything below was checked; the defects found are listed with the failure each would have
caused, then the checks that came back clean, then exactly what could not be verified here.

The machine this audit ran on has **no ROS 2 and no Docker daemon**, so every ROS-side change is *standard usage, not
exercised*. That is stated per item rather than left implied.

## Defects found and fixed

| # | defect | failure it would have caused |
|---|---|---|
| 1 | `rviz/tunnel_guard.rviz` sets `Fixed Frame: tunnel_guard_local`, and the node published no TF at all - no `TransformStamped`, no broadcaster | RViz comes up with a missing fixed frame and renders **nothing** |
| 2 | the node defaults to `input_topic: /lidar_points`, and `doubleT_obstacle` publishes on `/sensing/lidar/hesai128/pointcloud` | the documented run on that bag receives **nothing** and degrades to `unavailable` with no indication of why |
| 3 | `tf2_ros` was inherited from the base image rather than declared, immediately after the import was added for defect 1 | the node could fail to import in the container |
| 4 | the build's smoke check imported the native extension but not the node | a missing ROS dependency would surface **in the demonstration**, not in the build |
| 5 | the README had no container instructions at all - no `docker build`, no `docker run`, no DDS guidance - and the container needs `--network host` for host-side `ros2 bag play` to be visible | the evaluator **cannot run it**, or runs it and sees nothing |

Fixes, all in the tree: the node publishes an identity static transform from `tunnel_guard_local` to the frame the
incoming clouds declare (once per source frame, `tf2_ros.StaticTransformBroadcaster`); the silence watchdog logs an error
naming the configured topic, the timeout, the point-cloud topics **actually present**, and the parameter to restart with;
`ros-humble-tf2-ros` is declared; the build also runs `python3 -c "import tunnel_guard.ros_node"`; and the README carries
the full container flow with the reason host networking is required.

## Checks that came back clean

- **Topics**: the RViz config listens to `/perception/points_display` and `/perception/debug_markers`, which is exactly
  what the node publishes, plus `/perception/status`.
- **Parameters**: the launch declares `config`, `input_topic`, `rviz`, `input_reliability`, `input_timeout_s`; the node
  declares the same four plus `display_max_points`; the defaults agree, and the config path matches what the Dockerfile
  copies.
- **Display integrity**: every marker carries a 0.3 s lifetime *and* each published set opens with a `DELETEALL` marker
  (`action=3`, namespace `clear`), so vanished objects are cleared twice over; points travel as a `PointCloud2` with a
  display cap rather than one marker per point; RViz launches only when asked, so the detector runs headless.
- **Build inputs**: all nine `COPY` sources exist and are tracked in git - 60 files under `tunnel_guard/`, 205 under
  `configs/`, the rviz and launch files, the constraints - and none matches a `.gitignore` rule, so a fresh clone builds
  the same image. `CMD` points at `tunnel_guard.ros_node`.
- **Degradation**: empty, single-point, NaN-bearing and inf-bearing clouds return `unknown` /
  `insufficient_returns` / health `unavailable` with no objects and no exception; a backwards timestamp raises
  explicitly; a gap beyond `frame_max_gap_s` sets `gap_reset`.

## Not verified, and why

| item | reason |
|---|---|
| `docker build` | no Docker daemon on the development machine |
| `docker run` with `--network host` | same |
| `ros2 bag play` reaching the node over DDS | same; no ROS 2 either |
| the RViz window and its displays | same |
| `tf2_ros` and `rviz2` present in the image | expected from `ros:humble-ros-base` plus the declared packages, not checked here |
| the demonstration video the submission requires | cannot be produced without a running system |

The container path has been exercised once before, for a ten-scan replay under emulation; that is the only execution
evidence it has.

## What an evaluator should do first

```sh
docker build -t tunnel-guard .
docker run --rm -it --network host tunnel-guard
# in another terminal, on the host:
ros2 bag info <bag>                       # note the point-cloud topic
ros2 bag play -r 0.3 <bag>                # slow replay: queue depth 1, frame time above the stream rate
# if the topic differs from /lidar_points, restart the node with input_topic:=<topic>
```

If nothing appears, the node's own error names the topic it waited on and the point-cloud topics that are present, so the
cause is visible without reading this document.
