# Run and review

## Offline detector

The reproducible offline path reads the selected ROS 2 bag directly and records
one result for each unique acquisition timestamp:

```sh
uv run python -m tunnel_guard.run --experiment configs/<recipe>.json
```

It is the correct path for complete corpus evaluation because it does not have a
DDS queue. It is not live sensor-to-display latency evidence.

## Browser review

With an existing recorded run and its source bag:

```sh
.venv-iteration/bin/python -m tunnel_guard.review_viewer \
  --run build/goal-final-real \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_platform
```

Open `http://127.0.0.1:8765`. The viewer does not rerun inference. It checks
the source bag identity and acquisition timestamp before showing the saved
output. Red is confirmed intersection evidence, orange is a confirmed
unresolved hazard, yellow is unconfirmed candidate evidence, and grey is an
adjacent object. A displayed box is observed support, not an amodal object
shape; `no_obstacle_observed` is not route clearance.

### Browser 3D replay

The same server exposes `/3d`, with orbit controls, a forward-facing camera,
frame navigation, playback, point-size adjustment and envelope/object toggles.
For the locally retained 201-frame moving sequence:

```sh
.venv/bin/python -m tunnel_guard.review_viewer \
  --run build/perf-100ms-after \
  --bag data/sourcecraft_subset/for_hackathon/roundT_doubleT \
  --port 8766 --display-max-points 60000
```

Open `http://127.0.0.1:8766/3d`. This uses the repository's vendored Three.js
and OrbitControls under `SUSTechPOINTS/public/js/lib`; no CDN is used. Those
files must be present alongside the repository checkout. The existing source
identity and acquisition-time checks also apply to this view. Rendering samples
the display cloud only; saved detector outputs are unchanged.

The camera and clouds are in the per-scan processing frame. This shows the
surroundings passing the sensor, not a reconstructed world map or a validated
train trajectory. Speed selection requests acquisition-time pacing; fetching
and rendering can make playback slower. It is not an inference benchmark.
Adjacent objects are hidden initially and can be enabled with the grey-object
checkbox. A blank distance means no confirmed intersection was reported, not
that the route is clear.

## ROS 2, Docker and RViz

Use the one release command sequence in [ROS2.md](ROS2.md). The live and
recorded-RViz outputs use `/tunnel_guard/...`; no release command uses the
retired `/perception/...` contract. The data path is:

`PointCloud2 → detector → /tunnel_guard/result + distance/attention → cloud + markers`

Run `ros2 bag info` first, launch the node with the bag's actual input topic,
then start `ros2 bag play` in a separate process. Record the node console and
RViz screen for release evidence. A slowed replay is useful for inspecting a
person or unresolved state, but is not a runtime demonstration. The watchdog
must be exercised by stopping playback, and a restarted/rewound bag requires a
new node process because acquisition timestamps are intentionally monotonic.

## Evidence limits

The live node reports callback-to-publish elapsed time and explicit drop
counters. It cannot measure sensor-to-result age or DDS loss without an
established common clock or source sequence number; those fields stay explicit
as unavailable. Target hardware performance, camera/RViz GUI operation and
field safety remain unverified until a full native release run is captured.
