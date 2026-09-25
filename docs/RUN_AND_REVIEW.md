# Run and review

## Browser review on macOS / Linux

No ROS, external web service, CDN or browser package is required. The viewer
reads recorded JSONL decisions and the exact corresponding source cloud. It
never reruns inference, and it rejects a different bag metadata hash or a missing
acquisition timestamp. Source clouds stay local. Only their display copy is
sampled; candidate boxes and detector decisions are unchanged.

From the repository root, with the existing environment and real data:

```bash
.venv-iteration/bin/python -m tunnel_guard.review_viewer \
  --run build/goal-final-real \
  --bag data/sourcecraft_subset/for_hackathon/doubleT_platform
```

Open `http://127.0.0.1:8765`. Use the slider/number field for any recorded frame,
arrow buttons for stepping, and play for sequential review. Wheel zooms, dragging
pans, selecting an object in the table focuses it. XY and XZ views use equal metric
scale. Nearby infrastructure and text labels can be toggled without changing the
recorded detector output. All source returns are eligible for display sampling;
the display is not a claim that the detector evaluated every rendered point.

Red: confirmed intersection. Orange: a confirmed corridor-hazard candidate whose
intersection is not confirmed. Yellow: candidate. Gray: adjacent, even when its
measured presence is confirmed. Cyan: reference contour and distance
support point. Missing contour means unsupported geometry, not unlimited clearance.
Displayed distance is forward x in the configured processing frame, not distance
along a curved path or from the front bumper.

The table also shows measured XYZ extent and total/interior support counts.
Click an object to pause and inspect its confirmation reason. For saved diagnostic
frames the browser overlays its exact current-scan voxel representatives without
sampling; otherwise it explicitly reports that these points were not saved.
A small support box is not an estimate of the full object's size, and red means
intersection of the model rather than a verified physical collision hazard.
See [the current small-detection review](ENVELOPE_INTERVAL_REVIEW.md) for the
updated real run and example platform frame 223. The run name appears in the toolbar.

The `--run` directory must contain `<bag-name>.jsonl`, `detector.json`, and
`manifest.json` from `tunnel_guard.run`. `--bag` points to the original bag folder.
Arbitrary seeking works through bag record timestamps; inference uses acquisition
timestamps. Both identities are checked. Deskew-enabled runs require the recorded
RViz export instead: the original raw cloud would not match deskewed boxes.

## ROS2 Humble container

Container build and runtime evidence are tracked in the iteration report; do not
infer successful deployment merely from these commands.

```bash
docker build --platform linux/amd64 -t tunnel-guard:goal-amd64 .
docker run --platform linux/amd64 --rm -it --name tunnel-guard \
  -v "$PWD/data/sourcecraft_subset/for_hackathon:/data:ro" \
  tunnel-guard:goal-amd64 \
  ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
  input_topic:=/lidar_points input_timeout_s:=15.0
```

Find the actual source topic first (`ros2 bag info /data/<bag>` inside the
container); replace `input_topic` accordingly. In another terminal, replay within
that same container so no host DDS configuration is needed:

```bash
docker exec tunnel-guard bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /data/doubleT_platform --rate 0.02 --wait-for-all-acked 30000'
```

The suggested slow rate is for inspection, **not evidence of 10 Hz capability**.
The live subscription has depth 1. Reliable QoS is the default for the supplied
bags. For a best-effort hardware publisher, explicitly select
`input_reliability:=best_effort`; a reliable subscriber cannot receive from a
best-effort publisher. Under overload, scans may still be dropped. Missing scans cannot be counted exactly without a source sequence
counter. Processing time and measurement gaps remain visible. Full quantitative
comparison uses the offline runner (`python -m tunnel_guard.run --experiment ...`),
which processes every selected unique source measurement without a DDS queue.

Outputs:

- `/perception/points_display`: processed-frame cloud, sampled only for display.
- `/perception/debug_markers`: reference contour, measured support boxes, IDs,
  support points, distances, confirmation, and data quality.
- `/perception/status`: full result JSON with exact acquisition nanoseconds.
- `/perception/attention_required` (`std_msgs/msg/Bool`): true for every state
  except `no_obstacle_observed`, including `unknown`; this is not a braking
  command or movement authority.
- `/perception/nearest_obstacle_m` (`std_msgs/msg/Float32`): nearest confirmed
  reported hazard in the processing frame, or NaN.

Recorded RViz export writes this same five-topic set; it is recorded inference,
not a second detector implementation.

The adapter uses the same `decode_cloud`, `Detector`, and `ResultMessages` as the
offline path. Lack of fresh acquisitions beyond `input_timeout_s` publishes `unknown`
and clears cloud/markers and tracking. Duplicate acquisition times never renew the
watchdog or add temporal confirmation; the watermark survives silence. The watchdog
uses a steady clock even with paused simulation time. Backward time or a source-frame
change requires restarting with the appropriate config/epoch. Raw measurement
time is never subtracted from wall-clock time to invent a latency value. The JSON
records callback elapsed time, ingress/drop counters and separate
confirmed/candidate/unresolved ranges. It leaves `result_age_s` null until the
acquisition and host clocks are proven comparable.

On Ubuntu with an existing graphical session, pass its display and Xauthority
using your normal container GUI setup, then launch with `rviz:=true`. This image
includes RViz2 and the repository configuration. On macOS, Docker alone does not
provide a native X11 desktop: use the browser viewer above for immediate review.
No host-wide `xhost +` or disabled access control is required or recommended.

The image pins core perception dependencies for Python 3.10 and NumPy 1.x to match
Humble's native message ABI. Platform-specific performance and detection results
must be reported separately from the macOS Python 3.12 reference runs.

References: [ROS2 Humble QoS](https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html),
[official ROS Docker images](https://hub.docker.com/_/ros).

### Architecture

The delivery image targets the specified Intel/Ubuntu machine (`linux/amd64`).
Open3D 0.19 does not publish an official Linux ARM64 wheel because of a release
linker issue ([upstream #7130](https://github.com/isl-org/Open3D/issues/7130)).
On Apple Silicon this image runs under emulation; it is a runtime compatibility
check, not a speed measurement. Native macOS CLI/browser review retains the
existing Open3D 0.19 environment. No silent downgrade to Open3D 0.18 or unofficial
replacement wheel is used.

The 15 s input timeout in the slow-replay example exceeds its 5 s inter-scan wall
interval. Live 10 Hz operation defaults to 3 s. These are wall-clock watchdog
settings; temporal confirmation always uses the original acquisition time.

### Recorded review versus live display

RViz markers retain their existing 0.3 s lifetime. At a deliberately slowed bag
rate they can disappear between messages unless RViz uses the bag playback clock
(`use_sim_time:=true` and `ros2 bag play --clock`). The browser viewer retains the
selected recorded frame and is the verified interactive review interface here.
The live watchdog is wall-clock based and is independent of the playback clock.
All published detector outputs already use `tunnel_guard_local`; the adapter
publishes no identity TF and makes no mounting/extrinsic calibration claim.

The offline CLI also works in an installed image without a Git checkout. Its
manifest then records `git_revision: null` and `git_status: null`; source/config
hashes and source snapshots remain available. Mount the experiment recipe and
output directory explicitly, as shown by `configs/goal-ros-offline-prefix.json`.

### Viewer recovery and disk cleanup (2026-09-17)

The initial single-connection HTTP server was found listening but unresponsive
(15 s HTTP timeout). It now uses `ThreadingHTTPServer`, a 15 s per-connection
socket timeout and a separate bag Reader per frame request, so an idle browser
connection cannot monopolize the server or share SQLite reader state across
threads. The repaired server returned metadata for all 345 platform frames and
real frame 344 with 28,179 display points; browser reload was inspected.

To reclaim local disk, the two locally built `tunnel-guard` images and the 19
identified cache records from this iteration were removed. Existing unrelated
images, containers and volumes were retained. Three extracted copies under
`build/goal-data/for_hackathon/` (about 13 GiB) were removed after checking their
filenames against a successful full listing of the retained archive. Results,
source snapshots, annotations and the archive remain. Restore only the recording
needed for a new run, for example:

```bash
mkdir -p build/goal-data
tar --zstd -xf archive/for_hackathon.zst -C build/goal-data \
  for_hackathon/squareT_platform_squareT_switch
```

The platform browser example uses the existing `data/sourcecraft_subset` copy
and needs neither this extraction nor Docker. Rebuilding the Humble image is an
explicit separate operation; it is not necessary to view results on macOS.
