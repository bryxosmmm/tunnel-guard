# Frozen detector delivery and demo

## Source boundary

The working release candidate is **8539a3e1a8764a49b3647558c895a611eebf6a13**
(PR #35), selected after the user said the owner's last change was probably final.
This is a working assumption, not an explicit published freeze approval.
PR #35's description and owner comments still identify
**ec311a0bc78363791b5fe9bfaf1ea759bbf2b16a** as the verified baseline.
The later commit changes `detector.py` and `segmentation.py`; it is not a delivery-only commit.
Both complete source variants and the fetched PR metadata are retained in the evidence.

**Question for Semyon Morev:** does the freeze include the instance separation change
`8539a3e`, or does the release retain `ec311a0`? The delivery patch is measured separately
on both. No detector/configuration change is included in this PR relative to `8539a3e`.

`configs/detector.json` SHA256 in both variants:
`f1b0d0ea760c8b3a4e7171b6e9feb37c7e0ac9a38a5c310517b3fa41bcab9efa`.
The delivery follows PR #35's five `/perception/...` topics and imports PR #38's
Humble RViz topic-property correction. Do not combine it with the
`experiments/buyanov` `/tunnel_guard/...` launch/package.
Fixed frame is `tunnel_guard_local`; there is no identity TF.

## Local CLI and browser demo

Use Python 3.12 with the locked dependencies, build native from this checkout,
and place the original bag at `data/sourcecraft_subset/for_hackathon/doubleT_obstacle`.
The source archive supplied by the user is
`/Users/pbuyanov/Downloads/датасет/for_hackathon.zst`.
Do not overwrite an evidence directory: each run uses a new output path.

```sh
uv sync --locked
uv run python setup.py build_ext --inplace
uv run python -m tunnel_guard.run --experiment configs/delivery-demo.json
# Retain the annotation verbatim at its exact source commit.
git show 958cdb0db317cdb72006b6f75a8e138087946daa:annotations/doubleT-obstacle-person-crossing.json > build/person-crossing.json
uv run python -m tools.render_delivery_demo --run build/delivery/demo \
  --annotations build/person-crossing.json --output build/delivery/demo.html
python3 -m http.server 8765 --directory build/delivery
```

Open `http://localhost:8765/demo.html`. This is a browser review of actual outputs,
not RViz acceptance. It shows the unmodified measured display cloud, reference
contour, detector support, distance, presence confirmation and intersection confirmation.
A dashed box is the reviewed full-person volume from PR #40, not a replacement
for the detector's observed support. Target selection uses maximum 3D IoU for review;
it is not a claim that the annotation's IoU acceptance threshold was met.

Story: frames 0 and 6 show adjacent/boundary evidence; 10, 30 and 60 show the crossing;
70, 80 and 104 show the exit. The transition around 69–77 is uncertain in the
annotation provenance; frame 80 is the clear outside example. The scene can still
say `obstacle` due to another component after the person exits. Do not hide it or
present that status as the person's state. Neither boxes nor thresholds are adjusted.
Repeated frames are one event, and the labels are nonexhaustive.

## Actual ROS delivery experiment

Build the final source, then run the recorder and real node in the same container.
The complete recipe waits for each result, exercises duplicates/silence under a frozen
simulation clock, and resumes strictly newer original acquisition timestamps.
The separate rate-1 recipe offers the whole source at its recorded relative timing
on a producer thread; it does not wait for inference. Both record all five outputs.

```sh
export TG_BAG="$PWD/data/sourcecraft_subset/for_hackathon/doubleT_obstacle"
mkdir -p build/delivery/ros
# On Apple Silicon this is emulation, not an Intel benchmark.
docker build --platform linux/amd64 -t tunnel-guard:delivery .
docker run --platform linux/amd64 --rm \
  -v "$TG_BAG:/input/bag:ro" -v "$PWD/build/delivery/ros:/evidence" \
  -v "$PWD/tools:/recipe:ro" -v "$PWD/configs:/recipes:ro" \
  tunnel-guard:delivery bash -lc \
  'source /opt/ros/humble/setup.bash && python3 /recipe/replay_delivery.py /recipes/delivery-ros-complete.json /evidence/complete'
docker run --platform linux/amd64 --rm \
  -v "$TG_BAG:/input/bag:ro" -v "$PWD/build/delivery/ros:/evidence" \
  -v "$PWD/tools:/recipe:ro" -v "$PWD/configs:/recipes:ro" \
  tunnel-guard:delivery bash -lc \
  'source /opt/ros/humble/setup.bash && python3 /recipe/replay_delivery.py /recipes/delivery-ros-rate1.json /evidence/rate1'
uv run python -m tools.render_delivery_demo --run build/delivery/demo \
  --annotations build/person-crossing.json --live build/delivery/ros/complete \
  --output build/delivery/demo-with-freshness.html
```

For interactive ROS/RViz use one launch path in that image:

```sh
source /opt/ros/humble/setup.bash
ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
  config:=/opt/tunnel-guard/configs/detector.json \
  input_topic:=/sensing/lidar/hesai128/pointcloud input_timeout_s:=3.0 rviz:=true
ros2 bag play /input/bag --rate 1.0
```

See [TARGET_ACCEPTANCE.md](TARGET_ACCEPTANCE.md) for display mounts, QoS capture
and a GUI recording. A slow explanatory replay is separate from rate-1 evidence.

## Timing definitions

- `decode_s`: PointCloud2 fields into calibrated detector arrays.
- `detector_process_s` (ROS) / `inference_s` (CLI): the whole `Detector.process` call.
- `display_cloud_s`, `display_markers_s`, `display_build_s`: display construction.
  Marker timing includes contour, support and object labels; the build total also
  includes quality/status JSON. Cloud copying is included in cloud timing.
- `publication_calls_s`: time spent in the five rclpy publish calls, not subscriber receipt.
- `callback_to_publish_return_s`: callback entry to the last publish return; excludes
  incoming DDS queue and subsequent log emission. `callback_processing_s` in status
  ends before display construction and must not be called a full callback measurement.
- Recorder send/receipt timestamps are host-monotonic elapsed times. Their difference
  is offered-input-to-recorded-result delay, including inference, queues, transport
  and recorder work. It is not sensor age. `result_age_s` remains null.
- Offered-minus-processed acquisitions measures omissions across this delivery path;
  without a source sequence counter it cannot attribute each omission to DDS.
  KEEP_LAST depth 1 bounds the pending queue but cannot preempt an active callback.

No automated tests or substitute detector were created or run. Verification uses
actual continuous detector execution, real DDS recordings and comparison of their
outputs. Mac/native, Mac/amd64-emulated ROS and the unmeasured target Intel are distinct.

## What was actually completed on 2026-09-26

See [the measured report](../results/delivery-profile-20260926.md) and the [portable selected-message archive](../results/delivery-demo-evidence-20260926.zip). The browser demo includes an actual **startup without input** watchdog
recording under frozen simulation time. It does not claim the requested complete
fresh-scene → lost-input → recovery live sequence passed.

The amd64 image built, but Open3D 0.19.0 raised SIGILL on the first input in this
Docker Desktop emulator (reported CPU lacks AVX). Linux/arm64 could not install
that pinned Open3D version for Python 3.10. We retained both failures and did not
change Open3D, detector inputs or configuration to get a prettier result.

The CDR hypothesis was isolated using the same 1005 real recorded detector output
messages, through actual Humble publishers and raw DDS subscriptions. This measures
delivery of already serialized messages, not a live callback or sensor-to-display age:

```sh
# Inside the same Humble image, with the actual CLI output mounted under /evidence:
python3 /recipe/replay_display.py \
  --input /evidence/latest-after/doubleT_obstacle_rviz \
  --output /evidence/dds-roundtrip --mode roundtrip
python3 /recipe/replay_display.py \
  --input /evidence/latest-after/doubleT_obstacle_rviz \
  --output /evidence/dds-raw --mode raw
```

The production change uses the actual Humble `Publisher.publish(bytes)` API and
still creates the identical CDR payload using the existing shared serializer.
Both DDS runs retain received output bags and canonical CDR comparisons for all
five topics, including status and its acquisition timestamps. No subscriber-side
value or detector decision is replaced by an expected value.

Required resources for remaining acceptance: a Linux/amd64 host that can execute
the pinned Open3D wheel (preferably the target i7), Ubuntu 22.04/Humble or Docker,
original bag read access, and an X11/Wayland graphical session for RViz. Measure
full callback/overload at rate 1.0 there, then run the complete freshness recipe and
record RViz. No 10 Hz, i7 latency, complete-input live DDS or recovery claim is made.
