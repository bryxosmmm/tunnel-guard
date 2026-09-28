# Frozen detector delivery and demo

## Source boundary

This document records PR #43's delivery-only work, based on frozen
**7689b8057a3e35ab39ce0e671c37d7efc9cfd1d6** (PR #42 merge).
The inference source, native source and `configs/detector.json` at that commit match
PR #35 head **8539a3e1a8764a49b3647558c895a611eebf6a13**; the only change under
`tunnel_guard/` is `sustech.py`, which exports annotations. The earlier verified
baseline **ec311a0bc78363791b5fe9bfaf1ea759bbf2b16a** is retained as historical
comparison evidence. PR #43 changed no detector or detector configuration file.
The later integration also contains PR #44's native/CPU optimizations and review
corrections; this historical report does not qualify those changes. Its current
evidence is `results/review-integration-20260928.json`.

`configs/detector.json` SHA256 at the frozen `main` commit:
`f1b0d0ea760c8b3a4e7171b6e9feb37c7e0ac9a38a5c310517b3fa41bcab9efa`.
The delivery follows main's five `/perception/...` topics and RViz configuration
from PR #38. Do not combine it with the
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
uv run python -m tools.render_delivery_demo --run build/delivery/demo \
  --annotations annotations/doubleT-obstacle-person-crossing.json --output build/delivery/demo.html
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

`tools/compare_delivery.py BEFORE AFTER --output REPORT` now exits nonzero for
missing/extra acquisitions, missing topic messages, or changed non-runtime output.
It compares serialized status semantics as well as the four other CDR topics;
a matching common prefix is not a complete equivalent replay.

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
  --annotations annotations/doubleT-obstacle-person-crossing.json --live build/delivery/ros/complete \
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

After rebasing on `main`, a second full 201-scan CLI replay retained every detector
decision, acquisition timestamp and all 804 non-status display CDR payloads from
the earlier optimized replay. Its manifest, raw stage timings and comparison are
in the archive. The original before/after timing pair remains the controlled
performance comparison; the later `main` run is an integration check.

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
