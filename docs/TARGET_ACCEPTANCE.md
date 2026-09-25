# Target deployment acceptance record

This is a **manual field-like deployment check**, not a detector-quality
benchmark and not permission to call a route clear.  It closes the currently
unmeasured part of issues #25 and #26: the exact image, target machine, ROS 2
Humble runtime, watchdog behaviour, RViz output and recorded output contract.
Run it on the intended Ubuntu 22.04 / Intel (`linux/amd64`) machine with Docker
and a graphical session.  Keep the resulting directory with the release
candidate; do not replace it with a screenshot or a claim from another host.

## Inputs and evidence directory

Mount each selected, immutable input bag read-only. Record its path, the output
of `ros2 bag info`, the source PointCloud2 topic and the source message count
before starting. A partial extraction is useful for smoke work but does not
establish full-corpus acceptance. The example below uses `doubleT_obstacle`;
select its actual topic from the bag metadata, not the node's `/lidar_points`
default. Repeat the replay for every qualified bag when assessing a full corpus.

```sh
set -o pipefail
export TG_TAG=tunnel-guard:release-22
export TG_BAG=/absolute/path/to/doubleT_obstacle
export TG_TOPIC=/sensing/lidar/hesai128/pointcloud
export TG_TIMEOUT=3.0
export TG_XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
export TG_EVIDENCE="$PWD/build/target-acceptance-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$TG_EVIDENCE"

git rev-parse HEAD | tee "$TG_EVIDENCE/commit.txt"
sha256sum configs/detector.json | tee "$TG_EVIDENCE/config-sha256.txt"
sha256sum "$TG_BAG"/metadata.yaml "$TG_BAG"/*.db3 \
  | tee "$TG_EVIDENCE/input-sha256.txt"
docker build --platform linux/amd64 -t "$TG_TAG" . |& tee "$TG_EVIDENCE/build.log"
docker image inspect "$TG_TAG" > "$TG_EVIDENCE/image-inspect.json"
uname -a | tee "$TG_EVIDENCE/uname.txt"
lscpu | tee "$TG_EVIDENCE/lscpu.txt"
docker version |& tee "$TG_EVIDENCE/docker-version.txt"
```

The image ID in `image-inspect.json`
identifies a local build; a registry digest exists only if the image was pushed
or pulled from a registry. If a GPU is in scope, save `nvidia-smi` too. These
facts are needed to compare runs; they do not prove detection range, recall, or
latency.

## Launch, record and replay

Run the detector, recorder, player and RViz in the same container. The example
uses an authenticated X11/XWayland session on the Ubuntu host; check `DISPLAY`
and `TG_XAUTH` there before starting. If the target uses another display setup,
record the adapted mounts and command. Save the RViz log and a screen recording.
A headless import or exported bag cannot establish GUI acceptance.

```sh
docker run --platform linux/amd64 --rm -d --network host \
  --name tunnel-guard-acceptance \
  -e TG_TOPIC -e TG_TIMEOUT -e DISPLAY \
  -e XAUTHORITY=/tmp/tunnel-guard.Xauthority \
  -v "$TG_XAUTH:/tmp/tunnel-guard.Xauthority:ro" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  -v "$TG_BAG:/input/bag:ro" -v "$TG_EVIDENCE:/evidence" "$TG_TAG" \
  bash -lc 'source /opt/ros/humble/setup.bash && \
    exec ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
    input_topic:="$TG_TOPIC" input_timeout_s:="$TG_TIMEOUT"'

docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag info /input/bag' \
  |& tee "$TG_EVIDENCE/input-bag-info.txt"

docker exec -d tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && \
    echo $$ > /evidence/record.pid && \
    exec ros2 bag record -o /evidence/output \
    /perception/points_display /perception/debug_markers /perception/status \
    /perception/attention_required /perception/nearest_obstacle_m \
    > /evidence/record.log 2>&1'

docker exec -d tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && \
    exec rviz2 -d /opt/tunnel-guard/rviz/tunnel_guard.rviz \
    > /evidence/rviz.log 2>&1'

# Wait until the following queries show an RViz subscriber on both topics.
docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic info -v /perception/points_display' \
  |& tee "$TG_EVIDENCE/points-qos.txt"
docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic info -v /perception/debug_markers' \
  |& tee "$TG_EVIDENCE/markers-qos.txt"

docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /input/bag --rate 1.0 \
    --wait-for-all-acked 30000' |& tee "$TG_EVIDENCE/replay.log"
docker logs tunnel-guard-acceptance |& tee "$TG_EVIDENCE/node.log"
```

While playback is active, run this in a second terminal and retain the output:

```sh
docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 topic info -v "$TG_TOPIC"' \
  |& tee "$TG_EVIDENCE/input-qos.txt"
```

Keep `/perception/status` and `/perception/attention_required` visible in a
terminal next to RViz for the freshness screen recording; the text status is
authoritative when an object or marker disappears.

Stop the recorder only after the final messages arrive, then inspect the output
bag and retain its metadata:

```sh
docker exec tunnel-guard-acceptance bash -lc \
  'kill -INT "$(cat /evidence/record.pid)"'
docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag info /evidence/output' \
  |& tee "$TG_EVIDENCE/output-bag-info.txt"
docker stop tunnel-guard-acceptance
```

Before replay, inspect `ros2 topic info -v` for the input and all five outputs.
RViz must appear as a subscriber on both `/perception/points_display` and
`/perception/debug_markers`, with compatible QoS; save this output while RViz
is open. The supplied RViz config uses fixed frame `tunnel_guard_local`.
No identity TF is needed or justified. The source input publisher exists only
while playback runs, so capture input publisher/subscriber QoS during replay.

The output bag must declare exactly the five `/perception/...` topics and types
listed in [ROS2.md](ROS2.md).  Preserve the bag itself, its `metadata.yaml`,
the node/replay logs and the command arguments.  A bag recorder can prove what
was published, not that DDS delivered every source scan: the adapter's
`received`, `processed`, `invalid`, `duplicate` and `out_of_order` values in
`/perception/status` must be reviewed alongside the source message count.

## Observations to capture

1. Run an RViz2 session on the target using
   `/opt/tunnel-guard/rviz/tunnel_guard.rviz`; retain a screen recording showing
   measured cloud, supported reference contour, observed-support object boxes
   and the status terminal. The reference contour is not a vehicle-specific
   swept envelope. Show a real confirmed object if the replay produces one;
   retain unresolved and negative intervals rather than selecting only one
   successful frame.
2. After fresh scans establish a scene, repeat the *same* real PointCloud2
   message (unchanged acquisition timestamp) for longer than
   `input_timeout_s`, then stop input entirely. Retain the published status
   `unknown`, `attention_required=true`, NaN nearest distance, empty display
   cloud and `DELETEALL` marker array, plus RViz showing that stale objects
   disappeared. A stopped player alone does not exercise duplicate freshness.
3. Resume with strictly newer acquisition timestamps and show the scene
   recovering. Repeat the duplicate/silence phase with `use_sim_time=true` and
   frozen `/clock`; the watchdog must still expire. The retained actual-DDS
   delivery recipe in `results/issue33-delivery-20260925.zip` can drive these
   phases on real recorded PointCloud2 messages, but its prior emulated run is
   not this target GUI acceptance. Record the phase commands, input/output
   messages and any failed attempt. Do not replay an older bag epoch into the
   same node: restart it if timestamps must move backwards.
4. Extract `callback_processing_s` from recorded status rows and report its
   distribution separately for the qualified run.  `result_age_s` is expected
   to remain null until source and host clocks are demonstrated comparable.
   Neither number establishes sensor-to-warning latency.
5. For a full-corpus run, write down selected bags, input messages, output
   status counts, malformed/duplicate/out-of-order counts, host facts, image
   ID and observed failure modes.  Keep unavailable and unknown frames in the
   record; do not filter them out after seeing results.

## Decision boundary

Passing this record means only that this exact build exercised its published ROS
contract on the documented target and that its visible failure behaviour was
captured.  It does **not** validate collision avoidance, braking integration,
vehicle clearance, mounting calibration, field recall, nuisance-alarm rate,
end-to-end latency, or safe operating range.  Any such claim needs separately
qualified measurements, labels and operating policy.
