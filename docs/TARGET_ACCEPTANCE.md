# Target deployment acceptance record

This is a **manual field-like deployment check**, not a detector-quality
benchmark and not permission to call a route clear.  It closes the currently
unmeasured part of issues #25 and #26: the exact image, target machine, ROS 2
Humble runtime, watchdog behaviour, RViz output and recorded output contract.
Run it on the intended Ubuntu 22.04 / Intel (`linux/amd64`) machine with Docker
and a graphical session.  Keep the resulting directory with the release
candidate; do not replace it with a screenshot or a claim from another host.

## Inputs and evidence directory

Mount the complete, immutable input corpus read-only.  Record its path, the
output of `ros2 bag info`, the source PointCloud2 topic and the source message
count before starting.  A partial extraction is useful for smoke work but does
not establish full-corpus acceptance.

```sh
export TG_TAG=tunnel-guard:release-22
export TG_INPUT=/absolute/path/to/full-corpus-or-qualified-bag
export TG_EVIDENCE="$PWD/build/target-acceptance-$(date +%Y%m%dT%H%M%S)"
mkdir -p "$TG_EVIDENCE"

docker build --platform linux/amd64 -t "$TG_TAG" . |& tee "$TG_EVIDENCE/build.log"
docker image inspect "$TG_TAG" > "$TG_EVIDENCE/image-inspect.json"
uname -a | tee "$TG_EVIDENCE/uname.txt"
lscpu | tee "$TG_EVIDENCE/lscpu.txt"
docker version |& tee "$TG_EVIDENCE/docker-version.txt"
```

If a GPU is in scope, save `nvidia-smi` too.  Record the repository commit and
the detector configuration file copied into the image.  These facts are needed
to compare runs; they do not prove detection range, recall, or latency.

## Launch, record and replay

Use the actual PointCloud2 topic reported by the selected bag in place of
`/lidar_points`.  `TG_INPUT` may be one bag directory for a delivery smoke
check or a parent directory only when the replay command enumerates the
qualified corpus explicitly.

```sh
docker run --platform linux/amd64 --rm -d --name tunnel-guard-acceptance \
  -v "$TG_INPUT:/input:ro" -v "$TG_EVIDENCE:/evidence" "$TG_TAG" \
  bash -lc 'source /opt/ros/humble/setup.bash && \
    ros2 launch /opt/tunnel-guard/launch/tunnel_guard.launch.py \
    input_topic:=/lidar_points input_timeout_s:=3.0'

docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag info /input/<bag>' \
  |& tee "$TG_EVIDENCE/input-bag-info.txt"

docker exec -d tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag record -o /evidence/output \
    /perception/points_display /perception/debug_markers /perception/status \
    /perception/attention_required /perception/nearest_obstacle_m \
    > /evidence/record.log 2>&1 & echo $! > /evidence/record.pid'

docker exec tunnel-guard-acceptance bash -lc \
  'source /opt/ros/humble/setup.bash && ros2 bag play /input/<bag> --rate 1.0 \
    --wait-for-all-acked 30000' |& tee "$TG_EVIDENCE/replay.log"
docker logs tunnel-guard-acceptance |& tee "$TG_EVIDENCE/node.log"
```

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

The output bag must declare exactly the five `/perception/...` topics and types
listed in [ROS2.md](ROS2.md).  Preserve the bag itself, its `metadata.yaml`,
the node/replay logs and the command arguments.  A bag recorder can prove what
was published, not that DDS delivered every source scan: the adapter's
`received`, `processed`, `invalid`, `duplicate` and `out_of_order` values in
`/perception/status` must be reviewed alongside the source message count.

## Observations to capture

1. Run an RViz2 session on the target using
   `/opt/tunnel-guard/rviz/tunnel_guard.rviz`; retain a short screen recording
   covering cloud, markers, status and an `unknown` transition.  Set `rviz:=true`
   only with the normal authenticated display/Xauthority configuration.
2. While the node remains up, stop input for longer than `input_timeout_s`.
   Save the status message showing `unknown`, the empty display cloud and
   `DELETEALL` marker array.  Restart replay only with monotonically later
   source timestamps; otherwise restart the node, as the contract requires.
3. Extract `callback_processing_s` from recorded status rows and report its
   distribution separately for the qualified run.  `result_age_s` is expected
   to remain null until source and host clocks are demonstrated comparable.
   Neither number establishes sensor-to-warning latency.
4. For a full-corpus run, write down selected bags, input messages, output
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
