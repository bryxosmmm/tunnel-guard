# Recorded 3D server demonstration (2026-09-27)

The Ubuntu 22.04 x86-64 server holds the two original dataset archives under
`~/tunnel-guard-deploy/dataset/` and their extracted contents under
`~/tunnel-guard-deploy/data/`. The extracted data contains all six
`for_hackathon` bags and the `new_data` bag (seven `metadata.yaml` files).
The archives are retained; their SHA-256 hashes are recorded in
`~/tunnel-guard-deploy/tunnel-guard-SHA256SUMS`. The transfer finalizer records
the verified inventory in `logs/dataset-complete.json`.

The native Docker detector processed two complete, unlabelled recorded bags
without skipping frames. Results, source/config identities, and per-frame
timings are in `~/tunnel-guard-deploy/results/demo-full-v2/`; the run log is
`~/tunnel-guard-deploy/logs/full-demo-offline.log`.

| Bag | Frames | Candidate | Unresolved | Confirmed intersection | Processing p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `roundT_doubleT` | 252 | 1 | 244 | 6 | 475 / 592 ms |
| `doubleT_obstacle` | 201 | 2 | 39 | 160 | 613 / 700 ms |

One frame in `roundT_doubleT` has status `no_obstacle_observed`. These status
counts are frames, not independently validated obstacle events, and the bags
have no exhaustive labels. They establish neither detection accuracy nor
real-time performance. A separate ten-scan ROS 2 replay at 0.1x retained all
original acquisition timestamps; 0.25x dropped input under load. See
[ROS2.md](ROS2.md) for that bounded integration evidence.

Two loopback-only viewer containers run from `tunnel-guard:demo` on remote
ports 8766 (`roundT_doubleT`) and 8768 (`doubleT_obstacle`). To show both on a
workstation with SSH access, keep this tunnel open, replacing the endpoint:

```sh
ssh -N -L 127.0.0.1:8767:127.0.0.1:8766 \
  -L 127.0.0.1:8772:127.0.0.1:8768 USER@SERVER
```

Open `http://127.0.0.1:8767/3d` for the first complete recording, then
`http://127.0.0.1:8772/3d?frame=25&focus=160` for a selected object at 55.8 m
in the second. In that frame, the selected object is reported as intersecting;
the display also contains nearer unresolved observations. The full run did not
save diagnostic point arrays, so its selected box has no exact-support point
overlay. The retained 30-frame diagnostic preview can show those 74 support
points separately. The viewer samples the display cloud, while the detector
uses the full decoded cloud. It is replay in the sensor's processing frame,
not a reconstructed train trajectory or a field-safety result.
