# Sensor specification evidence — 2026-09-16

User supplied [Hesai downloads / Pandar128](https://www.hesaitech.com/downloads/#pandar128).
The linked manual is [Pandar128E3X v4p5, 128-en-251110](https://www.hesaitech.com/wp-content/uploads/2025/11/Pandar128E3X_v4p5_User_Manual_128-en-251110.pdf).
This identifies a lidar family, not the exact device/firmware/driver in our bags.

Manufacturer statements (PDF page numbers, one-based):

- Pages 15–18: rotation axis Z, azimuth zero along Y, nonuniform vertical channel angles;
  accurate angular corrections are device-specific.
- Page 19: 10/20 Hz; single and dual return modes; range specifications depend on reflectivity.
- Page 51: dual returns occupy adjacent blocks from the same firing; some modes may
  repeat identical returns. They are not independent temporal confirmations.
- Pages 119–120: point firing times require packet/block timing plus channel offsets.

Project implications:

- Do not flip `sensor_profile_verified` merely from this manual. Obtain exact model,
  calibration file, installed transform and driver configuration/version.
- Sensor housing axes do not establish how PointCloud2 was transformed by its producer.
  Current x=-raw_y, y=raw_x, z=raw_z is still an unverified installation assumption.
- Do not infer return mode from PointCloud2 array capacity or ring count alone.
- Packet timestamps described here do not prove the meaning of our FLOAT64 `timestamp`
  field. Confirm driver conversion and whether deskew/aggregation was already performed.
- Ask for scan assembly mode, return mode, firing/angle correction files, sensor-to-vehicle
  transform, hardware filtering settings and clock source. Q&A may resolve these questions.
- No advertised hardware range is a measured obstacle-detection range on these recordings.

No timing, deskew, sensor transform or detector threshold was changed based on the manual.

## Preserved fields and provenance

`decode_cloud` returns `(points, normalized_times, invalid_count, duration, attributes)`.
The first four values retain their existing meanings. `Scan.attributes` carries the same
`PointAttributes` object to replay consumers; the ROS adapter passes it to `Detector.process`.

- `values` retains available `intensity`, `ring` and `raw_time` columns in their original
  dtype, selected by the same XYZ-valid mask as the decoded coordinates. Absent fields stay
  absent. Invalid attribute values are preserved and flagged, not used to remove XYZ points.
- `source_indices` names flattened row-major slots in the original PointCloud2.
  `xyz_valid_mask` covers every original slot. Neither identifies a physical emission.
- Diagnostic `decoded_*`, `range_*`, `geometry_voxel_*` and `cluster_*` attribute arrays follow
  the exact existing row selections. Geometry and cluster representatives are not reassociated
  with a nearest raw point. The motion stage must preserve row order and count.
- The result's `sensor_attributes` summary reports availability and validity counts.
  `profile_confirmed=false`, `firing_identity=unknown`, `return_multiplicity=unknown` and
  unverified intensity calibration are deliberate, not permission to infer missing provenance.

Supported optional fields must be scalar PointCloud2 numeric fields. A present unsupported
layout is rejected explicitly. Real replay establishes behavior only for its observed layouts.
Field values are not inputs to geometry, association, confirmation or hazard decisions.
There is no intensity threshold, `(ring, raw_time)` deduplication or new deskew behavior.

This compact transfer is based on the presence separation in #11 and the numerical correction
in #13; it does not independently resolve the latter. The source change is `e48304c`, relative
to `f2dd7c4`. Full-panel recipes are `configs/attributes-before-20260923.json` and
`configs/attributes-after-20260923.json`: seed 20260915, serial ICP, the unchanged nine-recording
panel, and zero differences in every prior non-timing output. See
[decode and resource measurements](DECODE_AND_LATENCY.md) before deciding to enable it.
