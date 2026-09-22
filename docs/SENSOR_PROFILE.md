# Sensor field evidence and preservation — 2026-09-16, updated 2026-09-22

This file collects what is known about the sensor fields, what the decoder now preserves,
and where the boundary runs between "measurable from the recordings" and "must be requested
from the organizer". It does not install a calibration, a firing model or a mounting
transform.

## 1. Manual evidence

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

No advertised hardware range is a measured obstacle-detection range on these recordings.
Manual statements are cited as manufacturer claims, not verified against our devices.

## 2. Field preservation — issue #5, 2026-09-22

Before this change the decoder read only `x, y, z` and the time field; it discarded `intensity`
and `ring` at the input, so they were absent from the decoded scan and no decoder consumer
(detector, diagnostics, viewer) could see them. That loss is fixed. One path already read its
own copy: `sustech_import` parsed `intensity` out of the raw PointCloud2 payload into PCD files
independently of the decoder, so the loss was in the decoder, not in every tool. The decoder
now returns a five-tuple `(points, normalized_times, invalid_count, duration_s, attributes)`;
the first four values are bit-identical to the previous four-tuple on the same real messages.

Measured on 56 real messages (11,760,266 point observations, evidence
`results/issue-5-sensor-attributes-20260922.json`): no frame changed, maximum coordinate and
time difference 0.0, 0 attribute failures, and `intensity` (float32), `ring` (uint16) and
`raw_time` (float64, source field `timestamp`) are present in every frame with 0 invalid
values. Diagnostics keep every existing array bit-identical and add the decoded attribute
arrays with the same stage prefixes; the 12-frame detector replay (3598 object observations)
shows 0 discrete or structural differences and bit-identical legacy diagnostic arrays. The
browser viewer takes its `ring`, `intensity` and raw-time values from the `PointAttributes`
returned by `decode_cloud` for the displayed frame, not from `Scan.attributes` and not by
reparsing the payload. The detector panel is all-`no_obstacle_observed`, so it establishes
preservation and non-interference, not recall or hazard behaviour. Deskew stayed off, no
threshold was introduced, and no geometric decision changed.

Full per-recording tables, `decode_cloud` cost (p50 6.58 → 7.87 ms, p95 17.47 → 20.45 ms
on that panel), reproduction commands and limitations are in
[DECODE_AND_LATENCY.md](DECODE_AND_LATENCY.md), section "Sensor attributes preserved through
decode — 2026-09-22". They are not restated here. Ad-hoc out-of-contract probes run by earlier
workers are excluded from these numbers; no test suite or test file was introduced.

## 3. What the source fields are — and are not

The recorded layout is one row, little-endian, `point_step` 26, fields
`x/y/z` FLOAT32, `intensity` FLOAT32, `ring` UINT16, `timestamp` FLOAT64 at offset 18;
`is_dense` false. Two widths appear in the corpus (921,600 and 307,200 slots).

- **`intensity` is not a material identifier.** It is a raw, per-return strength reading.
  With no known-material reference at known ranges it cannot be calibrated to a material:
  the same surface changes its reading with range, incidence angle, reflectivity and receiver
  settings. Agreement of the channel on one patch between matched repeated passes would be
  diagnostic only — a driver gain or a different viewing geometry produces the same agreement,
  and it would not distinguish "material A vs material B" from "same coating, other geometry".
  No such matched-pass measurement has been made in this change, and no matched repeated passes
  have been identified here.
- **`ring` plus time does not identify a firing.** `ring` is a channel index (0…127 here); it
  is not a firing identity. The exact firing order, scan period and per-channel timing are not
  established from the available PointCloud2 alone: the manual (pages 119–120) requires
  packet/block timing plus channel offsets for point firing times. Header-to-header intervals
  and the span and resolution of the time column within one scan are measurable, but they are
  not sufficient to prove a firing model. One timestamp is shared by roughly 190 points (issue
  #5 evidence), so time alone is not a single firing either.
- **Array capacity and duration do not prove different sensors.** Slot counts (921,600 vs
  307,200) and point-time spans (68 ms vs 32 ms) differ between recordings, but that is
  consistent with one sensor in different modes or driver configurations as well as with
  different devices. PointCloud2 shape cannot decide this.
- **Multiple returns are not new temporal hits.** A second or later reflection of one firing
  arrives beside the first in the same scan; it is not a new measurement of a later instant.
  Existing support counts are voxel representatives from a single scan and confirmation uses
  distinct scans and timestamps, so voxel support is not a count of independent firings.
  Attributes report `return_multiplicity: "unknown"` and never group, deduplicate or average.

## 4. Time: raw preserved vs normalized

Two times are now carried side by side:

- **Raw time** is the source column itself, preserved with its original dtype and values under
  `PointAttributes.values["raw_time"]` (source field name recorded in `time_field`). On these
  bags it is absolute seconds on the header epoch (≈ 9.47×10⁸ s) and its values are close to
  the header epoch. That similarity is evidence the field uses the same epoch, not proof that
  the two clocks are synchronized or that the value is an emission instant. The `seconds` unit
  label applies only when the source time field is named `timestamp`; other time fields keep
  their values with `unknown` units.
- **Normalized time** is the existing per-scan `[0, 1]` point-time array the detector already
  consumes. It is unchanged, still computed by the decoder, and still the only time input to
  detection: attributes never reach the detection, support or temporal-confirmation code. Its
  physical units and the meaning of one unit remain unknown; the normalization is a rescaling
  of whatever the source field is, not a measured duration.

Emission semantics (scan period, firing order, per-channel offset, whether the driver already
deskewed or aggregated) remain unverified. Deskew remains disabled.

## 5. API and diagnostic contract

`io.PointAttributes` carries the preserved fields without changing any decision:

- `values: dict[str, np.ndarray]` — only the optional source columns that exist
  (`intensity`, `ring`, `raw_time`), original dtype and values, aligned to the decoded
  XYZ-valid rows; invalid values are preserved and flagged, never dropped and never allowed to
  alter the XYZ validity mask.
- `source_indices: np.ndarray` — maps each decoded row to its original flattened row-major
  `PointCloud2` slot.
- `xyz_valid_mask: np.ndarray` — boolean over every original slot (the full mask is stored once
  by the caller, not per stage).
- `time_field: str | None` — name of the source time column, or `None`.
- `summary()` — JSON-safe field availability, dtypes, units (`intensity` =
  `unverified_raw_counts`, `ring` = `ring_index`, `raw_time` = `seconds` only when the source
  time field is `timestamp`, otherwise `unknown`), valid/invalid counts, plus
  `profile_confirmed: false`, `firing_identity: "unknown"`,
  `return_multiplicity: "unknown"`, `intensity_calibration: "unverified"`.
- `arrays(indices=None)` — NPZ-ready dict aligned to the selected decoded rows: `source_indices`,
  each value key, and a `<key>_valid` boolean (finite, and for `ring` also nonnegative integral).

`Scan.attributes` is supplied by `iter_bag` for every decoded scan. `Detector.process` accepts
the attributes as an optional keyword-only `point_attributes`; it validates index/shape
alignment only, reports `sensor_attributes` in the result whenever supplied, and changes no
other signature behaviour. `capture_diagnostics` NPZ files keep every existing array unchanged
and add `decoded_xyz_valid_mask`, `decoded_source_indices`, `decoded_<value>`,
`decoded_<value>_valid`, plus the same attribute arrays under the `range_`, `geometry_voxel_`
and `cluster_` prefixes, each aligned to the exact rows that stage selected. The
`sensor_attributes` summary carries availability and provenance; it is not a claim of
independent firing support.

## 6. Research boundary: measurable vs external

Measurable from the recordings alone:

- presence, dtype and range of each field; invalid-return fraction per frame;
- raw-time span and resolution within a frame, number of distinct values, and the similarity of
  the field to the header epoch (evidence for the same epoch, not proof of synchronized clocks);
- the `ring` value set; slot/column counts per recording; header-to-header intervals.

Measurable in principle, not measured in this change:

- repeatability of `intensity` for one patch across matched repeated passes. No matched repeated
  passes have been identified here, so this remains a capability, not a result.

Not established from the recordings (requires the organizer's specification):

- a calibrated material mapping; the exact firing order, per-channel timing and scan period;
  whether the producer already deskewed, aggregated or filtered; the sensor-to-vehicle transform
  and mounting reference.

Neither the header epoch nor a per-frame time span can promote the time field into a firing
time, and intensity consistency across passes without known materials cannot promote the
channel into a calibrated material identifier. Both stay in the health caveats.

## 7. Request to the organizer (one short list)

1. Exact sensor model and firmware version; driver version and configuration.
2. Angle/firing correction files; firing order and scan period.
3. Return mode and scan assembly mode; whether the driver already deskews/aggregates or applies
   hardware filtering before publishing.
4. Time-field semantics: units, epoch, emission vs reception, per-point vs per-packet, clock
   source; whether the FLOAT64 `timestamp` is the driver's conversion.
5. Sensor-to-vehicle mounting transform and the mounting height reference.

Not requested because we can measure them ourselves: slot/column counts, within-frame time span,
`ring` set, invalid-return fraction, and the absence of TF/IMU/odometry topics in the bags.
Intensity repeatability across matched passes would also be self-measurable, but no matched
passes have been identified yet.

## 8. What was not changed

No timing, deskew, sensor transform, intensity threshold or detector decision was changed on
the basis of the manual, the preserved fields or any presumed cross-pass consistency.
`intensity` and `ring` are reported and inspectable; they are never an input to detection,
support or temporal confirmation, and no return identity is guessed.
