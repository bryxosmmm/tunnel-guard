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
  This does **not** establish issue #5's stronger requirement that multiple returns cannot
  increase spatial support: different returns can occupy different voxels. See section 9.

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

Blocking inputs for issue #5's pulse-independence criterion:

1. **Recording-to-profile mapping:** a stable device/profile identifier for each recording,
   exact model/firmware, driver repository revision and effective configuration, including
   return mode and scan assembly. A topic or `frame_id` string is not device identification.
2. **A short raw-packet interval synchronized with its published PointCloud2**, plus the
   device's correction file, or equivalent provenance-preserving packet/block/channel/return
   identifiers. We need to match published slots to known emissions and return indices,
   not merely guess from coincident values.
3. **The timestamp and processing contract:** how packet/block time and channel offsets
   produce `timestamp`; whether it denotes emission/reception; clock/epoch; and any prior
   deskew, aggregation, filtering or reordering. Specify how dual returns share time and
   how distinct emissions remain distinguishable after publication.

These inputs let us validate a firing key against packet provenance on matched real data
before using it to limit support. If publication has irreversibly merged firing identities,
the producer must preserve them; more statistics on the same six fields cannot restore them.

Material-labelled observations at known ranges/incidence are needed **only if material
discrimination is required**. None are available here: intensity is retained, but material
discrimination is explicitly refused. Cross-pass consistency would not by itself establish
material labels. Vehicle extrinsics remain a separate deployment requirement, not a
substitute for firing provenance.

Do not request what the recordings already provide: field layout/dtypes, valid-return
fractions, observed ring values, timestamp span/resolution, frame intervals or topic inventory.
Those measurements do not establish emission semantics or independent firings.

## 8. What was not changed

No timing, deskew, sensor transform, intensity threshold or detector decision was changed on
the basis of the manual, the preserved fields or any presumed cross-pass consistency.
`intensity` and `ring` are reported and inspectable; they are never an input to detection,
support or temporal confirmation, and no return identity is guessed.

## 9. Return ambiguity audit — issue #5 remains blocked

Recipe `configs/issue-5-return-ambiguity-20260922.json`; evidence
[`results/issue-5-return-ambiguity-20260922.json`](../results/issue-5-return-ambiguity-20260922.json).
Executed the real saved-scan inspector over **85 captured scans**, containing **17,191,300
decoded valid returns**, including all three captured geometry failures. This is the saved
diagnostic panel, not a new full-2,641-frame return-identity census.

Exact `(ring, raw_time)` groups span distinct 5 cm spatial cells in **176,224 groups**.
The largest observed group has two members; this is compatible with, but does not prove,
dual-return operation. Within non-noise cluster components, **5,443** such groups span
multiple cells (including rejected components). **348 emitted object observations** contain
at least one spanning group; all 348 are adjacent and unconfirmed. The complete per-scan
records are retained, compressed, with hashes. No physical firing or true return count is
assigned to these coincidences, and no causal alarm change is claimed.

The code distinguishes two mechanisms:

- Temporal `hits` admit at most one observation per tracked frame. The preceding full
  regression panel also measured zero repeated/future evidence timestamps. Multiple
  points in one scan do not add separate temporal hits.
- `support_voxels`, `uncertain_voxels`, density support and accumulated voxel cardinality
  remain **spatial** quantities. They do not group physical emissions. Accumulated support
  enters the confirmation gate (`Detector._associate`); spatial support also affects
  admission, path relation and immediate confirmation. Distinct returns from one pulse
  are therefore not guaranteed to contribute only once.

**Acceptance status:** preservation is verified, but criterion 3 (pulse-independent
support) is not established. The current data do not identify which observed coincident
pairs are the same emission. Consequently the audit cannot measure their causal influence
on confirmation or prove invariance to true multiple returns.

Safe boundary: preserve observations and explicit unknown-profile status; leave deskew and
intensity-based decisions off. Do not deduplicate approximate keys, invent independent-hit
counts, or relabel existing confirmations as pulse-independent. Suppressing every
unknown-profile confirmation would change recall and operator semantics and is not a
silent fix. The issue remains open, blocked on the three provenance inputs in section 7.

```sh
.venv-iteration/bin/python -m tunnel_guard.inspect_bag \
  --experiment configs/issue-5-return-ambiguity-20260922.json
```

Use a new output path for another run. No generated clouds or automated tests are part
of this investigation; detector decisions and thresholds are unchanged.

Local storage note: the five issue-5 replay runs' large JSONL outputs were subsequently
archived losslessly as `.jsonl.gz`. Each archive was decompressed and SHA-256 checked
before removing its uncompressed original; NPZ diagnostics were left intact. The audit
above ran before archival. Its current CLI expects `.jsonl`: restore required inputs
with `gunzip path/to/bag.jsonl.gz` before repeating it, with adequate disk space.
The local `build/issue-5-lossless-storage-reclamation.jsonl` journal lists all 31 archives,
original hashes and restoration commands. Archival reclaimed 4,133,430,529 bytes;
it is not a new detector evaluation.
