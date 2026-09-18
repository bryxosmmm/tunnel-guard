# Complete extended recording available — 2026-09-18

After the user freed disk space, the complete archive was extracted directly through ZIP → Zstandard → tar into `data/extended_full/new_data`. No intermediate 17 GB compressed copy was written. The extraction required at least 20 GiB reserve before starting and before each member write.

- 221 original DB3 files plus original `metadata.yaml`.
- 90,121,539,170 bytes extracted; about 37 GiB remained free afterward.
- Each file has a SHA256 digest in `data/extended_full/extraction.json`.
- All three previously extracted segment hashes match the full extraction.
- Original ZIP and the earlier partial views remain intact.

`tunnel_guard.inspect_recording_headers` inspected all **11,271 PointCloud2 headers**, in the source metadata's file order and database timestamp order. It checks per-file counts/starting timestamps against metadata and verifies its CDR-header reading against the standard ROS deserializer on the first complete message of every segment. Incremental SQLite blob reads avoid loading entire point arrays for every header.

Results: **0 duplicate acquisitions, 0 backward acquisition steps, 23 gaps greater than 0.5 seconds**. The separate acquisition and recording clock domains remain uncalibrated. This header audit does not validate all coordinates, rail identity, obstacle labels, or sensor extrinsics.

## Existing failure localized

A replay of actual segment 200/frame 48 geometry traced the stop to the rail-pair ambiguity condition at x=15 m. Leading histogram pairs have centers −0.45 m and −0.75 m, separation estimates 1.41 m, and scores 1.43885 and 1.25628. Only one earlier anchor has been accepted. This triggers the ambiguity stop and leaves insufficient paired support.

Their center separation is on the existing 0.30 m comparison boundary (floating arithmetic gives slightly more than 0.30). This boundary sensitivity must be reviewed together with the ambiguity semantics; the trace alone does not prove which physical pair is correct. No threshold or detection behavior was changed in this preparation step.

## Commands and next run

Executed:

```sh
.venv-iteration/bin/python -m tunnel_guard.inspect_recording_headers data/extended_subset/new_data_200 --output build/extended-inventory/header-prefix-200.json
.venv-iteration/bin/python -m tunnel_guard.inspect_recording_headers data/extended_full/new_data --output build/extended-inventory/full-recording-headers.json
git diff --check
```

Prepared, **not executed**:

```sh
.venv-iteration/bin/python -m tunnel_guard.run --experiment configs/extended-full.json
```

That recipe uses the original complete metadata so the tracker can continue through file boundaries and reset only according to actual measurement gaps/motion quality. It saves no large diagnostic point arrays by default. Complete detector inference still covers only the earlier 153-cloud sample; do not confuse the full header audit with full-corpus inference.

Evidence: `results/extended-full-ingest-20260918.json`. No automated tests were created or run.
