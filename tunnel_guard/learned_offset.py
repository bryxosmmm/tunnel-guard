"""Learn the bore-to-track offset from the section profile, and check it against the rails.

``bore_reference`` carries the offset measured near the train out to longer ranges, which only
holds while the cross-section stays the same. That assumption is what fails at a platform or
where the tunnel opens out. Here the offset is a function of the observed section instead,
learned from frames whose rails can be trusted.

No human annotation is involved: the rail estimator supplies the target wherever it is
trustworthy. Frames that ``center_stability`` marks as lock failures are excluded from
training, because their target is the quantity that is wrong; predicting on them afterwards is
a test on a regime the model never saw.

Recordings are held out whole. Splitting by frame would score near-duplicate neighbours
against each other and report a number that means nothing. The trivial predictor -- claiming
the track runs down the sensor axis -- is scored alongside, because most of this subset is
roughly aligned and a model can look strong merely by reproducing a zero it never learned.

scikit-learn is an optional extra (``pip install -e '.[learning]'``), imported where it is
used, following the same pattern as the published segmentation backends.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bore_reference import (FAR_RANGES_M, MIN_HEIGHT_M, NEAR_RANGES_M, SLAB_HALF_M,
                             centred_profile, symmetry_axis)
from .detector import load_config
from .geometry import TrackGeometry, voxel_representatives
from .io import iter_bag

PROFILE_BINS = 64


def build_profiles(bags: list[Path], detector: dict, cache: Path) -> dict:
    """Section profiles and rail centres at every probe range, cached because they are slow."""
    if cache.exists():
        return dict(np.load(cache, allow_pickle=True))
    columns: dict[str, list] = {k: [] for k in
                                ("bag", "frame", "range", "profile", "axis", "score", "width", "rail")}
    for index, bag in enumerate(bags):
        for scan in iter_bag(bag, detector):
            forward = ((scan.points[:, 0] >= detector["min_forward_m"])
                       & (scan.points[:, 0] <= detector["max_range_m"])
                       & (np.abs(scan.points[:, 1]) <= detector["context_half_width_m"]))
            geometry = TrackGeometry(voxel_representatives(scan.points[forward],
                                                           detector["geometry_voxel_m"]), detector)
            if not geometry.valid:
                continue
            height = scan.points[:, 2] - geometry.ground(scan.points)[0]
            anchors = geometry.rail_anchors
            for probe in NEAR_RANGES_M + FAR_RANGES_M:
                slab = ((np.abs(scan.points[:, 0] - probe) < SLAB_HALF_M)
                        & (height > MIN_HEIGHT_M) & (np.abs(scan.points[:, 1]) < 8))
                lateral = scan.points[slab, 1]
                axis, score = symmetry_axis(lateral)
                profile = None if axis is None else centred_profile(lateral, axis)
                if profile is None:
                    continue
                centred = lateral - axis
                anchored = len(anchors) >= 2 and anchors[0, 0] <= probe <= anchors[-1, 0]
                columns["bag"].append(index)
                columns["frame"].append(scan.index)
                columns["range"].append(probe)
                columns["profile"].append(profile.reshape(PROFILE_BINS, -1).mean(axis=1).astype(np.float32))
                columns["axis"].append(axis)
                columns["score"].append(score)
                columns["width"].append(float(np.quantile(centred, 0.95) - np.quantile(centred, 0.05)))
                columns["rail"].append(float(np.interp(probe, anchors[:, 0], anchors[:, 1]))
                                       if anchored else np.nan)
        print(f"  profiled {bag.name}", flush=True)
    data = {k: np.asarray(v) for k, v in columns.items()}
    data["bags"] = np.asarray([b.name for b in bags])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def excursion_frames(stability: Path, probe: str, limit: float) -> dict:
    """Frames whose rail centre is already known to be displaced, keyed by recording."""
    out: dict[str, dict[int, bool]] = {}
    for record in json.loads(stability.read_text())["recordings"]:
        name = Path(record["bag"]).name
        out[name] = {}
        for row in record["rows"]:
            probed = row["probes"].get(probe)
            if probed is not None:
                out[name][row["frame"]] = abs(probed["center_m"]) > limit
    return out


def assemble(data: dict, excursions: dict) -> dict:
    """One row per probed far range: features, target, and the two reference predictions."""
    names = list(data["bags"])
    index: dict[tuple[int, int], dict[float, int]] = {}
    for i in range(len(data["range"])):
        index.setdefault((int(data["bag"][i]), int(data["frame"][i])), {})[float(data["range"][i])] = i
    features, target, baseline, trivial, group, excursion = [], [], [], [], [], []
    for (bag, frame), probes in index.items():
        name = names[bag]
        flagged = excursions.get(name, {}).get(frame)
        if flagged is None:
            continue
        near = [probes[r] for r in sorted(NEAR_RANGES_M)
                if r in probes and np.isfinite(data["rail"][probes[r]])]
        if not near:
            continue
        offset = float(np.median([data["axis"][i] - data["rail"][i] for i in near]))
        near_profile = np.mean([data["profile"][i] for i in near], axis=0)
        near_width = float(np.mean([data["width"][i] for i in near]))
        near_score = float(np.min([data["score"][i] for i in near]))
        for probe, i in probes.items():
            if probe in NEAR_RANGES_M or not np.isfinite(data["rail"][i]):
                continue
            features.append(np.concatenate([
                data["profile"][i], near_profile,
                [probe, offset, data["score"][i], data["width"][i], near_width, near_score,
                 data["width"][i] - near_width, data["axis"][i]]]))
            target.append(float(data["axis"][i] - data["rail"][i]))
            baseline.append(offset)
            # Predicting target = axis is the claim that the track lies on the sensor axis.
            trivial.append(float(data["axis"][i]))
            group.append(name)
            excursion.append(bool(flagged))
    return {"x": np.asarray(features), "y": np.asarray(target),
            "baseline": np.asarray(baseline), "trivial": np.asarray(trivial),
            "group": np.asarray(group), "excursion": np.asarray(excursion)}


def leave_one_recording_out(rows: dict, seed: int) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    prediction = np.full(len(rows["y"]), np.nan)
    for held in sorted(set(rows["group"])):
        test = rows["group"] == held
        train = (~test) & (~rows["excursion"])
        if test.sum() < 20 or len(set(rows["group"][train])) < 2:
            continue
        model = HistGradientBoostingRegressor(max_depth=6, max_iter=500, learning_rate=0.05,
                                              l2_regularization=1.0, random_state=seed)
        model.fit(rows["x"][train], rows["y"][train])
        prediction[test] = model.predict(rows["x"][test])
    return prediction


def report(rows: dict, prediction: np.ndarray) -> dict:
    scored = np.isfinite(prediction)
    clean, excursion = scored & ~rows["excursion"], scored & rows["excursion"]
    axis = rows["x"][:, -1]
    disagreement = np.abs((axis - prediction) - (axis - rows["y"]))
    median = lambda v: round(float(np.median(v)), 3)
    out = {"offset_prediction": {}, "detection": {"by_recording": {}, "operating_points": {}}}
    for name in sorted(set(rows["group"])):
        at = clean & (rows["group"] == name)
        if at.sum() < 20:
            continue
        out["offset_prediction"][name] = {"rows": int(at.sum()), "median_abs_error_m": {
            "analytic_baseline": median(np.abs(rows["baseline"][at] - rows["y"][at])),
            "trivial_track_on_axis": median(np.abs(rows["trivial"][at] - rows["y"][at])),
            "model": median(np.abs(prediction[at] - rows["y"][at]))}}
        far = excursion & (rows["group"] == name)
        out["detection"]["by_recording"][name] = {
            "clean_rows": int(at.sum()), "excursion_rows": int(far.sum()),
            "median_disagreement_clean_m": median(disagreement[at]),
            "median_disagreement_excursion_m": median(disagreement[far]) if far.sum() >= 20 else None}
    out["offset_prediction"]["POOLED"] = {"rows": int(clean.sum()), "median_abs_error_m": {
        "analytic_baseline": median(np.abs(rows["baseline"][clean] - rows["y"][clean])),
        "trivial_track_on_axis": median(np.abs(rows["trivial"][clean] - rows["y"][clean])),
        "model": median(np.abs(prediction[clean] - rows["y"][clean]))}}
    for threshold in (0.3, 0.5, 0.75, 1.0, 1.5):
        out["detection"]["operating_points"][str(threshold)] = {
            "caught_excursion_rows": round(float(np.mean(disagreement[excursion] > threshold)), 3),
            "false_on_clean_rows": round(float(np.mean(disagreement[clean] > threshold)), 4)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, action="append", required=True)
    parser.add_argument("--stability", type=Path, required=True,
                        help="center_stability output that marks which frames the rails got wrong")
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--profile-cache", type=Path, default=Path("build/section-profiles.npz"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--excursion-probe", default="30.0")
    parser.add_argument("--excursion-limit-m", type=float, default=1.0)
    args = parser.parse_args()

    detector = load_config(args.config)
    detector = {**detector, "background": {**detector["background"], "enabled": False}}
    data = build_profiles(args.bag, detector, args.profile_cache)
    rows = assemble(data, excursion_frames(args.stability, args.excursion_probe,
                                           args.excursion_limit_m))
    prediction = leave_one_recording_out(rows, detector["seed"])
    result = report(rows, prediction)
    result |= {"config": str(args.config), "seed": detector["seed"], "rows": int(len(rows["y"])),
               "features": int(rows["x"].shape[1]),
               "protocol": ("Leave-one-recording-out. Excursion frames are excluded from training "
                            "because their target is the quantity that is wrong, so predicting on "
                            "them tests an unseen regime. The trivial predictor claims the track "
                            "runs down the sensor axis and is reported because this subset is "
                            "mostly aligned.")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1) + "\n")

    pooled = result["offset_prediction"]["POOLED"]["median_abs_error_m"]
    print(f"\n{result['rows']} rows, {result['features']} features")
    print(f"median offset error: analytic {pooled['analytic_baseline']}  "
          f"trivial {pooled['trivial_track_on_axis']}  model {pooled['model']} m")
    for threshold, point in result["detection"]["operating_points"].items():
        print(f"  disagreement > {threshold} m: catches {point['caught_excursion_rows']:.0%} of "
              f"excursion rows, {point['false_on_clean_rows']:.1%} false on clean")
    print(f"\nEvidence: {args.output}")


if __name__ == "__main__":
    main()
