"""Throwaway: array-level equivalence of the native kernels against their references.

Captures real per-frame inputs from a running detector, then evaluates each
kernel and the NumPy expression it replaces on the same arrays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from tunnel_guard import _native
from tunnel_guard import background as background_mod
from tunnel_guard import detector as detector_mod
from tunnel_guard import segmentation as segmentation_mod
from tunnel_guard.detector import Detector, load_config
from tunnel_guard.geometry import nearest_anchor_distance
from tunnel_guard.io import iter_bag

CAPTURE: dict = {}
RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    RESULTS.append((name, bool(ok), detail))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=3)
    args = parser.parse_args()

    real_labels = segmentation_mod.density_labels
    real_mask = background_mod.TunnelBackground.mask

    def capturing_labels(points, config):
        CAPTURE.setdefault("clouds", []).append(points)
        return real_labels(points, config)

    def capturing_mask(self, points, protected):
        CAPTURE.setdefault("masks", []).append((self, points, protected))
        return real_mask(self, points, protected)

    segmentation_mod.density_labels = capturing_labels
    detector_mod.density_labels = capturing_labels
    background_mod.TunnelBackground.mask = capturing_mask
    config = load_config(args.config)
    detector = Detector(config)
    scans = []
    for scan in iter_bag(args.bag, config, max_frames=args.frames):
        scans.append(scan)
        detector.process(scan.points, scan.timestamp_s, scan.point_times)
    segmentation_mod.density_labels = real_labels
    detector_mod.density_labels = real_labels
    background_mod.TunnelBackground.mask = real_mask

    # 1. range filter
    for scan in scans:
        points = np.asarray(scan.points, dtype=np.float64)
        radii = np.linalg.norm(points, axis=1)
        reference = np.flatnonzero((radii >= config["min_range_m"]) & (radii <= config["max_range_m"]))
        native = np.frombuffer(_native.range_indices(points, config["min_range_m"], config["max_range_m"]), dtype=np.int64)
        check("range_indices", np.array_equal(reference, native),
              f"frame {scan.index}: {len(reference)} vs {len(native)}")

    # 2. mutual graph vs the NumPy/scipy construction, including labels
    for cloud in CAPTURE.get("clouds", [])[:3]:
        distance = np.linalg.norm(cloud, axis=1)
        radius = np.clip(config["density_radius_m"] + config["density_angular_radius_rad"] * distance,
                         config["density_radius_m"], config["cluster_max_radius_m"])
        metric = cloud * np.array([1., 1., config["density_vertical_scale"]])
        neighbours = cKDTree(metric).query_ball_point(metric, radius, return_sorted=False)
        counts = np.fromiter(map(len, neighbours), dtype=np.int64, count=len(cloud))
        row = np.repeat(np.arange(len(cloud)), counts)
        column = np.concatenate(neighbours).astype(np.int64, copy=False)
        forward = row < column
        pairs = np.column_stack((row[forward], column[forward]))
        delta = metric[pairs[:, 0]] - metric[pairs[:, 1]]
        pairs = pairs[np.einsum("ij,ij->i", delta, delta) <= np.minimum(radius[pairs[:, 0]], radius[pairs[:, 1]])**2]
        edge_i = np.concatenate((pairs[:, 0], pairs[:, 1]))
        edge_j = np.concatenate((pairs[:, 1], pairs[:, 0]))
        reference = coo_matrix((np.ones(len(edge_i), dtype=np.uint8), (edge_i, edge_j)),
                               shape=(len(cloud), len(cloud))).tocsr()

        indptr, indices, data, degree = _native.mutual_graph(metric, radius, 0.25)
        indptr = np.frombuffer(indptr, dtype=np.int64)
        indices = np.frombuffer(indices, dtype=np.int64)
        data = np.frombuffer(data, dtype=np.uint8)
        degree = np.frombuffer(degree, dtype=np.int64) + 1
        native = csr_matrix((data, indices, indptr), shape=(len(cloud), len(cloud)))
        same_arrays = (np.array_equal(indptr, reference.indptr) and np.array_equal(indices, reference.indices)
                       and np.array_equal(data, reference.data))
        same_degree = np.array_equal(degree, np.asarray(reference.sum(axis=1)).ravel() + 1)
        check("mutual_graph arrays", same_arrays, f"{len(cloud)} points, {len(data)} directed edges")
        check("mutual_graph degree", same_degree)

        required = np.maximum(config["density_min_far"], np.ceil(config["density_min_near"] *
                              np.minimum(1., (config["density_reference_range_m"] / np.maximum(distance, 1.))**2)))
        core = degree >= required
        core_ids = np.flatnonzero(core)
        labels_reference = np.full(len(cloud), -1, dtype=np.int32)
        count, core_labels = connected_components(reference[core_ids][:, core_ids], directed=False)
        labels_reference[core_ids] = core_labels
        border_ids = np.flatnonzero(~core)
        if len(border_ids):
            dd, near = cKDTree(metric[core_ids]).query(metric[border_ids])
            accepted = dd <= np.minimum(radius[border_ids], radius[core_ids[near]])
            labels_reference[border_ids[accepted]] = core_labels[near[accepted]]
        weak_ids = np.flatnonzero(labels_reference < 0)
        if len(weak_ids):
            _, weak_labels = connected_components(reference[weak_ids][:, weak_ids], directed=False)
            labels_reference[weak_ids] = count + weak_labels

        labels_native = np.full(len(cloud), -1, dtype=np.int32)
        count_n, core_labels_n = connected_components(native[core_ids][:, core_ids], directed=False)
        labels_native[core_ids] = core_labels_n
        if len(border_ids):
            dd, near = cKDTree(metric[core_ids]).query(metric[border_ids])
            accepted = dd <= np.minimum(radius[border_ids], radius[core_ids[near]])
            labels_native[border_ids[accepted]] = core_labels_n[near[accepted]]
        weak_ids_n = np.flatnonzero(labels_native < 0)
        if len(weak_ids_n):
            _, weak_labels_n = connected_components(native[weak_ids_n][:, weak_ids_n], directed=False)
            labels_native[weak_ids_n] = count_n + weak_labels_n
        check("mutual_graph component labels", np.array_equal(labels_reference, labels_native),
              f"components {labels_reference.max() + 1} vs {labels_native.max() + 1}")

    # 3. mask helpers
    for instance, points, protected in CAPTURE.get("masks", [])[:1]:
        cfg = instance.config
        leveled = points @ instance.rotation.T
        for plane, transverse, strips, _ in instance.patches[:6]:
            low = min(strip[0] for strip in strips) - cfg["support_margin_m"]
            high = max(strip[1] for strip in strips) + cfg["support_margin_m"]
            reference = np.flatnonzero((leveled[:, 0] >= low) & (leveled[:, 0] <= high))
            reference = reference[np.abs(leveled[reference] @ plane[:3] + plane[3]) <= cfg["remove_distance_m"]]
            native = np.frombuffer(_native.patch_candidates(leveled, np.asarray(plane, dtype=float), low, high,
                                                            cfg["remove_distance_m"]), dtype=np.int64)
            check("patch_candidates", np.array_equal(reference, native), f"{len(reference)} vs {len(native)}")

            sample_distance = np.abs(instance.sample @ plane[:3] + plane[3])
            reference_protrusion = np.flatnonzero(
                instance.normal_reliable & (sample_distance >= cfg["protrusion_depth_m"])
                & (sample_distance <= cfg["protection_radius_m"])
                & (np.abs(instance.normals @ plane[:3]) < cfg["normal_alignment_cos"]))
            native_protrusion = np.frombuffer(
                _native.protrusion_ids(instance.sample, instance.normals, instance.normal_reliable,
                                       np.asarray(plane, dtype=float), cfg["protrusion_depth_m"],
                                       cfg["protection_radius_m"], cfg["normal_alignment_cos"]), dtype=np.int64)
            check("protrusion_ids", np.array_equal(reference_protrusion, native_protrusion),
                  f"{len(reference_protrusion)} vs {len(native_protrusion)}")

            if len(reference_protrusion) and len(reference):
                distances, _ = cKDTree(instance.sample[reference_protrusion]).query(leveled[reference])
                keep_reference = (distances > cfg["protection_radius_m"])
                hit = np.frombuffer(_native.within_radius(leveled[reference], instance.sample[reference_protrusion],
                                                          cfg["protection_radius_m"]), dtype=np.uint8).astype(bool)
                check("within_radius", np.array_equal(keep_reference, ~hit),
                      f"clear {int(keep_reference.sum())} vs {int((~hit).sum())}")

            near = np.flatnonzero(~protected[reference] & (leveled[reference, 0] >= low))
            inside_reference = np.zeros(len(near), dtype=bool)
            for strip_low, strip_high, bottom, top in strips:
                q = leveled[near]
                inside_reference |= ((q[:, 0] >= strip_low - cfg["support_margin_m"]) & (q[:, 0] <= strip_high + cfg["support_margin_m"])
                                     & (q[:, transverse] >= bottom - cfg["support_margin_m"])
                                     & (q[:, transverse] <= top + cfg["support_margin_m"]))
            inside_native = np.frombuffer(_native.strip_inside(leveled, near, np.asarray(strips, dtype=float),
                                                               cfg["support_margin_m"], int(transverse)), dtype=np.int64)
            check("strip_inside", np.array_equal(near[inside_reference], inside_native),
                  f"{int(inside_reference.sum())} vs {len(inside_native)}")

    # 4. nearest-anchor helper (already verified exhaustively earlier)
    rng = np.random.default_rng(1)
    anchors = np.sort(rng.uniform(0, 250, 40))
    x = rng.uniform(-10, 260, 5000)
    dense = np.min(np.abs(x[:, None] - anchors[None, :]), axis=1)
    check("nearest_anchor_distance", np.array_equal(dense, nearest_anchor_distance(x, anchors)))

    failures = [row for row in RESULTS if not row[1]]
    for name, ok, detail in RESULTS:
        print(f"{'ok  ' if ok else 'FAIL'} {name:26s} {detail}")
    print(f"\n{len(RESULTS) - len(failures)}/{len(RESULTS)} kernel checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
