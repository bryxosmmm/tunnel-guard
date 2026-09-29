"""Required C++ detector kernels.

Required C++ detector kernels. This module contains only native dispatch;
Python retains orchestration and algorithms without duplicate implementations.
"""
from __future__ import annotations


import numpy as np
from scipy.sparse import csr_matrix

# Grid resolution for the native pair search. It only bounds how many candidate
# points a radius query inspects: the enumeration covers each point's radius
# ball exactly, so the retained pair set cannot depend on this value.
GRID_CELL_M = 0.25


def native(config: dict):
    """Load the required detector extension or explain how to build it.

    `native_backend` selects the implementation: "cpu" (the default, and the only
    one the required build produces) or "cuda", which additionally needs the
    optional device backend from `python setup_cuda.py build_ext --inplace`. The
    device module exposes the same entry points and forwards the ones it has not
    ported to `_native`, so a missing kernel degrades to the CPU one rather than
    to a wrong answer. A "cuda" request that cannot be satisfied is an error, not
    a silent fallback: a run must not claim a device it did not use.
    """
    try:
        from . import _native
    except ImportError as exc:
        raise RuntimeError("Required detector kernels are unavailable; build them with `python setup.py build_ext --inplace`.") from exc
    backend = config.get("native_backend", "cpu")
    if backend == "cpu":
        return _native
    if backend != "cuda":
        raise ValueError(f"Unknown native_backend {backend!r}; expected 'cpu' or 'cuda'")
    try:
        from . import _native_cuda
    except ImportError as exc:
        raise RuntimeError(
            "native_backend is 'cuda' but the device backend is not importable; build it with "
            "`python setup_cuda.py build_ext --inplace` on a machine with the CUDA toolkit and a GPU."
        ) from exc
    return _native_cuda



def cpu_native(config: dict):
    """The CPU kernels, for a caller that must not touch the device backend.

    Reader preparation stays on the CPU while the detector uses the device.
    CUDA calls serialize access to shared device scratch; sending reader work
    there would add contention and transfers. CPU scratch is thread-local.
    """
    try:
        from . import _native
    except ImportError as exc:
        raise RuntimeError("Required detector kernels are unavailable; build them with "
                           "`python setup.py build_ext --inplace`.") from exc
    return _native


def segment_plane(points: np.ndarray, threshold: float, ransac_n: int, iterations: int, probability: float,
                  module):
    """Exact port of Open3D's SegmentPlane: same plane and same inlier set.

    Only the device backend implements it, and only `background.plane_backend: native`
    selects it. The port is verified against Open3D call by call by
    `scripts/bench_ransac_equivalence.py`, because these proposals define the tunnel
    surface the detector claims are measured against.
    """
    plane, indices = module.segment_plane(np.ascontiguousarray(points), threshold, int(ransac_n),
                                          int(iterations), probability)
    return np.frombuffer(plane, dtype=np.float64), np.frombuffer(indices, dtype=np.int64)


def segment_plane_seed(seed: int, module):
    """Seed the proposal stream, mirroring `o3d.utility.random.seed`."""
    module.segment_plane_seed(int(seed))


def grouped_medians(coords: np.ndarray, keys: np.ndarray, module) -> np.ndarray:
    """Per-key median of each coordinate, in ascending key order."""
    payload = module.grouped_medians(np.ascontiguousarray(coords), np.ascontiguousarray(keys, dtype=np.int64))
    return np.frombuffer(payload, dtype=np.float64).reshape(-1, 2)


def range_indices_open(points: np.ndarray, minimum: float, maximum: float, module):
    """Rows with `minimum < |p| < maximum`, in input order."""
    return np.frombuffer(module.range_indices_open(np.ascontiguousarray(points), minimum, maximum),
                         dtype=np.int64)


def range_summary(reduced: np.ndarray, frame: np.ndarray, crop: np.ndarray, observed: np.ndarray,
                  bins: np.ndarray, module):
    """Per-bin (returns, raw cropped returns, geometry-supported returns)."""
    payload = module.range_summary(np.ascontiguousarray(reduced), np.ascontiguousarray(frame),
                                   np.ascontiguousarray(crop), np.ascontiguousarray(observed),
                                   np.asarray(bins, dtype=float))
    return np.frombuffer(payload, dtype=np.int64).reshape(-1, 3)


def ground_profile(points: np.ndarray, plane: np.ndarray, segment: float, maximum: float, local_window: float,
                   half_width: float, inlier: float, min_support: float, slope_limit: float, module):
    """Track-bed profile anchors along the recording axis."""
    payload = module.ground_profile(np.ascontiguousarray(points), np.asarray(plane, dtype=float), segment, maximum,
                                    local_window / 2.0, half_width, inlier, min_support, slope_limit)
    return np.frombuffer(payload, dtype=np.float64).reshape(-1, 3)


def window_indices(points: np.ndarray, bounds: np.ndarray, module):
    """Concatenated window membership for every domain, with slice offsets."""
    indices, offsets = module.window_indices(np.ascontiguousarray(points), np.asarray(bounds, dtype=float))
    return np.frombuffer(indices, dtype=np.int64), np.frombuffer(offsets, dtype=np.int64)


def remove_rows(points: np.ndarray, rows: np.ndarray, buffer: np.ndarray, module):
    """Points with the listed rows dropped, written into `buffer`, order preserved."""
    kept = module.remove_rows(np.ascontiguousarray(points), np.ascontiguousarray(rows, dtype=np.int64), buffer)
    return buffer[:kept]


def support_strips(support: np.ndarray, transverse: int, strip_m: float, min_support: float, min_span: float,
                   module):
    """Observed strips (lo_x, hi_x, lo_t, hi_t) of a plane's support."""
    payload = module.support_strips(np.ascontiguousarray(support), int(transverse), strip_m, min_support, min_span)
    return np.frombuffer(payload, dtype=np.float64).reshape(-1, 4)


def component_labels(graph, subset, module):
    """Component labels of a graph subset, numbered like scipy's."""
    payload = module.component_labels(np.ascontiguousarray(graph.indptr, dtype=np.int64),
                                     np.ascontiguousarray(graph.indices, dtype=np.int64),
                                     np.ascontiguousarray(subset))
    return np.frombuffer(payload, dtype=np.int64)


def induced_subgraph(graph, rows: np.ndarray, module):
    """Unit-weight CSR subgraph for sorted unique nodes, preserving their row order."""
    offsets, columns = module.induced_subgraph(
        np.ascontiguousarray(graph.indptr, dtype=np.int64),
        np.ascontiguousarray(graph.indices, dtype=np.int64),
        np.ascontiguousarray(rows, dtype=np.int64))
    indices = np.frombuffer(columns, dtype=np.int64)
    size = len(rows)
    return csr_matrix((np.ones(len(indices), dtype=np.uint8), indices,
                       np.frombuffer(offsets, dtype=np.int64)), shape=(size, size))

def group_degrees(graph, rows: np.ndarray, group: np.ndarray, selected: np.ndarray, module):
    """In-group neighbour counts per row, in row order: all members, and only selected ones."""
    payload = module.group_degrees(
        np.ascontiguousarray(graph.indptr, dtype=np.int64),
        np.ascontiguousarray(graph.indices, dtype=np.int64),
        np.ascontiguousarray(rows, dtype=np.int64),
        np.ascontiguousarray(group, dtype=np.int64),
        np.ascontiguousarray(selected, dtype=bool))
    return (np.frombuffer(payload[0], dtype=np.int64), np.frombuffer(payload[1], dtype=np.int64))


def crop_voxels(frame, min_forward, half_width, size, module):
    return np.frombuffer(module.select_crop_voxels(np.ascontiguousarray(frame), min_forward, half_width, size), dtype=np.int64)



def range_indices(points, minimum, maximum, module):
    return np.frombuffer(module.range_indices(np.ascontiguousarray(points), minimum, maximum), dtype=np.int64)



def density_graph(metric, radius, config, module):
    indptr, indices, data, degree = module.mutual_graph(np.ascontiguousarray(metric), np.ascontiguousarray(radius), GRID_CELL_M)
    n = len(metric)
    graph = csr_matrix((np.frombuffer(data, dtype=np.uint8), np.frombuffer(indices, dtype=np.int64),
                        np.frombuffer(indptr, dtype=np.int64)), shape=(n, n))
    return graph, np.frombuffer(degree, dtype=np.int64) + 1



def patch_candidates(leveled, plane, low, high, distance, module):
    return np.frombuffer(module.patch_candidates(np.ascontiguousarray(leveled), np.asarray(plane, dtype=float), low, high, distance), dtype=np.int64)



def protrusion_ids(sample, normals, reliable, plane, depth, radius, alignment, module):
    return np.frombuffer(module.protrusion_ids(np.ascontiguousarray(sample), np.ascontiguousarray(normals), np.ascontiguousarray(reliable), np.asarray(plane, dtype=float), depth, radius, alignment), dtype=np.int64)



def keep_outside_radius(query, targets, radius, module):
    return ~np.frombuffer(module.within_radius(np.ascontiguousarray(query), np.ascontiguousarray(targets), radius), dtype=np.uint8).astype(bool)



def strip_inside(leveled, ids, strips, margin, transverse, module):
    return np.frombuffer(module.strip_inside(np.ascontiguousarray(leveled), np.ascontiguousarray(ids, dtype=np.int64), np.asarray(strips, dtype=float), margin, int(transverse)), dtype=np.int64)



def ground_values(points, plane, ground_anchors, maximum, module):
    z, uncertainty = module.ground_values(np.ascontiguousarray(points), np.asarray(plane, dtype=float), np.asarray(ground_anchors, dtype=float), maximum)
    return np.frombuffer(z, dtype=np.float64), np.frombuffer(uncertainty, dtype=np.float64)



def classify_geometry(points, geometry, module):
    if geometry.config.get("rail_frame_mode", "bed") != "bed":
        raise ValueError("rail_frame_mode must be bed; only the C++ bed classifier is supported")
    config = geometry.config
    rail_head = geometry.rail_head_height_m if geometry.rail_head_height_m is not None else float("nan")
    payload = module.classify_geometry(
        np.ascontiguousarray(points), np.asarray([] if geometry.plane is None else geometry.plane, dtype=float),
        np.asarray(geometry.ground_anchors, dtype=float), np.asarray(geometry.rail_anchors, dtype=float),
        np.asarray(config["envelope_segments_m"], dtype=float), rail_head,
        config["ground_max_uncertainty_m"], config["path_max_uncertainty_m"],
        config["ground_max_extrapolation_m"], config["path_max_extrapolation_m"],
        config["rail_half_width_m"], config["rail_vertical_margin_m"], config["min_running_height_m"],
        config["cluster_context_margin_m"], config["segmentation_context_half_width_m"],
        config["envelope_margin_m"], config["rail_max_heading"],
        config.get("path_curve_window_m", 30.0), config.get("path_curvature_significance", 4.0),
        config["rail_gauge_m"])
    return tuple(np.frombuffer(payload[i], dtype=np.uint8).astype(bool) if i in (0, 1, 3, 4, 5)
                 else np.frombuffer(payload[i], dtype=np.float64) for i in range(9))



RELATION_NAMES = ("adjacent", "intersecting", "unresolved", "unknown")
DISTANCE_METHODS = ("cluster_min_x", "supported_envelope_min_x", "unresolved_envelope_evidence_min_x")
REJECTION_NAMES = {0: None, 1: "below_weak_min_voxels", 2: "below_min_extent",
                   3: "weak_without_envelope_support"}


def cluster_objects(cloud, labels, core, boundary, density_core, heights, uncertain_support, config, module,
                    *, geometry_valid):
    """Component statistics for the cluster cloud, in one native call.

    Returns the accepted candidate dictionaries in component order, the rejection
    counts keyed by reason, and one (component_id, reason, points) row per
    component for the diagnostic listing. `_support_points` carries the
    component's own cloud rows, exactly as the reference loop slices them.
    """
    payload = module.cluster_components(
        np.ascontiguousarray(cloud), np.ascontiguousarray(labels, dtype=np.int64),
        np.ascontiguousarray(core), np.ascontiguousarray(boundary), np.ascontiguousarray(density_core),
        np.ascontiguousarray(heights), np.ascontiguousarray(uncertain_support),
        int(config["weak_min_voxels"]), int(config["immediate_min_voxels"]),
        float(config["cluster_min_extent_m"]), float(config["immediate_min_height_m"]),
        1 if config.get("obstacle_distance_mode", "cluster_min_x") == "envelope_support_min_x" else 0,
        int(config.get("claim_min_support_voxels", config["weak_min_voxels"])), int(geometry_valid))
    (ids, offsets, members, reasons, relations, distance_codes, bbox_min, bbox_max,
     centres, extents, height_spans, witnesses, distances, support_points, nearest_cluster,
     nearest_supported, nearest_unresolved, support_counts, dense_counts, envelope_counts,
     uncertain_counts,
     boundary_counts, immediate, interior_dense, interior_heights, intersection_immediate) = payload

    def integers(block):
        return np.frombuffer(block, dtype=np.int64)

    def reals(block):
        return np.frombuffer(block, dtype=np.float64)

    def flags(block):
        return np.frombuffer(block, dtype=np.uint8).astype(bool)

    ids, offsets, members = integers(ids), integers(offsets), integers(members)
    reasons, relations = integers(reasons), integers(relations)
    distance_codes = integers(distance_codes)
    bbox_min, bbox_max = reals(bbox_min), reals(bbox_max)
    centres, extents, height_spans = reals(centres), reals(extents), reals(height_spans)
    witnesses, distances = reals(witnesses), reals(distances)
    support_points = integers(support_points)
    nearest_cluster, nearest_supported = reals(nearest_cluster), reals(nearest_supported)
    nearest_unresolved = reals(nearest_unresolved)
    support_counts, dense_counts = integers(support_counts), integers(dense_counts)
    envelope_counts, uncertain_counts = integers(envelope_counts), integers(uncertain_counts)
    boundary_counts = integers(boundary_counts)
    interior_dense, interior_heights = integers(interior_dense), reals(interior_heights)
    immediate, intersection_immediate = flags(immediate), flags(intersection_immediate)
    objects, rows, rejected = [], [], {}
    for row, component in enumerate(ids):
        begin, end = int(offsets[row]), int(offsets[row + 1])
        reason = REJECTION_NAMES[int(reasons[row])]
        rows.append({"component_id": int(component), "reason": reason or "accepted", "points": end - begin})
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        objects.append({
            "bbox_min": bbox_min[3 * row:3 * row + 3].tolist(),
            "bbox_max": bbox_max[3 * row:3 * row + 3].tolist(),
            "component_id": int(component),
            "cluster_nearest_x_m": float(nearest_cluster[row]),
            "supported_envelope_nearest_x_m": None if np.isnan(nearest_supported[row]) else float(nearest_supported[row]),
            "unresolved_envelope_nearest_x_m": None if np.isnan(nearest_unresolved[row]) else float(nearest_unresolved[row]),
            "center": centres[3 * row:3 * row + 3].tolist(),
            "extent_m": extents[3 * row:3 * row + 3].tolist(),
            "distance_m": float(distances[row]),
            "distance_method": DISTANCE_METHODS[int(distance_codes[row])],
            "distance_support_point": witnesses[3 * row:3 * row + 3].tolist(),
            "distance_support_points": int(support_points[row]),
            "path_relation": RELATION_NAMES[int(relations[row])],
            "support_voxels": int(support_counts[row]),
            "density_core_voxels": int(dense_counts[row]),
            "in_envelope_voxels": int(envelope_counts[row]),
            "uncertain_voxels": int(uncertain_counts[row]),
            "_support_points": cloud[members[begin:end]],
            "_support_indices": members[begin:end],
            "boundary_uncertain_voxels": int(boundary_counts[row]),
            "height_above_bed_m": [float(h) if np.isfinite(h) else None
                                  for h in height_spans[2 * row:2 * row + 2]],
            "immediate": bool(immediate[row]),
            "interior_density_core_voxels": int(interior_dense[row]),
            "interior_height_span_m": float(interior_heights[row]),
            "intersection_immediate": bool(intersection_immediate[row]),
        })
    return objects, rejected, rows


def normal_statistics(sample: np.ndarray, radius: float, max_nn: int, min_neighbors: int, module):
    """Neighbour counts, covariances, eigenvalues and normals in one grid pass.

    The eigenvalues and the smallest eigenvector are produced natively by Jacobi
    rotations with a fixed sweep count, so the result is a deterministic function
    of the covariance and stays accurate for repeated eigenvalues.
    """
    counts, components, eigenvalues, normals = module.normal_covariances(
        np.ascontiguousarray(sample), radius, int(max_nn), int(min_neighbors))
    return (np.frombuffer(counts, dtype=np.int64),
            np.frombuffer(components, dtype=np.float64).reshape(-1, 6),
            np.frombuffer(eigenvalues, dtype=np.float64).reshape(-1, 3),
            np.frombuffer(normals, dtype=np.float64).reshape(-1, 3))


def voxel_counts(stacks, size, module):
    return np.frombuffer(module.voxel_counts(list(stacks), size), dtype=np.int64)



def mask_candidates(leveled, planes, bounds, distance, module):
    union, candidates, offsets = module.mask_candidates(np.ascontiguousarray(leveled), np.asarray(planes, dtype=float), np.asarray(bounds, dtype=float), distance)
    return np.frombuffer(union, dtype=np.int64), np.frombuffer(candidates, dtype=np.int64), np.frombuffer(offsets, dtype=np.int64)



def mask_apply(leveled: np.ndarray, protected: np.ndarray, background: np.ndarray, planes: np.ndarray,
               transverse: np.ndarray, strips: np.ndarray, strip_offsets: np.ndarray, candidates: np.ndarray,
               candidate_offsets: np.ndarray, reliable: np.ndarray, aligned: np.ndarray, sample: np.ndarray,
               normals: np.ndarray, normal_reliable: np.ndarray, margin: float, depth: float, radius: float,
               alignment: float, module) -> None:
    """Mark the caller's background mask for every patch, in patch order.

    `reliable` and `aligned` are the neighbour-query results expanded to
    candidate order, exactly as the reference loop in
    tunnel_guard/background.py indexes them. This is the native branch only;
    that loop is the reference.
    """
    module.mask_apply(np.ascontiguousarray(leveled), protected, background,
                      np.asarray(planes, dtype=float), transverse, np.asarray(strips, dtype=float),
                      strip_offsets, candidates, candidate_offsets, reliable, aligned,
                      sample, normals, normal_reliable, margin, depth, radius, alignment)
