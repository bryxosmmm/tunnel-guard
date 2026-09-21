"""Optional native kernels, each with the NumPy expression it must reproduce.

The native paths are selected by `native_kernels` in the detector recipe and
require the locally built `tunnel_guard._native` extension. Every function here
returns exactly what the NumPy branch returns: same selection, same arithmetic
order, same tie-breaking. Nothing in this module defines a threshold or changes
a decision; it only evaluates predicates the detector already applies.

Keeping both branches side by side is deliberate: the NumPy branch is the
reference, the native branch is an accelerator, and the project has no test
suite in which a silent divergence could hide.
"""
from __future__ import annotations

from itertools import chain

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial import cKDTree

# Grid resolution for the native pair search. It only bounds how many candidate
# points a radius query inspects: the enumeration covers each point's radius
# ball exactly, so the retained pair set cannot depend on this value.
GRID_CELL_M = 0.25


def native(config: dict):
    """Return the extension module when this recipe selects native kernels."""
    if not config.get("native_kernels", False):
        return None
    from . import _native
    return _native


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


def crop_voxels(frame: np.ndarray, min_forward: float, half_width: float, size: float, module=None):
    """Indices of the voxel representatives inside a longitudinal window.

    Numerically identical to reducing frame[crop] on the same grid: the retained
    measurement per voxel is the first in input order and the rows come back in
    key order. The native branch does the crop, the key computation and the
    reduction in one partitioned pass, so the cropped copy never exists.
    """
    if module is not None:
        return np.frombuffer(module.select_crop_voxels(np.ascontiguousarray(frame), min_forward,
                                                       half_width, size), dtype=np.int64)
    crop = (frame[:, 0] >= min_forward) & (np.abs(frame[:, 1]) < half_width)
    window = np.flatnonzero(crop)
    cropped = frame[window]
    if not len(cropped):
        return np.empty(0, dtype=np.int64)
    _, first = np.unique(np.floor(cropped / size).astype(np.int64), axis=0, return_index=True)
    return window[first]


def range_indices(points: np.ndarray, minimum: float, maximum: float, module=None) -> np.ndarray:
    """Rows whose radius lies inside the sensor band."""
    if module is not None:
        return np.frombuffer(module.range_indices(points, minimum, maximum), dtype=np.int64)
    radii = np.linalg.norm(points, axis=1)
    return np.flatnonzero((radii >= minimum) & (radii <= maximum))


def density_graph(metric: np.ndarray, radius: np.ndarray, config: dict, module=None):
    """Mutual-radius symmetric graph and per-row partner count of a cluster cloud.

    The retained pairs are { (i, j) : i < j, d2(i, j) <= min(r_i, r_j)^2 } with d2
    accumulated as dx*dx + dy*dy + dz*dz. The NumPy branch's radius query is only
    a candidate filter and is a strict superset of that predicate.
    """
    n = len(metric)
    if module is not None:
        indptr, indices, data, degree = module.mutual_graph(metric, radius, GRID_CELL_M)
        indptr = np.frombuffer(indptr, dtype=np.int64)
        indices = np.frombuffer(indices, dtype=np.int64)
        data = np.frombuffer(data, dtype=np.uint8)
        # scipy accepts the canonical CSR directly; the arrays already carry
        # sorted columns and summed duplicates, exactly like coo_matrix().tocsr().
        graph = csr_matrix((data, indices, indptr), shape=(n, n))
        return graph, np.frombuffer(degree, dtype=np.int64) + 1
    from .geometry import query_workers
    tree = cKDTree(metric)
    configured = config.get("query_workers", -1)
    neighbors = tree.query_ball_point(metric, radius, return_sorted=False,
                                      workers=query_workers(n, configured))
    counts = np.fromiter(map(len, neighbors), dtype=np.int64, count=n)
    row = np.repeat(np.arange(n), counts)
    column = np.fromiter(chain.from_iterable(neighbors), dtype=np.int64, count=int(counts.sum()))
    forward = row < column
    pairs = np.column_stack((row[forward], column[forward]))
    if len(pairs):
        delta = metric[pairs[:, 0]] - metric[pairs[:, 1]]
        pairs = pairs[np.einsum("ij,ij->i", delta, delta)
                      <= np.minimum(radius[pairs[:, 0]], radius[pairs[:, 1]])**2]
    edge_i = np.concatenate((pairs[:, 0], pairs[:, 1]))
    edge_j = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix((np.ones(len(edge_i), dtype=np.uint8), (edge_i, edge_j)), shape=(n, n)).tocsr()
    return graph, np.asarray(graph.sum(axis=1)).ravel() + 1


def patch_candidates(leveled: np.ndarray, plane: np.ndarray, low: float, high: float,
                     distance: float, module=None) -> np.ndarray:
    """Rows inside a patch's longitudinal window and plane distance band."""
    if module is not None:
        return np.frombuffer(module.patch_candidates(leveled, np.asarray(plane, dtype=float),
                                                     low, high, distance), dtype=np.int64)
    window = np.flatnonzero((leveled[:, 0] >= low) & (leveled[:, 0] <= high))
    return window[np.abs(leveled[window] @ plane[:3] + plane[3]) <= distance]


def protrusion_ids(sample: np.ndarray, normals: np.ndarray, reliable: np.ndarray, plane: np.ndarray,
                   depth: float, radius: float, alignment: float, module=None) -> np.ndarray:
    """Samples that protect an attachment edge against this plane."""
    if module is not None:
        return np.frombuffer(module.protrusion_ids(sample, normals, reliable,
                                                   np.asarray(plane, dtype=float),
                                                   depth, radius, alignment), dtype=np.int64)
    distance = np.abs(sample @ plane[:3] + plane[3])
    return np.flatnonzero(reliable & (distance >= depth) & (distance <= radius)
                          & (np.abs(normals @ plane[:3]) < alignment))


def keep_outside_radius(query: np.ndarray, targets: np.ndarray, radius: float, module=None) -> np.ndarray:
    """Per-query flag: no target point lies within the Euclidean radius."""
    if module is not None:
        hit = np.frombuffer(module.within_radius(query, targets, radius), dtype=np.uint8)
        return ~hit.astype(bool)
    distance, _ = cKDTree(targets).query(query)
    return distance > radius


def strip_inside(leveled: np.ndarray, ids: np.ndarray, strips: np.ndarray, margin: float,
                 transverse: int, module=None) -> np.ndarray:
    """Given points landing inside any of a patch's observed strips."""
    if module is not None:
        return np.frombuffer(module.strip_inside(leveled, ids, np.asarray(strips, dtype=float),
                                                 margin, int(transverse)), dtype=np.int64)
    q = leveled[ids]
    inside = np.zeros(len(ids), dtype=bool)
    for strip_low, strip_high, bottom, top in strips:
        inside |= ((q[:, 0] >= strip_low - margin) & (q[:, 0] <= strip_high + margin)
                   & (q[:, transverse] >= bottom - margin) & (q[:, transverse] <= top + margin))
    return ids[inside]


def ground_values(points: np.ndarray, plane: np.ndarray, ground_anchors: np.ndarray, maximum: float,
                  module=None):
    """Track-bed reference height and extrapolation uncertainty."""
    if module is not None:
        z, uncertainty = module.ground_values(np.ascontiguousarray(points), np.asarray(plane, dtype=float),
                                              np.asarray(ground_anchors, dtype=float), maximum)
        return np.frombuffer(z, dtype=np.float64), np.frombuffer(uncertainty, dtype=np.float64)
    from .geometry import nearest_anchor_distance
    x = points[:, 0]
    shift = np.interp(x, ground_anchors[:, 0], ground_anchors[:, 1])
    nearest = nearest_anchor_distance(x, ground_anchors[:, 0])
    uncertainty = np.interp(x, ground_anchors[:, 0], ground_anchors[:, 2]) + nearest * 0.008
    uncertainty[nearest > maximum] = np.inf
    return points[:, :2] @ plane[:2] + plane[2] + shift, uncertainty


def classify_geometry(points: np.ndarray, geometry, module=None):
    """Envelope, rail-relative and ground support classification.

    Returns core, context, height, observed, nominal_overlap, boundary; the NumPy
    reference stays in TrackGeometry.classify for recipes without native kernels.
    """
    if module is None or geometry.plane is None or len(geometry.ground_anchors) < 1 \
            or len(geometry.rail_anchors) < 2:
        return None
    config = geometry.config
    # The C++ kernel implements the legacy bed basis. Experimental local 3D
    # frames use the shared NumPy classifier until profiling warrants a port.
    if config.get("rail_frame_mode", "bed") == "local_3d":
        return None
    rail_head = geometry.rail_head_height_m if geometry.rail_head_height_m is not None else float("nan")
    payload = module.classify_geometry(
        np.ascontiguousarray(points), np.asarray(geometry.plane, dtype=float),
        np.asarray(geometry.ground_anchors, dtype=float), np.asarray(geometry.rail_anchors, dtype=float),
        np.asarray(config["envelope_segments_m"], dtype=float), rail_head,
        config["ground_max_uncertainty_m"], config["path_max_uncertainty_m"],
        config["ground_max_extrapolation_m"], config["path_max_extrapolation_m"],
        config["rail_half_width_m"], config["rail_vertical_margin_m"], config["min_running_height_m"],
        config["cluster_context_margin_m"], config["segmentation_context_half_width_m"],
        config["envelope_margin_m"], config["rail_max_heading"],
        config.get("path_curve_window_m", 30.0), config.get("path_curvature_significance", 4.0))
    return tuple(np.frombuffer(payload[index], dtype=np.uint8).astype(bool) if index in (0, 1, 3, 4, 5)
                 else np.frombuffer(payload[index], dtype=np.float64) for index in range(6))


RELATION_NAMES = ("adjacent", "intersecting", "unresolved")
RELATION_REASONS = ("inside_heuristic_path_and_ground_interval", "envelope_boundary_uncertainty",
                    "unsupported_nominal_envelope", "outside_envelope_evidence")
DISTANCE_METHODS = ("cluster_min_x", "supported_envelope_min_x", "unresolved_envelope_evidence_min_x")
REJECTION_NAMES = {0: None, 1: "below_weak_min_voxels", 2: "below_min_extent",
                   3: "weak_without_envelope_support"}


def cluster_objects(cloud, labels, core, boundary, density_core, heights, uncertain_support, config, module):
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
        1 if config.get("obstacle_distance_mode", "cluster_min_x") == "envelope_support_min_x" else 0)
    (ids, offsets, members, reasons, relations, distance_codes, relation_reasons, bbox_min, bbox_max,
     centres, extents, height_spans, witnesses, distances, support_points, nearest_cluster,
     nearest_supported, nearest_unresolved, support_counts, dense_counts, envelope_counts,
     boundary_counts, immediate, interior_dense, interior_heights, intersection_immediate) = payload

    def integers(block):
        return np.frombuffer(block, dtype=np.int64)

    def reals(block):
        return np.frombuffer(block, dtype=np.float64)

    def flags(block):
        return np.frombuffer(block, dtype=np.uint8).astype(bool)

    ids, offsets, members = integers(ids), integers(offsets), integers(members)
    reasons, relations = integers(reasons), integers(relations)
    distance_codes, relation_reasons = integers(distance_codes), integers(relation_reasons)
    bbox_min, bbox_max = reals(bbox_min), reals(bbox_max)
    centres, extents, height_spans = reals(centres), reals(extents), reals(height_spans)
    witnesses, distances = reals(witnesses), reals(distances)
    support_points = integers(support_points)
    nearest_cluster, nearest_supported = reals(nearest_cluster), reals(nearest_supported)
    nearest_unresolved = reals(nearest_unresolved)
    support_counts, dense_counts = integers(support_counts), integers(dense_counts)
    envelope_counts, boundary_counts = integers(envelope_counts), integers(boundary_counts)
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
            "_support_points": cloud[members[begin:end]],
            "boundary_uncertain_voxels": int(boundary_counts[row]),
            "path_relation_reason": RELATION_REASONS[int(relation_reasons[row])],
            "height_above_bed_m": [float(height_spans[2 * row]), float(height_spans[2 * row + 1])],
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


def voxel_counts(stacks, size: float, module=None):
    """Distinct voxel count per evidence stack, one call for every object.

    Equal to [len(voxel_representatives(stack, size)) for stack in stacks]; the
    counts are integers, so the native and NumPy branches agree exactly.
    """
    if module is not None:
        payload = module.voxel_counts(list(stacks), size)
        return np.frombuffer(payload, dtype=np.int64)
    return np.array([len(np.unique(np.floor(stack / size).astype(np.int64), axis=0)) if len(stack) else 0
                     for stack in stacks], dtype=np.int64)


def mask_candidates(leveled: np.ndarray, planes: np.ndarray, bounds: np.ndarray, distance: float,
                    module=None):
    """All patch candidates, their per-patch slices, and their unique union."""
    if module is not None:
        union, candidates, offsets = module.mask_candidates(
            np.ascontiguousarray(leveled), np.asarray(planes, dtype=float), np.asarray(bounds, dtype=float), distance)
        return (np.frombuffer(union, dtype=np.int64), np.frombuffer(candidates, dtype=np.int64),
                np.frombuffer(offsets, dtype=np.int64))
    offset = [0]
    prepared = []
    for plane, bound in zip(planes, bounds):
        ids = patch_candidates(leveled, plane, bound[0], bound[1], distance)
        prepared.append(ids)
        offset.append(offset[-1] + len(ids))
    union = np.unique(np.concatenate(prepared)) if prepared else np.empty(0, dtype=np.int64)
    return union, np.concatenate(prepared) if prepared else np.empty(0, dtype=np.int64), np.asarray(offset, dtype=np.int64)


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
