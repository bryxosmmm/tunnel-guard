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
    rail_head = geometry.rail_head_height_m if geometry.rail_head_height_m is not None else float("nan")
    payload = module.classify_geometry(
        np.ascontiguousarray(points), np.asarray(geometry.plane, dtype=float),
        np.asarray(geometry.ground_anchors, dtype=float), np.asarray(geometry.rail_anchors, dtype=float),
        np.asarray(config["envelope_segments_m"], dtype=float), rail_head,
        config["ground_max_uncertainty_m"], config["path_max_uncertainty_m"],
        config["ground_max_extrapolation_m"], config["path_max_extrapolation_m"],
        config["rail_half_width_m"], config["rail_vertical_margin_m"], config["min_running_height_m"],
        config["cluster_context_margin_m"], config["segmentation_context_half_width_m"],
        config["envelope_margin_m"], config["rail_max_heading"])
    return tuple(np.frombuffer(payload[index], dtype=np.uint8).astype(bool) if index in (0, 1, 3, 4, 5)
                 else np.frombuffer(payload[index], dtype=np.float64) for index in range(6))
