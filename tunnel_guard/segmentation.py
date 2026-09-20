"""Range-adaptive DBSCAN connectivity and original published comparison backends.

DBSCAN: Ester et al., KDD 1996. Adaptive epsilon/min-support is our adaptation,
not a reproduction claim. TRAVEL and HDBSCAN backends call released code directly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def adaptive_pairs(metric_points: np.ndarray, radius: np.ndarray,
                   base_radius: float) -> np.ndarray:
    """Return exactly the pairs accepted by ``distance <= min(r_i, r_j)``.

    Asking one KD-tree for ``radius.max()`` creates millions of near-field
    candidate pairs only to discard them with the much smaller local radius.
    Partitioning the query points by radius keeps that over-query bounded.  An
    edge is owned by its smaller-radius endpoint (then by index for a tie), so
    every accepted undirected edge is emitted exactly once.
    """
    # Bound the number of trees for unusual recipes without changing the edge
    # predicate; wider buckets only over-query and are filtered exactly below.
    width = max(float(base_radius) / 3.0, float(np.ptp(radius)) / 32.0,
                np.finfo(float).eps)
    buckets = np.floor((radius - radius.min()) / width).astype(np.int32)
    full_tree = cKDTree(metric_points)
    first, second = [], []
    for bucket in np.unique(buckets):
        ids = np.flatnonzero(buckets == bucket)
        limit = np.nextafter(float(radius[ids].max()), np.inf)
        candidates = cKDTree(metric_points[ids]).sparse_distance_matrix(
            full_tree, limit, output_type="coo_matrix")
        left = ids[candidates.row]
        right = candidates.col
        owns = ((radius[left] < radius[right])
                | ((radius[left] == radius[right]) & (left < right)))
        possible = (left != right) & owns
        left, right = left[possible], right[possible]
        delta = metric_points[left] - metric_points[right]
        accepted = (np.einsum("ij,ij->i", delta, delta)
                    <= np.minimum(radius[left], radius[right]) ** 2)
        first.append(left[accepted])
        second.append(right[accepted])
    if not first:
        return np.empty((0, 2), dtype=np.intp)
    return np.column_stack((np.concatenate(first), np.concatenate(second)))


def density_labels(points: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """Dense cores do not join through a thin chain of border points.

    Return strong instance labels and a mask of density-core points. Unassigned
    spatial components remain weak hypotheses; they cannot trigger immediately.
    """
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool)
    distance = np.linalg.norm(points, axis=1)
    radius = np.clip(config["density_radius_m"] + config["density_angular_radius_rad"] * distance,
                     config["density_radius_m"], config["cluster_max_radius_m"])
    metric_points = points * np.array([1., 1., config["density_vertical_scale"]])
    pairs = adaptive_pairs(metric_points, radius, config["density_radius_m"])
    edge_i = np.concatenate((pairs[:, 0], pairs[:, 1]))
    edge_j = np.concatenate((pairs[:, 1], pairs[:, 0]))
    graph = coo_matrix((np.ones(len(edge_i), dtype=np.uint8), (edge_i, edge_j)), shape=(n, n)).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel() + 1
    required = np.maximum(config["density_min_far"], np.ceil(config["density_min_near"] *
                          np.minimum(1., (config["density_reference_range_m"] / np.maximum(distance, 1.))**2)))
    core = degree >= required
    labels = np.full(n, -1, dtype=np.int32)
    core_ids = np.flatnonzero(core)
    count = 0
    if len(core_ids):
        count, core_labels = connected_components(graph[core_ids][:, core_ids], directed=False)
        labels[core_ids] = core_labels
        border_ids = np.flatnonzero(~core)
        if len(border_ids):
            dd, near = cKDTree(metric_points[core_ids]).query(metric_points[border_ids])
            accepted = dd <= np.minimum(radius[border_ids], radius[core_ids[near]])
            labels[border_ids[accepted]] = core_labels[near[accepted]]
    weak_ids = np.flatnonzero(labels < 0)
    if len(weak_ids):
        _, weak_labels = connected_components(graph[weak_ids][:, weak_ids], directed=False)
        labels[weak_ids] = count + weak_labels
    return labels, core


def level_rotation(plane: np.ndarray) -> np.ndarray:
    """Orthonormal track-bed frame, unlike shearing z while retaining lateral y."""
    up = np.array([-plane[0], -plane[1], 1.0])
    up /= np.linalg.norm(up)
    forward = np.array([1.0, 0.0, 0.0])
    forward -= up * np.dot(up, forward)
    forward /= np.linalg.norm(forward)
    return np.vstack((forward, np.cross(up, forward), up))


def published_labels(points: np.ndarray, plane: np.ndarray, config: dict, method: str) -> np.ndarray:
    if method == "hdbscan":
        import hdbscan
        return hdbscan.HDBSCAN(**config["hdbscan"]).fit_predict(points)
    if method == "travel":
        import travel_seg
        aligned = (points @ level_rotation(plane).T).astype(np.float32)
        clusterer = travel_seg.ObjectCluster(travel_seg.ObjectClusterConfig(**config["travel_objects"]))
        return clusterer.segment_objects(aligned).astype(np.int32) - 1
    raise ValueError(f"Unknown published segmentation method: {method}")
