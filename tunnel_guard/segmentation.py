"""Range-adaptive DBSCAN connectivity and original published comparison backends.

DBSCAN: Ester et al., KDD 1996. Adaptive epsilon/min-support is our adaptation,
not a reproduction claim. TRAVEL and HDBSCAN backends call released code directly.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from . import accelerator
from .geometry import query_workers


def density_labels(points: np.ndarray, config: dict, *,
                   surface_contact: np.ndarray,
                   surface_plane: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, dict[int, dict]]:
    """Dense cores do not join through a thin chain of border points.

    Return instance labels, density-core membership and original split-group bounds.
    Unassigned spatial components remain weak hypotheses; they cannot trigger immediately.
    """
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=bool), {}
    distance = np.linalg.norm(points, axis=1)
    radius = np.clip(config["density_radius_m"] + config["density_angular_radius_rad"] * distance,
                     config["density_radius_m"], config["cluster_max_radius_m"])
    metric_points = points * np.array([1., 1., config["density_vertical_scale"]])
    # Mutual-radius edges: identical selection in both branches, see accelerator.
    module = accelerator.native(config)
    graph, degree = accelerator.density_graph(metric_points, radius, config, module)
    configured = config.get("query_workers", -1)
    required = np.maximum(config["density_min_far"], np.ceil(config["density_min_near"] *
                          np.minimum(1., (config["density_reference_range_m"] / np.maximum(distance, 1.))**2)))
    core = degree >= required
    labels = np.full(n, -1, dtype=np.int32)
    core_ids = np.flatnonzero(core)
    count = 0
    if len(core_ids):
        core_labels = accelerator.component_labels(graph, core, module)
        count = int(core_labels.max()) + 1
        labels[core_ids] = core_labels
        border_ids = np.flatnonzero(~core)
        if len(border_ids):
            dd, near = cKDTree(metric_points[core_ids]).query(
                metric_points[border_ids], workers=query_workers(len(border_ids), configured))
            accepted = dd <= np.minimum(radius[border_ids], radius[core_ids[near]])
            labels[border_ids[accepted]] = core_labels[near[accepted]]
    # Only the nodes still unlabelled go on to the weak pass: the border
    # assignment above may already have claimed some non-core points.
    weak_ids = np.flatnonzero(labels < 0)
    if len(weak_ids):
        weak_mask = np.zeros(len(core), dtype=bool)
        weak_mask[weak_ids] = True
        weak_labels = accelerator.component_labels(graph, weak_mask, module)
        labels[weak_ids] = count + weak_labels
    groups = _separate_running_surface(points, metric_points, labels, core, graph, radius, required,
                                      surface_contact, surface_plane, config)
    return labels, core, groups


def _separate_running_surface(points, metric, labels, core, graph, radius, required, contact, plane, config):
    """Partition connectivity, never delete support or certify the surface as clear.

    A split needs an independently dense upright body above the contact interval,
    meeting the existing geometric-presence requirements. Broad surfaces and
    ambiguous low hypotheses retain their original grouping. Every separated
    part still needs its own density and object-admission support. Overlapping
    bed-plane footprints remain together; unsupported fragments are reattached.
    No support is removed and the original density evidence remains unchanged.
    """
    groups = {}
    if not np.any(contact):
        return groups
    rotation = level_rotation(plane)
    footprint = points @ rotation[:2].T
    elevated = ~contact
    elevated_core = core & elevated & (
        np.asarray(graph @ elevated.astype(np.int32)).ravel() + 1 >= required)
    order = np.argsort(labels, kind="stable")
    boundaries = np.r_[0, np.flatnonzero(np.diff(labels[order])) + 1, len(labels)]
    # One label's nodes are a contiguous run of `order`, so the two conditions that skip a
    # label are evaluated for every label at once. Measured, this loop runs 1 476 times per
    # frame and only about 43 of those labels have anything to split: the per-label gathers
    # and reductions were almost entirely overhead. The predicates are the same ones, on the
    # same runs - the counts are integers and the height span is max minus min.
    starts = boundaries[:-1]
    has_contact = np.add.reduceat(contact[order].astype(np.int64), starts) > 0
    has_elevated = np.add.reduceat(elevated_core[order].astype(np.int64), starts) > 0
    core_counts = np.add.reduceat(core[order].astype(np.int64), starts)
    height_low = np.minimum.reduceat(points[order, 2].astype(float, copy=False), starts)
    height_high = np.maximum.reduceat(points[order, 2].astype(float, copy=False), starts)
    # A split needs an ANCHORED body: at least `immediate_min_voxels` points of one child whose
    # own in-child degree reaches `required` and whose height span reaches
    # `immediate_min_height_m`. A child of the elevated core is a subset of the label's
    # elevated-core points, so a label holding fewer of those than a body needs, or whose
    # elevated core does not span that height, cannot produce one. The two tests are strictly
    # necessary - a superset spans at least as far as its subset - and they are taken from
    # reductions over the same runs, so a label excluded here is one the per-label partition
    # below would have discarded anyway, after paying for its subgraph, components and query.
    elevated_core_counts = np.add.reduceat(elevated_core[order].astype(np.int64), starts)
    elevated_height = np.where(elevated_core[order], points[order, 2], -np.inf)
    # A label with no elevated core carries the sentinel at both ends; the span of those is a
    # NaN, which fails the comparison below exactly as an impossible span should. The sentinel
    # is deliberate, so the subtraction's invalid-value warning is suppressed rather than
    # avoided by inventing a height.
    with np.errstate(invalid="ignore"):
        elevated_low = np.minimum.reduceat(elevated_height, starts)
        elevated_high = np.maximum.reduceat(elevated_height, starts)
        elevated_span = elevated_high - elevated_low
    candidates = np.flatnonzero(has_contact & has_elevated
                               & (core_counts >= config["immediate_min_voxels"])
                               & ((height_high - height_low) >= config["immediate_min_height_m"])
                               & (elevated_core_counts >= config["immediate_min_voxels"])
                               & (elevated_span >= config["immediate_min_height_m"]))
    next_label = int(labels.max()) + 1
    native = accelerator.native(config)
    # The elevated core is labelled once for the whole scan, and each parent's in-child neighbour
    # counts come from one native pass over the graph's own edges, instead of inducing a subgraph
    # per parent and per child: measured on doubleT_obstacle the split made 54 induced-subgraph
    # calls per scan for 12 candidates, 11.4 ms of the stage's 28.6 ms. A parent's children ARE the
    # elevated-core components inside it, because two elevated-core points in different parents
    # belong to different density components and so cannot share an edge. A parent's numbering is
    # recovered from the global labels by ranking its children by their first row, which is the
    # order a per-parent labelling assigns (verified against the per-parent call on 605 parents
    # drawn from 300 random graphs by `check_global_labels.py`).
    child_of_point = np.full(len(points), -1, dtype=np.int64)
    group_of_point = np.full(len(points), -1, dtype=np.int64)
    if len(candidates):
        child_of_point[np.flatnonzero(elevated_core)] = accelerator.component_labels(
            graph, elevated_core, native)
    for index in candidates:
        start, end = int(starts[index]), int(boundaries[index + 1])
        indices = order[start:end]
        core_ids = np.flatnonzero(elevated_core[indices])
        core_rows = indices[core_ids]
        child_values, child_first, child_inverse = np.unique(
            child_of_point[core_rows], return_index=True, return_inverse=True)
        rank = np.empty(len(child_values), dtype=np.int64)
        rank[np.argsort(child_first, kind="stable")] = np.arange(len(child_values))
        core_labels = rank[child_inverse]
        count = len(child_values)
        children = np.full(len(indices), -1, dtype=np.int32)
        children[core_ids] = core_labels
        borders = np.flatnonzero(~elevated_core[indices])
        dd, near = cKDTree(metric[core_rows]).query(metric[indices[borders]], workers=1)
        accepted = dd <= np.minimum(radius[indices[borders]], radius[core_rows[near]])
        children[borders[accepted]] = core_labels[near[accepted]]
        remainder = np.flatnonzero(children < 0)
        if len(remainder):
            _, remaining_labels = connected_components(
                accelerator.induced_subgraph(graph, indices[remainder], native), directed=False)
            children[remainder] = count + remaining_labels
        if count == 1 and not len(remainder):
            continue
        group_of_point[indices] = children
        child_degree, child_marked_degree = accelerator.group_degrees(
            graph, indices, group_of_point, elevated, native)
        group_of_point[indices] = -1
        supported = []
        anchored = False
        for child in np.unique(children):
            members = np.flatnonzero(children == child)
            if len(members) < config["weak_min_voxels"]:
                continue
            child_indices = indices[members]
            extent = np.ptp(points[child_indices].astype(float, copy=False), axis=0)
            if extent.max() < config["cluster_min_extent_m"]:
                continue
            own_degree = child_degree[members]
            if np.any(own_degree >= required[child_indices]):
                supported.append(child)
            if not anchored and child < count:
                # A core child holds its assigned borders too, and those are not elevated, so
                # the anchored body is measured with the elevated-only count, exactly as the
                # per-child `child_graph @ elevated` produced it.
                above = elevated[child_indices]
                body = child_indices[above & (child_marked_degree[members] >= required[child_indices])]
                if (len(body) >= config["immediate_min_voxels"]
                        and np.ptp(points[body, 2].astype(float, copy=False)) >= config["immediate_min_height_m"]):
                    vertical_span = np.ptp(points[body] @ rotation[2])
                    anchored = vertical_span > np.ptp(footprint[body], axis=0).max()
        if not anchored or len(supported) < 2:
            continue
        unassigned = np.flatnonzero(~np.isin(children, supported))
        if len(unassigned):
            seeds = np.flatnonzero(np.isin(children, supported))
            # The parent's own subgraph is induced only where a weighted search needs it -
            # measured, that is about one candidate in 40.
            local_graph = accelerator.induced_subgraph(graph, indices, native)
            weighted = local_graph.astype(float)
            source = np.repeat(np.arange(len(indices)), np.diff(weighted.indptr))
            weighted.data[:] = np.linalg.norm(
                metric[indices[source]] - metric[indices[weighted.indices]], axis=1)
            _, _, nearest = dijkstra(weighted, directed=False, indices=seeds,
                                      min_only=True, return_predecessors=True)
            if np.any(nearest[unassigned] < 0):
                raise ValueError("Disconnected original density component")
            children[unassigned] = children[nearest[unassigned]]
        changed = True
        while changed:
            changed = False
            identities = np.unique(children)
            bounds = [(footprint[indices[children == child]].min(axis=0),
                       footprint[indices[children == child]].max(axis=0)) for child in identities]
            for i in range(len(identities)):
                for j in range(i + 1, len(identities)):
                    if np.all(np.maximum(bounds[i][0], bounds[j][0])
                              <= np.minimum(bounds[i][1], bounds[j][1])):
                        children[children == identities[j]] = identities[i]
                        changed = True
                        break
                if changed:
                    break
        child_ids, counts = np.unique(children, return_counts=True)
        if len(child_ids) == 1:
            continue
        largest = child_ids[np.argmax(counts)]
        parent = int(labels[indices[0]])
        original = points[indices]
        low = original.min(axis=0).astype(float)
        high = original.max(axis=0).astype(float)
        reference = {"parent": parent, "center": ((low + high) / 2).tolist(),
                     "extent_m": (high - low).tolist(),
                     "bbox_min": low.tolist(), "bbox_max": high.tolist()}
        groups[parent] = reference
        for child in child_ids:
            if child != largest:
                labels[indices[children == child]] = next_label
                groups[next_label] = reference
                next_label += 1
    return groups


def level_rotation(plane: np.ndarray) -> np.ndarray:
    """Orthonormal track-bed frame, unlike shearing z while retaining lateral y."""
    up = np.array([-plane[0], -plane[1], 1.0])
    up /= np.linalg.norm(up)
    forward = np.array([1.0, 0.0, 0.0])
    forward -= up * np.dot(up, forward)
    forward /= np.linalg.norm(forward)
    return np.vstack((forward, np.cross(up, forward), up))


def published_labels(points: np.ndarray, plane: np.ndarray | None, config: dict, method: str) -> np.ndarray:
    if method == "hdbscan":
        import hdbscan
        return hdbscan.HDBSCAN(**config["hdbscan"]).fit_predict(points)
    if method == "travel":
        import travel_seg
        # Without a measured plane, retain the configured processing frame;
        # do not fabricate a measured bed normal merely to segment the cloud.
        aligned = (points if plane is None else points @ level_rotation(plane).T).astype(np.float32)
        clusterer = travel_seg.ObjectCluster(travel_seg.ObjectClusterConfig(**config["travel_objects"]))
        return clusterer.segment_objects(aligned).astype(np.int32) - 1
    raise ValueError(f"Unknown published segmentation method: {method}")
