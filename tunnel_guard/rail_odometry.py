"""Experimental measured-rail correction; does not observe longitudinal displacement."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class RailOdometry:
    def __init__(self, config):
        self.config = config
        self.plan = config["rail_motion_correction"]
        if self.plan["method"] != "measured_lines_fixed_forward":
            raise ValueError("Unknown rail motion correction method")
        if config.get("deskew_enabled", False):
            raise ValueError("This experiment requires the unchanged deskew-disabled input")
        self.reset()

    def reset(self):
        self.previous = None
        self.corrected_pose = None
        self.map_correction = None

    def selection(self, points, geometry, *, broad=False):
        if not geometry.valid:
            return np.zeros(len(points), dtype=bool), np.zeros(len(points))
        anchors = geometry.rail_anchors
        x = points[:, 0]
        center = np.interp(x, anchors[:, 0], anchors[:, 1])
        gauge = np.interp(x, anchors[:, 0], anchors[:, 2])
        bed, uncertainty = geometry.ground(points)
        height = points[:, 2] - bed
        lo, hi = self.config["rail_height_bounds_m"]
        width = self.config["rail_half_width_m"] * (1.0 if broad else 0.5)
        lateral = points[:, 1] - center
        selected = ((x >= anchors[0, 0]) & (x <= anchors[-1, 0])
                    & (x <= self.plan["max_rail_range_m"])
                    & (height > lo) & (height < hi)
                    & (uncertainty < self.config["ground_max_uncertainty_m"])
                    & (np.abs(np.abs(lateral) - gauge / 2) < width))
        return selected, lateral

    def rails(self, points, geometry):
        selected, lateral = self.selection(points, geometry)
        sides = []
        for sign in (-1, 1):
            cloud = points[selected & (sign * lateral > 0)]
            if len(cloud):
                cloud = cloud[np.argsort(cloud[:, 0], kind="stable")]
                step = max(1, int(np.ceil(len(cloud) / self.plan["max_points_per_rail"])))
                cloud = cloud[::step]
            sides.append(cloud)
        return sides

    def lines(self, cloud):
        k = self.plan["line_neighbors"]
        _, neighbors = cKDTree(cloud).query(cloud, k=k, workers=1)
        neighborhoods = cloud[neighbors]
        centers = neighborhoods.mean(axis=1)
        centered = neighborhoods - centers[:, None, :]
        values, vectors = np.linalg.eigh(np.einsum("nki,nkj->nij", centered, centered) / k)
        good = values[:, 2] > self.plan["line_eigenratio"] * np.maximum(values[:, 1], 1e-12)
        centers, tangents = centers[good], vectors[good, :, 2]
        projection = np.eye(3)[None, :, :] - np.einsum("ni,nj->nij", tangents, tangents)
        return centers, projection

    def fit(self, current, previous, raw):
        info = {"accepted": False, "reason": "insufficient_measured_rail_support"}
        minimum = self.plan["min_points_per_rail"]
        if any(len(p) < minimum or np.ptp(p[:, 0]) < self.config["rail_min_span_m"]
               for p in current + previous):
            return raw, info
        pairs = []
        for source, target in zip(current, previous):
            centers, projectors = self.lines(target)
            if len(centers) < minimum:
                return raw, info
            aligned = source @ raw[:3, :3].T + raw[:3, 3]
            distances, indices = cKDTree(centers).query(aligned, workers=1)
            keep = distances < self.plan["max_match_distance_m"]
            if np.count_nonzero(keep) < minimum:
                return raw, info
            pairs.append((source[keep], centers[indices[keep]], projectors[indices[keep]]))
        source = np.concatenate([p[0] for p in pairs])
        centers = np.concatenate([p[1] for p in pairs])
        projectors = np.concatenate([p[2] for p in pairs])
        # Fix correspondences for the bounded local solve; no curve endpoints act as x anchors.
        residual = lambda transform: np.einsum("nij,nj->ni", projectors,
            source @ transform[:3, :3].T + transform[:3, 3] - centers)
        raw_error = float(np.mean(np.sum(residual(raw)**2, axis=1)))
        rail_strengths, rail_axes = np.linalg.eigh(projectors.mean(axis=0))
        info.update(matched_points=len(source), rail_translation_eigenvalues=rail_strengths.tolist(),
                    rail_weak_direction=rail_axes[:, 0].tolist(), raw_rail_mse_m2=raw_error,
                    forward_translation_policy="raw_ICP_retained_not_observed_by_rail_constraint")
        proposed = raw.copy()
        lever = max(float(np.sqrt(np.mean(np.sum(source**2, axis=1)))), 1.0)
        for _ in range(self.plan["iterations"]):
            rotated = source @ proposed[:3, :3].T
            error = residual(proposed)
            jacobian = np.zeros((len(source), 3, 5))
            jacobian[:, 1, 0] = 1.0
            jacobian[:, 2, 1] = 1.0
            x, y, z = rotated.T / lever
            jacobian[:, 0, 3], jacobian[:, 0, 4] = z, -y
            jacobian[:, 1, 2], jacobian[:, 1, 4] = -z, x
            jacobian[:, 2, 2], jacobian[:, 2, 3] = y, -x
            jacobian = np.einsum("nij,njk->nik", projectors, jacobian)
            weights = np.sqrt(np.minimum(1.0, self.plan["robust_scale_m"] /
                                         np.maximum(np.linalg.norm(error, axis=1), 1e-12)))
            delta, _, rank, singular = np.linalg.lstsq(
                (jacobian * weights[:, None, None]).reshape(-1, 5),
                -(error * weights[:, None]).reshape(-1), rcond=self.plan["svd_relative_cutoff"])
            if rank < 5:
                info["reason"] = "rail_correction_rank_deficient"
                return raw, info
            proposed[1:3, 3] += delta[:2]
            proposed[:3, :3] = Rotation.from_rotvec(delta[2:] / lever).as_matrix() @ proposed[:3, :3]
        translation = float(np.linalg.norm(proposed[1:3, 3] - raw[1:3, 3]))
        angle = float(Rotation.from_matrix(proposed[:3, :3] @ raw[:3, :3].T).magnitude())
        new_error = float(np.mean(np.sum(residual(proposed)**2, axis=1)))
        info.update(proposed_rail_mse_m2=new_error, correction_translation_m=translation,
                    correction_rotation_rad=angle, forward_translation_change_m=float(proposed[0, 3] - raw[0, 3]))
        if (translation > self.plan["max_correction_translation_m"]
                or angle > self.plan["max_correction_rotation_rad"]):
            info["reason"] = "outside_declared_trust_region"
            return raw, info
        if not new_error < raw_error:
            info["reason"] = "rail_fit_not_improved"
            return raw, info
        info["reason"] = "requires_nonrail_validation"
        return proposed, info

    def update(self, geometry, reduced, source, raw_pose, quality):
        raw_alignment = {key: quality.get(key) for key in ("valid", "median_residual_m", "overlap")}
        rails = self.rails(reduced, geometry)
        rail_mask, _ = self.selection(source, geometry, broad=True)
        nonrail = source[~rail_mask]
        record = {"pose": raw_pose.copy(), "source": source, "nonrail": nonrail, "rails": rails}
        info = {"accepted": False, "reason": "first_frame", "pose_source": "raw_ICP"}
        if self.previous is None:
            corrected = raw_pose.copy()
        else:
            previous = self.previous
            raw = np.linalg.inv(previous["pose"]) @ raw_pose
            proposed = raw
            if quality["valid"]:
                proposed, info = self.fit(rails, previous["rails"], raw)
            else:
                info["reason"] = "raw_registration_not_accepted"
            if info["reason"] == "requires_nonrail_validation":
                # Disjoint from the rail fit, but NOT an independent sensor or ground truth.
                min_validation = self.plan["min_nonrail_points"]
                if min(len(nonrail), len(previous["nonrail"])) < min_validation:
                    info["reason"] = "insufficient_nonrail_validation"
                else:
                    def score(points, tree, transform):
                        aligned = points @ transform[:3, :3].T + transform[:3, 3]
                        distances, _ = tree.query(aligned, workers=1)
                        return float(np.median(distances)), float(np.mean(distances < 2 * self.config["odometry_max_residual_m"]))
                    nonrail_tree = cKDTree(previous["nonrail"])
                    before = score(nonrail, nonrail_tree, raw)
                    after = score(nonrail, nonrail_tree, proposed)
                    all_after = score(source, cKDTree(previous["source"]), proposed)
                    info.update(nonrail_before={"median_m": before[0], "overlap": before[1]},
                                nonrail_after={"median_m": after[0], "overlap": after[1]},
                                all_after={"median_m": all_after[0], "overlap": all_after[1]})
                    if (after[0] <= before[0] and after[1] >= before[1]
                            and all_after[0] <= quality["median_residual_m"]
                            and all_after[1] >= quality["overlap"]):
                        info.update(accepted=True, reason="rail_fit_and_nonrail_validation_accepted")
                        quality.update(median_residual_m=all_after[0], overlap=all_after[1],
                            position_sigma_m=max(all_after[0], self.config["tracking_pose_sigma_m"]))
                    else:
                        info["reason"] = "independent_alignment_gate_rejected"
            accepted = info["accepted"]
            relative = proposed if accepted else raw
            if accepted:
                corrected = self.corrected_pose @ proposed
                self.map_correction = corrected @ np.linalg.inv(raw_pose)
            else:
                corrected = (raw_pose.copy() if self.map_correction is None
                             else self.map_correction @ raw_pose)
            info.update(raw_relative=raw.tolist(), used_relative=relative.tolist(),
                        pose_source="accumulated_checked_relative_transforms")
        self.previous, self.corrected_pose = record, corrected
        info["raw_alignment"] = raw_alignment
        info.update(current_rail_points=[len(p) for p in rails],
                    longitudinal_accuracy="unverified", nominal_gauge_constraint=False)
        return corrected, info
