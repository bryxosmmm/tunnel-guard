"""Experimental forward-only motion from a longitudinal wall-return profile.

Method provenance: EhimenNathan/tunnelguard-lct2026, adapt.py, revision
2569926368276a1af6768a39584e54d6f4733312; see docs/FAST_MOTION.md.
This implementation contains only histogram correlation and motion state, not
that project's clutter adaptation, classifier, or complete detector.
"""
from collections import deque

import numpy as np


def validate_profile_motion(config: dict) -> None:
    method = config.get("motion_estimator", "kiss_icp")
    if method not in ("kiss_icp", "longitudinal_profile"):
        raise ValueError("motion_estimator must be kiss_icp or longitudinal_profile")
    if method == "kiss_icp":
        return
    if (config.get("deskew_enabled", False) or config.get("overlap_motion_geometry", False)
            or config.get("background_refit_travel_m", 0) != 0
            or config.get("background_refit_every_frames", 1) != 1):
        raise ValueError("longitudinal_profile requires no deskew/overlap and a fresh geometry fit each scan")
    cfg = config["profile_motion"]
    for key in ("bin_m", "max_gap_s", "max_speed_m_s", "minimum_search_m", "prior_sigma_m",
                "correlation_floor", "normalization_offset"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"profile_motion.{key} must be positive and finite")
    for key in ("forward_m", "absolute_lateral_m", "height_m"):
        bounds = cfg[key]
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError(f"profile_motion.{key} must be two finite increasing bounds")
    for key in ("trend_bins", "history_steps", "speed_min_samples"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"profile_motion.{key} must be a positive integer")
    bins = len(np.arange(*cfg["forward_m"], cfg["bin_m"]))
    kernel = np.asarray(cfg["smoothing_kernel"], dtype=float)
    if (bins < 3 or cfg["trend_bins"] > bins or kernel.ndim != 1 or not 0 < len(kernel) <= bins
            or not np.isfinite(kernel).all() or np.any(kernel < 0) or not np.isclose(kernel.sum(), 1)
            or not np.isfinite(cfg["prior_weight"]) or cfg["prior_weight"] < 0
            or cfg["speed_min_samples"] > cfg["history_steps"]):
        raise ValueError("Invalid profile_motion smoothing, prior, or history configuration")


class LongitudinalProfileMotion:
    def __init__(self, config: dict):
        self.config = config
        self.stations = np.arange(*config["forward_m"], config["bin_m"])
        self.edges = np.r_[self.stations, self.stations[-1] + config["bin_m"]]
        self.trend_kernel = np.ones(config["trend_bins"]) / config["trend_bins"]
        self.smoothing_kernel = np.asarray(config["smoothing_kernel"], dtype=float)
        self.position = 0.0
        self.previous_signature = None
        self.previous_stamp = None
        self.steps = deque(maxlen=config["history_steps"])
        self.speeds = deque(maxlen=config["history_steps"])

    def invalidate(self):
        """Break the correspondence chain without inventing a displacement."""
        self.previous_signature = None
        self.previous_stamp = None
        self.steps.clear()
        self.speeds.clear()

    def update(self, forward, lateral, height, stamp):
        cfg = self.config
        lateral_lo, lateral_hi = cfg["absolute_lateral_m"]
        height_lo, height_hi = cfg["height_m"]
        selected = ((np.abs(lateral) > lateral_lo) & (np.abs(lateral) < lateral_hi)
                    & (height > height_lo) & (height < height_hi)
                    & (forward > self.stations[0]) & (forward < self.stations[-1]))
        counts = np.histogram(forward[selected], bins=self.edges)[0].astype(float)
        trend = np.convolve(counts, self.trend_kernel, mode="same")
        signal = (counts - trend) / np.sqrt(trend + cfg["normalization_offset"])
        signal = np.convolve(signal, self.smoothing_kernel, mode="same")
        norm = np.linalg.norm(signal)
        if norm > 0:
            signal = signal / norm
        dt = -1.0 if self.previous_stamp is None else stamp - self.previous_stamp
        if self.previous_signature is not None and 0 < dt <= cfg["max_gap_s"]:
            if len(self.speeds) >= cfg["speed_min_samples"]:
                prior = float(np.median(self.speeds)) * dt
            else:
                prior = float(np.median(self.steps)) if self.steps else None
            limit = int(min(len(self.stations) // 2,
                            np.ceil(max(cfg["minimum_search_m"], cfg["max_speed_m_s"] * dt) / cfg["bin_m"])))
            correlations = np.empty(limit + 1)
            for offset in range(limit + 1):
                previous = self.previous_signature[offset:]
                current = signal[:len(signal) - offset]
                denominator = max(np.linalg.norm(previous) * np.linalg.norm(current), cfg["correlation_floor"])
                correlations[offset] = float(np.dot(previous, current) / denominator)
            scores = correlations if prior is None else correlations - 0.5 * (
                (np.arange(limit + 1) * cfg["bin_m"] - prior) / cfg["prior_sigma_m"]) ** 2 * cfg["prior_weight"]
            peak = int(np.argmax(scores))
            fraction = 0.0
            if 0 < peak < limit:
                left, middle, right = correlations[peak - 1:peak + 2]
                curvature = left - 2 * middle + right
                if curvature < 0:
                    fraction = 0.5 * (left - right) / curvature
            displacement = cfg["bin_m"] * (peak + fraction)
            self.position += displacement
            self.steps.append(displacement)
            self.speeds.append(displacement / dt)
        else:
            self.steps.clear()
            self.speeds.clear()
        self.previous_signature = signal
        self.previous_stamp = stamp
