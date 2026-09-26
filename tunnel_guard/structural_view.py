"""Render one frame's structural verdict so the two conditions can be seen, not inferred.

Left: returns in the (longitudinal, lateral) plane. Right: the same returns in the
cross-section (lateral, height above the running surface) with the reference contour drawn.
Blue is the tunnel's own cross-section, red is interior evidence the claim rests on, orange is
edge-uncertain evidence that is not structural. Diagnosis only: reads a bag, writes one PNG.

    python -m tunnel_guard.structural_view --bag <bag> --frame 300 --output build/view.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tunnel_guard.detector import load_config
from tunnel_guard.geometry import TrackGeometry, voxel_representative_indices
from tunnel_guard.io import iter_bag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/detector.json"))
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = load_config(args.config)
    for scan in iter_bag(args.bag, config, every=1, max_frames=args.frame + 1):
        if scan.index != args.frame:
            continue
        points = scan.points
        radii = np.linalg.norm(points, axis=1)
        points = points[(radii >= config["min_range_m"]) & (radii <= config["max_range_m"])]
        frame = points[(points[:, 0] >= config["min_forward_m"]) & (np.abs(points[:, 1]) < config["context_half_width_m"])]
        reduced = frame[voxel_representative_indices(frame, config["geometry_voxel_m"])]
        geometry = TrackGeometry(reduced, config)
        masks, section = geometry.classify_with_section(reduced, remove_background=False, include_boundary=True)
        _core, _context, _height, observed, interior, boundary = masks
        structural = geometry.structural_mask(reduced, section)
        lateral, running, x = section.lateral, section.running_height, reduced[:, 0]
        claim = interior & observed & ~structural
        doubt = boundary & ~structural & ~claim
        rest = ~(structural | claim | doubt)
        figure, (left, right) = plt.subplots(1, 2, figsize=(20, 8))
        for axis, (px, py, name) in ((left, (x, lateral, "x")), (right, (lateral, running, "lateral"))):
            for mask, colour, label, size in ((rest, "0.8", "not structural, no evidence", 1),
                                              (structural, "#1f77b4", "tunnel cross-section", 2),
                                              (claim, "#d62728", "interior evidence (claim)", 8),
                                              (doubt, "#ff7f0e", "edge-uncertain, not structural", 8)):
                axis.scatter(px[mask], py[mask], s=size, c=colour, label=label, linewidths=0)
            axis.set_xlabel(f"{name} (m)")
            axis.grid(alpha=0.3)
        left.set_ylabel("lateral (m)")
        right.set_ylabel("height above running surface (m)")
        contour = np.asarray(config["envelope_segments_m"], dtype=float)
        heights = np.linspace(contour[0, 0], contour[-1, 1], 200)
        widths = geometry.reference_half_width(heights)
        for sign in (-1, 1):
            right.plot(sign * widths, heights, "k-", linewidth=2, label="reference contour")
        right.set_ylim(-0.6, 4.6)
        right.set_xlim(-4, 4)
        right.legend(loc="upper right", fontsize=8)
        left.legend(loc="upper right", fontsize=8)
        left.set_title(f"{args.bag.name} frame {args.frame}: structural {int(structural.sum())},"
                       f" claim {int(claim.sum())}, edge-uncertain {int(doubt.sum())}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(args.output, dpi=90)
        print(f"wrote {args.output}: structural {int(structural.sum())} claim {int(claim.sum())} doubt {int(doubt.sum())}")
        return


if __name__ == "__main__":
    main()
