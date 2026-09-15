"""Safety-relevant geometry failures: unsupported route and sensor roll."""
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from tunnel_guard.detector import cluster_candidates, load_config
from tunnel_guard.geometry import TrackGeometry


class GeometryFailures(unittest.TestCase):
    def setUp(self):
        self.config = load_config(Path(__file__).resolve().parents[1] / "configs/detector.json")
        x, y = np.meshgrid(np.arange(2., 36., .15), np.arange(-2., 2.01, .15))
        floor = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.7)))
        rails = []
        for side in [-.76, .76]:
            x, y = np.meshgrid(np.arange(2., 36., .10), side + np.array([-.025, 0., .025]))
            rails.append(np.column_stack((x.ravel(), y.ravel(), np.full(x.size, -1.5))))
        self.scene = np.vstack((floor, *rails))

    def test_unsupported_path_cannot_erase_low_object_as_infinite_width_rail(self):
        geometry = TrackGeometry(self.scene, self.config)
        self.assertTrue(geometry.valid)
        obstacle = np.array([[100., 0., -1.41], [100., 0., -1.25]])
        objects = cluster_candidates(obstacle, geometry, self.config)
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0]["path_relation"], "unresolved")
        self.assertAlmostEqual(objects[0]["bbox_min"][2], -1.41)

    def test_roll_does_not_move_adjacent_object_into_vehicle_envelope(self):
        rotation = Rotation.from_euler("x", 10., degrees=True).as_matrix()
        geometry = TrackGeometry(self.scene @ rotation.T, self.config)
        self.assertTrue(geometry.valid)
        point = np.array([[10., 1.50, 0.]]) @ rotation.T
        core, _, _, observed, nominal = geometry.classify(point)
        self.assertTrue(observed[0])
        self.assertFalse(core[0])
        self.assertFalse(nominal[0])


if __name__ == "__main__":
    unittest.main()
