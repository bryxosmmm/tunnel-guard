"""Regression tests for failure-prone data, temporal, and evaluation contracts."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tunnel_guard.detector import Detector, load_config
from tunnel_guard.evaluate import evaluate_frames
from tunnel_guard.io import decode_cloud
from tunnel_guard.segmentation import density_labels


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "detector.json"


class Contracts(unittest.TestCase):
    def test_padded_big_endian_cloud_preserves_coordinates(self):
        data = bytearray(64)
        records = np.ndarray((2, 2), dtype=np.dtype({"names": ["x", "y", "z"],
            "formats": [">f4"] * 3, "offsets": [0, 4, 8], "itemsize": 12}),
            buffer=data, strides=(32, 12))
        records["x"] = [[1, 2], [3, np.nan]]
        records["y"] = [[4, 5], [6, 7]]
        records["z"] = [[8, 9], [10, 11]]
        message = SimpleNamespace(fields=[SimpleNamespace(name=n, count=1, datatype=7, offset=i*4)
            for i, n in enumerate(("x", "y", "z"))], is_bigendian=True, width=2, height=2,
            row_step=32, point_step=12, data=data)
        points, times, invalid, _ = decode_cloud(message, np.eye(3), np.array([0, 0, 1]))
        np.testing.assert_allclose(points, [[1, 4, 9], [2, 5, 10], [3, 6, 11]])
        self.assertEqual(invalid, 1)
        self.assertEqual(times.size, 0)
        message.data = data[:-1]
        with self.assertRaises(ValueError):
            decode_cloud(message, np.eye(3), np.zeros(3))

    def test_temporal_confirmation_counts_frames_and_one_to_one_matches(self):
        detector = Detector(load_config(CONFIG))
        obj = {"center": [10., 0., 0.], "extent_m": [.2, .1, .1], "immediate": False,
               "_support_points": np.array([[9.9, 0, 0], [10, .1, .1], [10.1, 0, 0]])}
        detector.frame_number = 1
        first = [copy.deepcopy(obj), copy.deepcopy(obj)]
        detector._associate(first, np.eye(4), 0., True)
        self.assertTrue(all(not o["confirmed"] for o in first))
        self.assertNotEqual(first[0]["track_id"], first[1]["track_id"])
        detector.frame_number = 2
        second = [copy.deepcopy(obj)]
        detector._associate(second, np.eye(4), .1, True)
        self.assertTrue(second[0]["confirmed"])
        detector.frame_number = 3
        rejected = [copy.deepcopy(obj)]
        detector._associate(rejected, np.eye(4), .2, False)
        self.assertFalse(rejected[0]["confirmed"])

    def test_empty_input_never_means_clear_and_time_reversal_rejected(self):
        detector = Detector(load_config(CONFIG))
        result = detector.process(np.empty((0, 3)), 1.)
        self.assertEqual(result["status"], "unknown")
        self.assertIsNone(result["nearest_obstacle_m"])
        with self.assertRaises(ValueError):
            detector.process(np.empty((0, 3)), 1.)

    def test_nonexhaustive_labels_cannot_manufacture_precision(self):
        box = {"bbox_min": [10, 0, 0], "bbox_max": [11, 1, 1]}
        panel = {"label_status": "provisional_geometry", "prediction_scope": "collision_hazards",
                 "minimum_iou": .25, "frames": [{"bag": "b", "frame": 0, "exhaustive": False,
                 "objects": [box | {"event_id": "object"}]}]}
        predictions = {("b", 0): {"status": "obstacle", "objects": [box | {
            "confirmed": True, "distance_m": 10, "path_relation": "intersecting"}]}}
        result = evaluate_frames(predictions, panel)
        self.assertEqual(result["annotated_object_recall"], 1.)
        self.assertIsNone(result["precision_exhaustive_only"])
        self.assertIsNone(result["f1_exhaustive_only"])
        predictions[("b", 0)]["objects"][0]["bbox_min"] = [12, 0, 0]
        predictions[("b", 0)]["objects"][0]["bbox_max"] = [13, 1, 1]
        self.assertEqual(evaluate_frames(predictions, panel)["annotated_object_recall"], 0.)

    def test_thin_border_chain_cannot_join_two_dense_objects(self):
        rng = np.random.default_rng(14)
        first = rng.uniform(-.15, .15, size=(160, 3)) + [10, 0, 0]
        second = rng.uniform(-.15, .15, size=(160, 3)) + [11.6, 0, 0]
        chain = np.column_stack((np.arange(10.15, 11.46, .1), np.zeros(14), np.zeros(14)))
        labels, core = density_labels(np.vstack((first, second, chain)), load_config(CONFIG))
        self.assertTrue(core[:320].all())
        self.assertNotEqual(labels[0], labels[160])
        self.assertEqual(len(set(labels[:160])), 1)
        self.assertEqual(len(set(labels[160:320])), 1)


if __name__ == "__main__":
    unittest.main()
