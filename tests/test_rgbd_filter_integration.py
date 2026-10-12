import unittest
import numpy as np

from src.experiments.exp030_rgbd_filter_integration import (
    run_pipeline,
)
from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


class TestRGBDFilterIntegration(unittest.TestCase):

    def setUp(self):
        self.depth = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 5.0],
        ], dtype=np.float64)

        self.mask = np.array([
            [True, True, False],
            [True, True, True],
        ], dtype=bool)

        self.rgb = np.array([
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 0],
            ],
            [
                [0, 0, 255],
                [255, 255, 0],
                [0, 255, 255],
            ],
        ], dtype=np.uint8)

    def test_normal_rgbd_filter_statistics(self):
        result = run_pipeline(
            self.depth, self.rgb, self.mask
        )

        expected_points = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [1.0, 0.5, 5.0],
        ])

        expected_colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])

        expected_selected = np.array([
            [True, True, False],
            [True, True, True],
        ])

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["original_count"], 5)
        self.assertEqual(result["filtered_count"], 4)

        np.testing.assert_array_equal(
            result["selected"], expected_selected
        )

        np.testing.assert_allclose(
            result["points"],
            expected_points,
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            result["colors"],
            expected_colors,
            rtol=0, atol=1e-12
        )

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [2, 2, 2, 2, 0]
        )

        np.testing.assert_array_equal(
            result["keep_mask"],
            [True, True, True, True, False]
        )

        np.testing.assert_array_equal(
            result["kept_indices"],
            [0, 1, 2, 3]
        )

        np.testing.assert_allclose(
            result["filtered_points"],
            expected_points[:4],
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            result["filtered_colors"],
            expected_colors[:4],
            rtol=0, atol=1e-12
        )

        before = compute_pointcloud_statistics(
            result["points"]
        )

        after = result["stats"]

        references = [
            (before["centroid"], [0.24, 0.14, 1.8]),
            (before["bbox_extent"], [1.0, 0.5, 4.0]),
            (after["centroid"], [0.05, 0.05, 1.0]),
            (after["bbox_extent"], [0.1, 0.1, 0.0]),
        ]

        for actual, expected in references:
            np.testing.assert_allclose(
                actual, expected,
                rtol=0, atol=1e-12
            )

    def test_empty_target_mask(self):
        empty_mask = np.zeros_like(
            self.mask, dtype=bool
        )

        result = run_pipeline(
            self.depth, self.rgb, empty_mask
        )

        self.assertEqual(
            result["status"], "EMPTY_MASK"
        )

        self.assertEqual(result["original_count"], 0)
        self.assertEqual(result["filtered_count"], 0)

        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(result["colors"].shape, (0, 3))
        self.assertEqual(
            result["filtered_points"].shape, (0, 3)
        )
        self.assertEqual(
            result["filtered_colors"].shape, (0, 3)
        )

        self.assertEqual(result["keep_mask"].shape, (0,))
        self.assertEqual(
            result["neighbor_counts"].shape, (0,)
        )
        self.assertEqual(
            result["kept_indices"].shape, (0,)
        )

        self.assertFalse(result["selected"].any())
        self.assertIsNone(result["stats"])

    def test_no_valid_target_depth(self):
        invalid_depth = np.zeros_like(self.depth)

        result = run_pipeline(
            invalid_depth, self.rgb, self.mask
        )

        self.assertEqual(
            result["status"], "NO_VALID_DEPTH"
        )

        self.assertTrue(self.mask.any())
        self.assertFalse(result["selected"].any())

        self.assertEqual(result["original_count"], 0)
        self.assertEqual(result["filtered_count"], 0)
        self.assertEqual(result["points"].shape, (0, 3))
        self.assertEqual(
            result["filtered_points"].shape, (0, 3)
        )

        self.assertIsNone(result["stats"])

    def test_all_filtered_by_strict_threshold(self):
        result = run_pipeline(
            self.depth,
            self.rgb,
            self.mask,
            radius=0.11,
            min_neighbors=3,
        )

        self.assertEqual(
            result["status"], "ALL_FILTERED"
        )

        self.assertEqual(result["original_count"], 5)
        self.assertEqual(result["filtered_count"], 0)

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [2, 2, 2, 2, 0]
        )

        np.testing.assert_array_equal(
            result["keep_mask"],
            [False, False, False, False, False]
        )

        self.assertEqual(
            result["filtered_points"].shape, (0, 3)
        )

        self.assertEqual(
            result["filtered_colors"].shape, (0, 3)
        )

        self.assertIsNone(result["stats"])

    def test_single_survivor_low_confidence(self):
        sparse_mask = np.array([
            [True, True, False],
            [True, False, False],
        ], dtype=bool)

        result = run_pipeline(
            self.depth,
            self.rgb,
            sparse_mask,
            radius=0.11,
            min_neighbors=2,
        )

        self.assertEqual(
            result["status"], "LOW_CONFIDENCE"
        )

        self.assertEqual(result["original_count"], 3)
        self.assertEqual(result["filtered_count"], 1)

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [2, 1, 1]
        )

        np.testing.assert_array_equal(
            result["kept_indices"], [0]
        )

        np.testing.assert_allclose(
            result["filtered_points"],
            [[0.0, 0.0, 1.0]],
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            result["filtered_colors"],
            [[1.0, 0.0, 0.0]],
            rtol=0, atol=1e-12
        )

        self.assertIsNotNone(result["stats"])

        np.testing.assert_allclose(
            result["stats"]["centroid"],
            [0.0, 0.0, 1.0],
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            result["stats"]["bbox_extent"],
            [0.0, 0.0, 0.0],
            rtol=0, atol=1e-12
        )

    def test_nonprefix_indices_preserve_rgb_alignment(self):
        """
        The outlier is in the MIDDLE of point order.
        Correct kept indices must be [0, 1, 3, 4].
        """

        depth = np.array([
            [1.0, 1.0, 5.0],
            [1.0, 1.0, 0.0],
        ], dtype=np.float64)

        mask = np.array([
            [True, True, True],
            [True, True, False],
        ], dtype=bool)

        rgb = np.array([
            [
                [255, 0, 0],      # Red
                [0, 255, 0],      # Green
                [0, 0, 255],      # Blue outlier
            ],
            [
                [255, 255, 0],    # Yellow
                [0, 255, 255],    # Cyan
                [255, 0, 255],    # Unselected
            ],
        ], dtype=np.uint8)

        result = run_pipeline(
            depth, rgb, mask,
            radius=0.11,
            min_neighbors=2,
        )

        expected_points = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.1, 1.0],
        ])

        expected_colors = np.array([
            [1.0, 0.0, 0.0],   # Red
            [0.0, 1.0, 0.0],   # Green
            [1.0, 1.0, 0.0],   # Yellow
            [0.0, 1.0, 1.0],   # Cyan
        ])

        self.assertEqual(result["status"], "SUCCESS")

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [2, 2, 0, 2, 2]
        )

        np.testing.assert_array_equal(
            result["keep_mask"],
            [True, True, False, True, True]
        )

        np.testing.assert_array_equal(
            result["kept_indices"],
            [0, 1, 3, 4]
        )

        np.testing.assert_allclose(
            result["filtered_points"],
            expected_points,
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            result["filtered_colors"],
            expected_colors,
            rtol=0, atol=1e-12
        )

        # Demonstrate why taking the first M colors
        # would silently corrupt RGB-XYZ alignment.
        self.assertFalse(
            np.array_equal(
                result["filtered_colors"],
                result["colors"][:4],
            )
        )

        np.testing.assert_allclose(
            result["stats"]["centroid"],
            [0.05, 0.05, 1.0],
            rtol=0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
