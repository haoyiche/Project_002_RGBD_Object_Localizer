import unittest
import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


class TestRGBDStatsIntegration(unittest.TestCase):

    def setUp(self):
        self.depth = np.array([
            [1.0, 0.0, 2.0],
            [2.0, 3.0, 4.0],
        ], dtype=np.float64)

        self.mask = np.array([
            [True, True, False],
            [True, False, True],
        ], dtype=bool)

        self.rgb = np.array([
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        ], dtype=np.uint8)

        self.intrinsics = dict(
            fx=2.0,
            fy=2.0,
            cx=1.0,
            cy=1.0,
        )

    def extract(self, depth=None, mask=None):
        if depth is None:
            depth = self.depth

        if mask is None:
            mask = self.mask

        return mask_to_pointcloud(
            depth,
            self.rgb,
            mask,
            **self.intrinsics,
        )

    def test_rgbd_to_statistics_reference(self):
        points, colors, selected = self.extract()

        expected_mask = np.array([
            [True, False, False],
            [True, False, True],
        ])

        expected_points = np.array([
            [-0.5, -0.5, 1.0],
            [-1.0,  0.0, 2.0],
            [ 2.0,  0.0, 4.0],
        ])

        expected_colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ])

        np.testing.assert_array_equal(
            selected, expected_mask
        )

        np.testing.assert_allclose(
            points, expected_points,
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            colors, expected_colors,
            rtol=0, atol=1e-12
        )

        stats = compute_pointcloud_statistics(points)

        expected_stats = {
            "centroid": [1/6, -1/6, 7/3],
            "min_bound": [-1.0, -0.5, 1.0],
            "max_bound": [2.0, 0.0, 4.0],
            "bbox_center": [0.5, -0.25, 2.5],
            "bbox_extent": [3.0, 0.5, 3.0],
            "median_center": [-0.5, 0.0, 2.0],
        }

        self.assertEqual(
            set(stats.keys()),
            set(expected_stats.keys()),
        )

        for key, expected in expected_stats.items():
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    stats[key], expected,
                    rtol=0, atol=1e-12
                )
                self.assertEqual(
                    stats[key].shape, (3,)
                )
                self.assertEqual(
                    stats[key].dtype, np.float64
                )

    def test_empty_target_mask(self):
        empty_mask = np.zeros_like(
            self.mask, dtype=bool
        )

        points, colors, selected = self.extract(
            mask=empty_mask
        )

        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        self.assertFalse(selected.any())

        with self.assertRaises(ValueError):
            compute_pointcloud_statistics(points)

    def test_all_target_depth_invalid(self):
        invalid_depth = np.zeros_like(self.depth)

        points, colors, selected = self.extract(
            depth=invalid_depth
        )

        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        self.assertFalse(selected.any())

        with self.assertRaises(ValueError):
            compute_pointcloud_statistics(points)


if __name__ == "__main__":
    unittest.main()
