import unittest
import numpy as np

from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


class TestPointcloudStatistics(unittest.TestCase):

    def test_independent_geometry_reference(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
        ])

        result = compute_pointcloud_statistics(points)

        expected = {
            "centroid": [2/3, 2/3, 1.0],
            "min_bound": [0.0, 0.0, 1.0],
            "max_bound": [2.0, 2.0, 1.0],
            "bbox_center": [1.0, 1.0, 1.0],
            "bbox_extent": [2.0, 2.0, 0.0],
            "median_center": [0.0, 0.0, 1.0],
        }

        self.assertEqual(set(result), set(expected))

        for key, value in expected.items():
            np.testing.assert_allclose(
                result[key], value,
                rtol=0, atol=1e-12
            )
            self.assertEqual(result[key].shape, (3,))
            self.assertEqual(result[key].dtype, np.float64)

    def test_single_point(self):
        point = [[3.0, -2.0, 1.5]]

        result = compute_pointcloud_statistics(point)

        for key in [
            "centroid",
            "min_bound",
            "max_bound",
            "bbox_center",
            "median_center",
        ]:
            np.testing.assert_array_equal(
                result[key],
                [3.0, -2.0, 1.5]
            )

        np.testing.assert_array_equal(
            result["bbox_extent"],
            [0.0, 0.0, 0.0]
        )

    def test_accept_integer_list(self):
        result = compute_pointcloud_statistics([
            [0, 0, 1],
            [2, 2, 3],
        ])

        np.testing.assert_array_equal(
            result["centroid"], [1.0, 1.0, 2.0]
        )

        for value in result.values():
            self.assertEqual(value.dtype, np.float64)

    def test_independent_axis_extrema(self):
        points = np.array([
            [-4.0,  1.0,  3.0],
            [ 2.0, -5.0,  8.0],
            [ 1.0,  4.0, -2.0],
        ])

        result = compute_pointcloud_statistics(points)

        expected = {
            "centroid": [-1/3, 0.0, 3.0],
            "min_bound": [-4.0, -5.0, -2.0],
            "max_bound": [2.0, 4.0, 8.0],
            "bbox_center": [-1.0, -0.5, 3.0],
            "bbox_extent": [6.0, 9.0, 10.0],
            "median_center": [1.0, 1.0, 3.0],
        }

        for key, value in expected.items():
            np.testing.assert_allclose(
                result[key], value,
                rtol=0, atol=1e-12
            )

    def test_outlier_sensitivity(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 2.0, 1.0],
            [1.0, 1.0, 10.0],
        ])

        result = compute_pointcloud_statistics(points)

        np.testing.assert_allclose(
            result["centroid"], [1.0, 1.0, 2.8]
        )
        np.testing.assert_allclose(
            result["bbox_center"], [1.0, 1.0, 5.5]
        )
        np.testing.assert_allclose(
            result["bbox_extent"], [2.0, 2.0, 9.0]
        )
        np.testing.assert_allclose(
            result["median_center"], [1.0, 1.0, 1.0]
        )

    def test_translation_invariance(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [2.0, 2.0, 1.0],
        ])

        translation = np.array([3.0, -2.0, 0.5])

        old = compute_pointcloud_statistics(points)
        new = compute_pointcloud_statistics(
            points + translation
        )

        for key in [
            "centroid",
            "min_bound",
            "max_bound",
            "bbox_center",
            "median_center",
        ]:
            np.testing.assert_allclose(
                new[key] - old[key],
                translation,
                rtol=0, atol=1e-12
            )

        np.testing.assert_allclose(
            new["bbox_extent"],
            old["bbox_extent"],
            rtol=0, atol=1e-12
        )

    def test_reject_empty_cloud(self):
        with self.assertRaises(ValueError):
            compute_pointcloud_statistics(
                np.empty((0, 3))
            )

    def test_reject_wrong_shapes(self):
        cases = [
            np.ones((4, 2)),
            np.ones((4, 4)),
            np.ones((3,)),
            np.ones((2, 2, 3)),
        ]

        for points in cases:
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    compute_pointcloud_statistics(points)

    def test_reject_nonfinite_coordinates(self):
        for bad in [np.nan, np.inf, -np.inf]:
            with self.subTest(bad=str(bad)):
                points = [
                    [0.0, 0.0, 1.0],
                    [2.0, bad, 1.0],
                ]

                with self.assertRaises(ValueError):
                    compute_pointcloud_statistics(points)

    def test_reject_nonnumeric_and_preserve_input(self):
        with self.assertRaises(ValueError):
            compute_pointcloud_statistics([
                ["bad", 0, 1]
            ])

        points = np.array([
            [0.0, 0.0, 1.0],
            [2.0, 2.0, 3.0],
        ])

        original = points.copy()

        compute_pointcloud_statistics(points)

        np.testing.assert_array_equal(points, original)


if __name__ == "__main__":
    unittest.main()
