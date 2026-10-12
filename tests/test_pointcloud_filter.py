import math
import unittest

import numpy as np

from src.geometry.pointcloud_filter import (
    radius_outlier_filter,
)


def independent_neighbor_counts(points, radius):
    """
    Independent reference using explicit Python loops.
    Does not use KD-Tree or the production filter.
    """
    counts = []

    for i in range(len(points)):
        count = 0

        for j in range(len(points)):
            if i == j:
                continue

            distance = math.dist(points[i], points[j])

            if distance <= radius:
                count += 1

        counts.append(count)

    return np.asarray(counts, dtype=np.int64)


class TestPointcloudFilter(unittest.TestCase):

    def setUp(self):
        self.points = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.0, 0.1, 1.0],
            [0.1, 0.1, 1.0],
            [1.0, 1.0, 5.0],
        ], dtype=np.float64)

    def test_five_point_independent_references(self):
        cases = [
            (0.11, 2, [2, 2, 2, 2, 0], [0, 1, 2, 3]),
            (0.15, 2, [3, 3, 3, 3, 0], [0, 1, 2, 3]),
            (0.15, 3, [3, 3, 3, 3, 0], [0, 1, 2, 3]),
            (0.15, 4, [3, 3, 3, 3, 0], []),
            (0.09, 2, [0, 0, 0, 0, 0], []),
        ]

        for radius, k, expected_counts, expected_indices in cases:
            with self.subTest(radius=radius, k=k):
                result = radius_outlier_filter(
                    self.points, radius, k
                )

                np.testing.assert_array_equal(
                    result["neighbor_counts"],
                    expected_counts,
                )

                np.testing.assert_array_equal(
                    result["kept_indices"],
                    expected_indices,
                )

                np.testing.assert_array_equal(
                    result["filtered_points"],
                    self.points[expected_indices],
                )

    def test_inclusive_radius_boundary(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [3.0, 3.0, 1.0],
        ])

        result = radius_outlier_filter(
            points, radius=1.0, min_neighbors=2
        )

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [2, 1, 1, 0],
        )

        np.testing.assert_array_equal(
            result["keep_mask"],
            [True, False, False, False],
        )

        np.testing.assert_array_equal(
            result["kept_indices"], [0]
        )

    def test_duplicate_coordinates(self):
        points = [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
        ]

        result = radius_outlier_filter(
            points, 0.1, 1
        )

        np.testing.assert_array_equal(
            result["neighbor_counts"],
            [1, 1, 0],
        )

        np.testing.assert_array_equal(
            result["kept_indices"],
            [0, 1],
        )

    def test_empty_input_contract(self):
        result = radius_outlier_filter(
            np.empty((0, 3)), 0.1, 2
        )

        self.assertEqual(
            set(result),
            {
                "filtered_points",
                "keep_mask",
                "neighbor_counts",
                "kept_indices",
            },
        )

        self.assertEqual(
            result["filtered_points"].shape, (0, 3)
        )
        self.assertEqual(
            result["keep_mask"].shape, (0,)
        )
        self.assertEqual(
            result["neighbor_counts"].shape, (0,)
        )
        self.assertEqual(
            result["kept_indices"].shape, (0,)
        )

        self.assertEqual(
            result["filtered_points"].dtype,
            np.float64,
        )
        self.assertEqual(
            result["keep_mask"].dtype,
            np.bool_,
        )
        self.assertTrue(
            np.issubdtype(
                result["neighbor_counts"].dtype,
                np.integer,
            )
        )
        self.assertTrue(
            np.issubdtype(
                result["kept_indices"].dtype,
                np.integer,
            )
        )

    def test_single_point_removed(self):
        result = radius_outlier_filter(
            [[0.0, 0.0, 1.0]],
            0.1,
            1,
        )

        np.testing.assert_array_equal(
            result["neighbor_counts"], [0]
        )
        np.testing.assert_array_equal(
            result["keep_mask"], [False]
        )
        self.assertEqual(
            result["filtered_points"].shape, (0, 3)
        )

    def test_one_pass_not_iterative(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ])

        first = radius_outlier_filter(
            points, 0.11, 2
        )

        np.testing.assert_array_equal(
            first["neighbor_counts"],
            [1, 2, 1],
        )
        np.testing.assert_array_equal(
            first["kept_indices"],
            [1],
        )

        # Second call is a NEW filtering operation.
        second = radius_outlier_filter(
            first["filtered_points"], 0.11, 2
        )

        self.assertEqual(
            second["filtered_points"].shape, (0, 3)
        )

    def test_integer_input_and_no_mutation(self):
        integer_points = [
            [0, 0, 1],
            [1, 0, 1],
        ]

        result = radius_outlier_filter(
            integer_points, 1.0, 1
        )

        self.assertEqual(
            result["filtered_points"].dtype,
            np.float64,
        )

        np.testing.assert_array_equal(
            result["kept_indices"], [0, 1]
        )

        points = np.array(integer_points, dtype=np.float64)
        original = points.copy()

        radius_outlier_filter(points, 1.0, 1)

        np.testing.assert_array_equal(
            points, original
        )

    def test_reject_invalid_shapes(self):
        cases = [
            np.array([]),
            np.ones((3,)),
            np.ones((4, 2)),
            np.ones((4, 4)),
            np.ones((2, 2, 3)),
            np.array(3.0),
        ]

        for points in cases:
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    radius_outlier_filter(
                        points, 0.1, 1
                    )

    def test_reject_nonfinite_points(self):
        for bad in [np.nan, np.inf, -np.inf]:
            with self.subTest(bad=str(bad)):
                points = [
                    [0.0, 0.0, 1.0],
                    [1.0, bad, 1.0],
                ]

                with self.assertRaises(ValueError):
                    radius_outlier_filter(
                        points, 0.1, 1
                    )

    def test_reject_nonnumeric_points(self):
        with self.assertRaises(ValueError):
            radius_outlier_filter(
                [["bad", 0, 1]], 0.1, 1
            )

    def test_reject_invalid_radius(self):
        invalid = [
            0.0,
            -0.1,
            np.nan,
            np.inf,
            -np.inf,
            True,
            np.bool_(False),
            "0.1",
            None,
        ]

        for radius in invalid:
            with self.subTest(radius=str(radius)):
                with self.assertRaises(ValueError):
                    radius_outlier_filter(
                        self.points, radius, 1
                    )

    def test_reject_invalid_min_neighbors(self):
        invalid = [
            0,
            -1,
            2.5,
            True,
            np.bool_(False),
            "2",
            None,
        ]

        for k in invalid:
            with self.subTest(k=str(k)):
                with self.assertRaises(ValueError):
                    radius_outlier_filter(
                        self.points, 0.1, k
                    )

    def test_rgb_alignment_and_output_types(self):
        points = np.array([
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [5.0, 5.0, 5.0],
        ])

        colors = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        result = radius_outlier_filter(
            points, 0.15, 1
        )

        filtered_colors = colors[
            result["kept_indices"]
        ]

        np.testing.assert_array_equal(
            result["kept_indices"],
            [0, 1],
        )

        np.testing.assert_array_equal(
            result["filtered_points"],
            points[:2],
        )

        np.testing.assert_array_equal(
            filtered_colors,
            colors[:2],
        )

        self.assertEqual(
            result["filtered_points"].dtype,
            np.float64,
        )
        self.assertEqual(
            result["keep_mask"].dtype,
            np.bool_,
        )

    def test_seeded_random_against_independent_oracle(self):
        rng = np.random.default_rng(20261012)

        points = rng.normal(
            loc=0.0, scale=0.5, size=(64, 3)
        )

        # Deliberately introduce repeated coordinates.
        points[5] = points[2]
        points[9] = points[2]

        for radius in [0.15, 0.35, 0.8]:
            expected_counts = independent_neighbor_counts(
                points, radius
            )

            for k in [1, 2, 4]:
                with self.subTest(radius=radius, k=k):
                    result = radius_outlier_filter(
                        points, radius, k
                    )

                    expected_keep = expected_counts >= k
                    expected_indices = np.flatnonzero(
                        expected_keep
                    )

                    np.testing.assert_array_equal(
                        result["neighbor_counts"],
                        expected_counts,
                    )

                    np.testing.assert_array_equal(
                        result["keep_mask"],
                        expected_keep,
                    )

                    np.testing.assert_array_equal(
                        result["kept_indices"],
                        expected_indices,
                    )

                    np.testing.assert_array_equal(
                        result["filtered_points"],
                        points[expected_keep],
                    )

    def test_parameter_monotonicity(self):
        small = radius_outlier_filter(
            self.points, 0.11, 2
        )

        large = radius_outlier_filter(
            self.points, 0.15, 2
        )

        self.assertTrue(
            np.all(
                large["neighbor_counts"]
                >= small["neighbor_counts"]
            )
        )

        # Increasing radius cannot remove previously kept points.
        self.assertTrue(
            np.all(
                (~small["keep_mask"])
                | large["keep_mask"]
            )
        )

        strict = radius_outlier_filter(
            self.points, 0.15, 4
        )

        # Increasing k cannot add previously removed points.
        self.assertTrue(
            np.all(
                (~strict["keep_mask"])
                | large["keep_mask"]
            )
        )


if __name__ == "__main__":
    unittest.main()
