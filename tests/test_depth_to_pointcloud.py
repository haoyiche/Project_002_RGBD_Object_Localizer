import unittest

import numpy as np

from src.geometry.depth_to_pointcloud import depth_to_pointcloud
from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


IMPLEMENTATIONS = [
    ("V0", depth_to_pointcloud),
    ("V1", depth_to_pointcloud_vectorized),
]

PARAMS = dict(fx=2.0, fy=2.0, cx=1.0, cy=0.0)


class TestDepthToPointCloud(unittest.TestCase):

    def test_independent_xyz_reference(self):
        depth = np.array([
            [1.0, 0.0, 3.0],
            [2.0, np.nan, 4.0],
        ])

        expected = np.array([
            [-0.5, 0.0, 1.0],
            [1.5, 0.0, 3.0],
            [-1.0, 1.0, 2.0],
            [2.0, 2.0, 4.0],
        ])

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                result = func(depth, **PARAMS)
                np.testing.assert_allclose(
                    result, expected, rtol=0, atol=1e-12
                )

    def test_invalid_depth_filtering(self):
        depth = np.array([
            [np.nan, np.inf, -1.0],
            [0.0, 2.0, 3.0],
        ])

        expected = np.array([
            [0.0, 1.0, 2.0],
            [1.5, 1.5, 3.0],
        ])

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                result = func(depth, **PARAMS)
                np.testing.assert_allclose(result, expected)

    def test_all_invalid_returns_zero_by_three(self):
        depth = np.zeros((2, 3))

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                result = func(depth, **PARAMS)
                self.assertEqual(result.shape, (0, 3))
                self.assertEqual(result.dtype, np.float64)

    def test_single_row_and_column(self):
        cases = [
            (np.array([[1.0, 2.0, 0.0, 3.0]]), (3, 3)),
            (np.array([[1.0], [2.0], [0.0], [3.0]]), (3, 3)),
        ]

        for name, func in IMPLEMENTATIONS:
            for depth, expected_shape in cases:
                with self.subTest(
                    implementation=name, shape=depth.shape
                ):
                    result = func(depth, **PARAMS)
                    self.assertEqual(result.shape, expected_shape)

    def test_row_major_depth_order(self):
        depth = np.array([
            [1.0, 0.0, 3.0],
            [2.0, np.nan, 4.0],
        ])
        expected_z = np.array([1.0, 3.0, 2.0, 4.0])

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                result = func(depth, **PARAMS)
                np.testing.assert_array_equal(
                    result[:, 2], expected_z
                )

    def test_reject_non_2d_input(self):
        inputs = [
            np.array([1.0, 2.0]),
            np.ones((2, 3, 1)),
        ]

        for name, func in IMPLEMENTATIONS:
            for depth in inputs:
                with self.subTest(
                    implementation=name, ndim=depth.ndim
                ):
                    with self.assertRaises(ValueError):
                        func(depth, **PARAMS)

    def test_reject_empty_depth_map(self):
        for shape in [(0, 3), (3, 0)]:
            for name, func in IMPLEMENTATIONS:
                with self.subTest(
                    implementation=name, shape=shape
                ):
                    with self.assertRaises(ValueError):
                        func(np.zeros(shape), **PARAMS)

    def test_reject_invalid_focal_lengths(self):
        for key, value in [("fx", 0.0), ("fy", -1.0)]:
            params = dict(PARAMS)
            params[key] = value

            for name, func in IMPLEMENTATIONS:
                with self.subTest(
                    implementation=name, parameter=key
                ):
                    with self.assertRaises(ValueError):
                        func(np.ones((2, 3)), **params)

    def test_reject_nonfinite_intrinsics(self):
        for key in PARAMS:
            for bad_value in [np.nan, np.inf]:
                params = dict(PARAMS)
                params[key] = bad_value

                for name, func in IMPLEMENTATIONS:
                    with self.subTest(
                        implementation=name,
                        parameter=key,
                        value=str(bad_value),
                    ):
                        with self.assertRaises(ValueError):
                            func(np.ones((2, 3)), **params)

    def test_cx_shift_depends_on_depth(self):
        depth = np.array([
            [1.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ])

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                old = func(depth, **PARAMS)

                params_new = dict(PARAMS)
                params_new["cx"] = 0.0
                new = func(depth, **params_new)

                delta = new - old

                np.testing.assert_allclose(
                    delta[:, 0],
                    old[:, 2] / 2.0,
                    rtol=0,
                    atol=1e-12,
                )
                np.testing.assert_allclose(
                    delta[:, 1:],
                    np.zeros_like(delta[:, 1:]),
                    rtol=0,
                    atol=1e-12,
                )

    def test_accept_2d_python_list(self):
        depth = [[1.0, 0.0], [2.0, 3.0]]

        for name, func in IMPLEMENTATIONS:
            with self.subTest(implementation=name):
                result = func(depth, **PARAMS)
                self.assertEqual(result.shape, (3, 3))

    def test_v0_v1_random_consistency(self):
        rng = np.random.default_rng(27)
        depth = rng.uniform(0.4, 3.0, size=(9, 7))

        depth[0, 1] = 0.0
        depth[3, 2] = np.nan
        depth[7, 6] = np.inf

        v0 = depth_to_pointcloud(depth, **PARAMS)
        v1 = depth_to_pointcloud_vectorized(depth, **PARAMS)

        self.assertEqual(v0.shape, v1.shape)

        np.testing.assert_allclose(
            v0, v1, rtol=0, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
