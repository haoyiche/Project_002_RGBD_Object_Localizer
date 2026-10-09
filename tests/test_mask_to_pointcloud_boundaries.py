import unittest
import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


class TestMaskToPointCloudBoundaries(unittest.TestCase):

    def setUp(self):
        self.depth = np.ones((2, 2), dtype=np.float64)
        self.rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        self.mask = np.ones((2, 2), dtype=bool)
        self.params = dict(
            fx=2.0, fy=2.0, cx=0.0, cy=0.0
        )

    def test_reject_empty_depth_map(self):
        for shape in [(0, 3), (3, 0)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    mask_to_pointcloud(
                        np.zeros(shape),
                        self.rgb,
                        self.mask,
                        **self.params
                    )

    def test_reject_wrong_mask_shape(self):
        wrong_mask = np.ones((2, 3), dtype=bool)

        with self.assertRaises(ValueError):
            mask_to_pointcloud(
                self.depth,
                self.rgb,
                wrong_mask,
                **self.params
            )

    def test_reject_wrong_rgb_dtype(self):
        wrong_rgb = self.rgb.astype(np.float32)

        with self.assertRaises(ValueError):
            mask_to_pointcloud(
                self.depth,
                wrong_rgb,
                self.mask,
                **self.params
            )

    def test_reject_nonfinite_intrinsics(self):
        for key in self.params:
            for value in [np.nan, np.inf]:
                with self.subTest(key=key, value=str(value)):
                    params = dict(self.params)
                    params[key] = value

                    with self.assertRaises(ValueError):
                        mask_to_pointcloud(
                            self.depth,
                            self.rgb,
                            self.mask,
                            **params
                        )

    def test_empty_target_mask(self):
        empty_mask = np.zeros((2, 2), dtype=bool)

        points, colors, selected = mask_to_pointcloud(
            self.depth,
            self.rgb,
            empty_mask,
            **self.params
        )

        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        self.assertEqual(selected.shape, (2, 2))
        self.assertFalse(selected.any())

    def test_mixed_invalid_depth_filtering(self):
        depth = np.array([
            [np.nan, 0.0],
            [np.inf, -1.0],
            [2.0, 3.0],
        ])

        rgb = np.array([
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0]],
            [[255, 0, 0], [255, 255, 0]],
        ], dtype=np.uint8)

        mask = np.ones((3, 2), dtype=bool)

        points, colors, selected = mask_to_pointcloud(
            depth,
            rgb,
            mask,
            **self.params
        )

        expected_points = np.array([
            [0.0, 2.0, 2.0],
            [1.5, 3.0, 3.0],
        ])

        expected_colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        expected_mask = np.array([
            [False, False],
            [False, False],
            [True, True],
        ])

        np.testing.assert_allclose(
            points, expected_points,
            rtol=0, atol=1e-12
        )

        np.testing.assert_allclose(
            colors, expected_colors,
            rtol=0, atol=1e-12
        )

        np.testing.assert_array_equal(
            selected, expected_mask
        )


if __name__ == "__main__":
    unittest.main()
