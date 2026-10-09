import unittest
import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud


class TestMaskToPointCloud(unittest.TestCase):

    # ==========================================
    # Test 1: RGB and XYZ Alignment
    # ==========================================

    def test_distinct_rgb_colors_align_with_xyz(self):
        depth = np.array([
            [1.0, 0.0],
            [2.0, 3.0],
        ], dtype=np.float64)

        target_mask = np.array([
            [True, True],
            [False, True],
        ], dtype=bool)

        rgb = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ], dtype=np.uint8)

        points, colors, selected = mask_to_pointcloud(
            depth_m=depth,
            rgb=rgb,
            target_mask=target_mask,
            fx=2.0,
            fy=2.0,
            cx=0.0,
            cy=0.0,
        )

        expected_points = np.array([
            [0.0, 0.0, 1.0],
            [1.5, 1.5, 3.0],
        ])

        expected_colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        expected_mask = np.array([
            [True, False],
            [False, True],
        ])

        self.assertEqual(points.shape, (2, 3))
        self.assertEqual(colors.shape, (2, 3))

        np.testing.assert_array_equal(
            selected,
            expected_mask,
        )

        np.testing.assert_allclose(
            points,
            expected_points,
            rtol=0,
            atol=1e-12,
        )

        np.testing.assert_allclose(
            colors,
            expected_colors,
            rtol=0,
            atol=1e-12,
        )

    # ==========================================
    # Test 2: Reject Wrong RGB Shape
    # ==========================================

    def test_reject_rgb_shape_mismatch(self):
        depth = np.ones((2, 2))
        rgb = np.zeros((3, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        with self.assertRaises(ValueError):
            mask_to_pointcloud(
                depth,
                rgb,
                mask,
                2.0, 2.0, 0.0, 0.0,
            )

    # ==========================================
    # Test 3: Reject Non-Boolean Mask
    # ==========================================

    def test_reject_nonboolean_mask(self):
        depth = np.ones((2, 2))
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)

        # Wrong dtype: uint8 instead of bool
        mask = np.ones((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            mask_to_pointcloud(
                depth,
                rgb,
                mask,
                2.0, 2.0, 0.0, 0.0,
            )

    # ==========================================
    # Test 4: Reject Zero Focal Length
    # ==========================================

    def test_reject_zero_fx(self):
        depth = np.ones((2, 2))
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        with self.assertRaises(ValueError):
            mask_to_pointcloud(
                depth,
                rgb,
                mask,
                0.0, 2.0, 0.0, 0.0,
            )

    # ==========================================
    # Test 5: All Target Depths Invalid
    # ==========================================

    def test_all_invalid_target_depth(self):
        depth = np.zeros((2, 2))
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.ones((2, 2), dtype=bool)

        points, colors, selected = mask_to_pointcloud(
            depth,
            rgb,
            mask,
            2.0, 2.0, 0.0, 0.0,
        )

        self.assertEqual(points.shape, (0, 3))
        self.assertEqual(colors.shape, (0, 3))
        self.assertEqual(selected.shape, (2, 2))

        self.assertEqual(points.dtype, np.float64)
        self.assertEqual(colors.dtype, np.float64)
        self.assertEqual(selected.dtype, np.bool_)

        self.assertFalse(selected.any())


if __name__ == "__main__":
    unittest.main()
