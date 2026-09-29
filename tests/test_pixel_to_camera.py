import unittest

from src.geometry.pixel_to_camera import pixel_to_camera


class TestPixelToCamera(unittest.TestCase):

    def test_valid_principal_point(self):
        point = pixel_to_camera(
            u=320,
            v=240,
            depth_m=1.2,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
        )

        self.assertAlmostEqual(point[0], 0.0)
        self.assertAlmostEqual(point[1], 0.0)
        self.assertAlmostEqual(point[2], 1.2)

    def test_reject_zero_fx(self):
        with self.assertRaises(ValueError):
            pixel_to_camera(
                u=320,
                v=240,
                depth_m=1.2,
                fx=0,
                fy=600,
                cx=320,
                cy=240,
            )

    def test_reject_negative_fy(self):
        with self.assertRaises(ValueError):
            pixel_to_camera(
                u=320,
                v=240,
                depth_m=1.2,
                fx=600,
                fy=-1,
                cx=320,
                cy=240,
            )

    def test_reject_zero_depth(self):
        with self.assertRaises(ValueError):
            pixel_to_camera(
                u=320,
                v=240,
                depth_m=0,
                fx=600,
                fy=600,
                cx=320,
                cy=240,
            )

    def test_reject_negative_depth(self):
        with self.assertRaises(ValueError):
            pixel_to_camera(
                u=320,
                v=240,
                depth_m=-1.2,
                fx=600,
                fy=600,
                cx=320,
                cy=240,
            )

    def test_large_depth_is_not_geometry_error(self):
        point = pixel_to_camera(
            u=380,
            v=240,
            depth_m=1200,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
        )

        self.assertAlmostEqual(point[0], 120.0)
        self.assertAlmostEqual(point[2], 1200.0)

    def test_out_of_image_pixel_is_not_geometry_error(self):
        point = pixel_to_camera(
            u=800,
            v=240,
            depth_m=1.2,
            fx=600,
            fy=600,
            cx=320,
            cy=240,
        )

        self.assertAlmostEqual(point[0], 0.96)
        self.assertAlmostEqual(point[1], 0.0)


if __name__ == "__main__":
    unittest.main()
