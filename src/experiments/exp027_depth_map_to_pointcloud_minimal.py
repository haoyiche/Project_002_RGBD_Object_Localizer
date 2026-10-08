import numpy as np

from src.geometry.depth_to_pointcloud import depth_to_pointcloud
from src.geometry.pixel_to_camera import pixel_to_camera


def main():

    print("=== EXP027 Minimal Depth-to-PointCloud ===")

    depth_m = np.array([
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 2.0],
    ])


    fx = 2.0
    fy = 2.0
    cx = 1.0
    cy = 0.0

    points = depth_to_pointcloud(
        depth_m=depth_m,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    print()
    print("Depth map:")
    print(depth_m)

    print()
    print("Point cloud:")
    print(points)

    print()
    print("Shape:", points.shape)

    expected = np.array([
        [-0.5, 0.0, 1.0],
        [ 0.0, 0.0, 1.0],
        [-1.0, 1.0, 2.0],
        [ 0.0, 1.0, 2.0],
        [ 1.0, 1.0, 2.0],
    ])

    # Shape verification
    assert points.shape == (5, 3)

    # Geometry verification
    np.testing.assert_allclose(
        points,
        expected,
        rtol=0,
        atol=1e-12,
    )

    # Consistency with EXP026
    reference = pixel_to_camera(
        u=1,
        v=1,
        depth_m=2.0,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    np.testing.assert_allclose(
        points[3],
        reference,
        rtol=0,
        atol=1e-12,
    )

    print()
    print("All EXP027 minimal checks PASSED.")


if __name__ == "__main__":
    main()