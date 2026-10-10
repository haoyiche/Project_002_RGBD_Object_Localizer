import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics


def main():
    print("=== EXP029 RGB-D Integration ===")

    # 1. Synthetic RGB-D input
    depth_m = np.array([
        [1.0, 0.0, 2.0],
        [2.0, 3.0, 4.0],
    ], dtype=np.float64)

    target_mask = np.array([
        [True, True, False],
        [True, False, True],
    ], dtype=bool)

    rgb = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    ], dtype=np.uint8)

    intrinsics = dict(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
    )

    # 2. EXP028: Target Mask -> Point Cloud
    points, colors, selected = mask_to_pointcloud(
        depth_m, rgb, target_mask, **intrinsics
    )

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

    assert points.shape == (3, 3)
    assert colors.shape == (3, 3)

    print("Selected mask:")
    print(selected)
    print("Target points:")
    print(points)
    print("Target colors:")
    print(colors)
    print("EXP028 extraction: PASS")

    # 3. EXP029: Point Cloud -> Statistics
    stats = compute_pointcloud_statistics(points)

    expected_stats = {
        "centroid": [1/6, -1/6, 7/3],
        "min_bound": [-1.0, -0.5, 1.0],
        "max_bound": [2.0, 0.0, 4.0],
        "bbox_center": [0.5, -0.25, 2.5],
        "bbox_extent": [3.0, 0.5, 3.0],
        "median_center": [-0.5, 0.0, 2.0],
    }

    for name, expected in expected_stats.items():
        np.testing.assert_allclose(
            stats[name], expected,
            rtol=0, atol=1e-12,
            err_msg=f"Mismatch in {name}"
        )
        print(name, ":", stats[name])

    print("EXP029 statistics: PASS")

    # 4. E4: Empty target handling
    empty_mask = np.zeros_like(
        target_mask, dtype=bool
    )

    empty_points, empty_colors, empty_selected = (
        mask_to_pointcloud(
            depth_m, rgb, empty_mask, **intrinsics
        )
    )

    assert empty_points.shape == (0, 3)
    assert empty_colors.shape == (0, 3)
    assert not empty_selected.any()

    print("EXP028 empty output: PASS")

    # EXP029 must reject empty point clouds
    try:
        compute_pointcloud_statistics(empty_points)
    except ValueError:
        print("EXP029 empty input rejection: PASS")
    else:
        raise AssertionError(
            "EXP029 should reject an empty point cloud"
        )

    # 5. Business-layer integration example
    if empty_points.shape[0] == 0:
        location_result = None
    else:
        location_result = (
            compute_pointcloud_statistics(empty_points)
        )

    assert location_result is None

    print("Empty target routing: PASS")
    print("EXP029 RGB-D Integration: PASS")


if __name__ == "__main__":
    main()
