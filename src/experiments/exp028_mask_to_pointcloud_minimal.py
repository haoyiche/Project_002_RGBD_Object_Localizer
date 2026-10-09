import numpy as np


def extract_target_pointcloud(
    depth_m, rgb, target_mask, fx, fy, cx, cy
):
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    select_mask = target_mask & valid_depth

    v, u = np.nonzero(select_mask)
    z = depth_m[select_mask]

    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy

    points = np.column_stack((x, y, z))
    colors = rgb[select_mask].astype(np.float64) / 255.0

    return points, colors, select_mask


def main():
    print("=== EXP028 Minimal Target Point Cloud ===")

    depth_m = np.array([
        [2.0, 1.0, 0.0, 3.0],
        [1.5, np.nan, 2.5, 4.0],
        [0.0, 1.2, 0.0, 5.0],
    ], dtype=np.float64)

    target_mask = np.array([
        [False, True, True, False],
        [True, False, True, False],
        [False, True, False, False],
    ])

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[target_mask] = [240, 120, 30]

    params = dict(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=1.0,
    )

    points, colors, select_mask = extract_target_pointcloud(
        depth_m, rgb, target_mask, **params
    )

    # Independent reference calculated by hand
    expected_points = np.array([
        [0.0, -0.5, 1.0],
        [-0.75, 0.0, 1.5],
        [1.25, 0.0, 2.5],
        [0.0, 0.6, 1.2],
    ], dtype=np.float64)

    expected_mask = np.array([
        [False, True, False, False],
        [True, False, True, False],
        [False, True, False, False],
    ])

    expected_color = np.array(
        [240, 120, 30], dtype=np.float64
    ) / 255.0

    assert np.array_equal(select_mask, expected_mask)
    assert points.shape == (4, 3)
    assert colors.shape == (4, 3)

    np.testing.assert_allclose(
        points, expected_points, rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        colors,
        np.tile(expected_color, (4, 1)),
        rtol=0,
        atol=1e-12,
    )

    print("Select mask:")
    print(select_mask)
    print("Target points:")
    print(points)
    print("Target colors:")
    print(colors)
    print("Geometry and RGB alignment: PASS")

    # All target depths invalid
    empty_depth = np.zeros_like(depth_m)
    empty_points, empty_colors, _ = extract_target_pointcloud(
        empty_depth, rgb, target_mask, **params
    )

    assert empty_points.shape == (0, 3)
    assert empty_colors.shape == (0, 3)

    print("Empty target: PASS")
    print("EXP028 minimal experiment PASSED.")


if __name__ == "__main__":
    main()
