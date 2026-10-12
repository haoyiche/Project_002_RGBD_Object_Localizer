import numpy as np

from src.geometry.pointcloud_filter import (
    radius_outlier_filter,
)
from src.experiments.exp030_radius_minimal import (
    radius_filter_minimal,
)


def main():
    print("=== EXP030 Production Smoke Test ===")

    points = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.0, 0.1, 1.0],
        [0.1, 0.1, 1.0],
        [1.0, 1.0, 5.0],
    ])

    # Original minimal experiment comparison
    for radius, k in [
        (0.11, 2),
        (0.15, 2),
        (0.15, 3),
        (0.15, 4),
        (0.09, 2),
    ]:
        result = radius_outlier_filter(
            points, radius, k
        )

        minimal_points, minimal_counts, minimal_keep, _ = (
            radius_filter_minimal(points, radius, k)
        )

        np.testing.assert_array_equal(
            result["neighbor_counts"], minimal_counts
        )

        np.testing.assert_array_equal(
            result["keep_mask"], minimal_keep
        )

        np.testing.assert_array_equal(
            result["filtered_points"], minimal_points
        )

        np.testing.assert_array_equal(
            result["kept_indices"],
            np.flatnonzero(minimal_keep),
        )

        print(f"radius={radius}, k={k}: PASS")

    print("NumPy / KD-Tree Consistency: PASS")

    # Radius boundary
    boundary_points = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [3.0, 3.0, 1.0],
    ])

    boundary = radius_outlier_filter(
        boundary_points, 1.0, 2
    )

    np.testing.assert_array_equal(
        boundary["neighbor_counts"], [2, 1, 1, 0]
    )

    np.testing.assert_array_equal(
        boundary["kept_indices"], [0]
    )

    print("Boundary Semantics: PASS")

    # Duplicate coordinates
    duplicates = radius_outlier_filter(
        [[0, 0, 1], [0, 0, 1], [3, 0, 1]],
        0.1,
        1,
    )

    np.testing.assert_array_equal(
        duplicates["neighbor_counts"], [1, 1, 0]
    )

    print("Duplicate Coordinates: PASS")

    # Empty input
    empty = radius_outlier_filter(
        np.empty((0, 3)), 0.1, 2
    )

    assert empty["filtered_points"].shape == (0, 3)
    assert empty["keep_mask"].shape == (0,)
    assert empty["neighbor_counts"].shape == (0,)
    assert empty["kept_indices"].shape == (0,)
    assert empty["filtered_points"].dtype == np.float64
    assert empty["keep_mask"].dtype == np.bool_

    print("Empty Input: PASS")

    # Single point
    single = radius_outlier_filter(
        [[0, 0, 1]], 0.1, 1
    )

    np.testing.assert_array_equal(
        single["neighbor_counts"], [0]
    )
    assert single["filtered_points"].shape == (0, 3)

    print("Single Point: PASS")

    # Input contract
    for bad_radius in [
        0.0, -0.1, np.nan, np.inf, True
    ]:
        try:
            radius_outlier_filter(points, bad_radius, 2)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid radius accepted")

    for bad_k in [0, -1, 2.5, True]:
        try:
            radius_outlier_filter(points, 0.1, bad_k)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid k accepted")

    for bad_points in [
        np.ones((3, 2)),
        [[0, np.nan, 1]],
        [["bad", 0, 1]],
    ]:
        try:
            radius_outlier_filter(bad_points, 0.1, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid points accepted")

    print("Invalid Input Rejection: PASS")

    # RGB alignment
    rgb_points = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [5.0, 5.0, 5.0],
    ])

    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    aligned = radius_outlier_filter(
        rgb_points, 0.15, 1
    )

    filtered_colors = colors[aligned["keep_mask"]]

    np.testing.assert_array_equal(
        aligned["kept_indices"], [0, 1]
    )

    np.testing.assert_array_equal(
        filtered_colors,
        [[1, 0, 0], [0, 1, 0]]
    )

    print("RGB Alignment: PASS")
    print("EXP030 Production Smoke Test: PASS")


if __name__ == "__main__":
    main()
