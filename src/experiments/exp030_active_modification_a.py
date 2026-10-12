import numpy as np

from src.experiments.exp030_radius_minimal import (
    radius_filter_minimal,
)

from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


def main():
    print("=== EXP030 Active Modification A ===")

    # 1. Original five-point cloud
    points = np.array([
        [0.0, 0.0, 1.0],  # A
        [0.1, 0.0, 1.0],  # B
        [0.0, 0.1, 1.0],  # C
        [0.1, 0.1, 1.0],  # D
        [1.0, 1.0, 5.0],  # E
    ], dtype=np.float64)

    min_neighbors = 2

    # 2. Original experiment
    original_radius = 0.11

    old_filtered, old_counts, old_keep, _ = (
        radius_filter_minimal(
            points,
            original_radius,
            min_neighbors,
        )
    )

    # 3. Active Modification A
    modified_radius = 0.15

    new_filtered, new_counts, new_keep, _ = (
        radius_filter_minimal(
            points,
            modified_radius,
            min_neighbors,
        )
    )

    print("\nOriginal radius:", original_radius)
    print("Modified radius:", modified_radius)
    print("Min neighbors:", min_neighbors)

    print("\nOriginal counts:", old_counts)
    print("Modified counts:", new_counts)

    print("\nOriginal keep mask:", old_keep)
    print("Modified keep mask:", new_keep)

    # 4. A1: Neighbor counts
    np.testing.assert_array_equal(
        old_counts,
        [2, 2, 2, 2, 0],
    )

    np.testing.assert_array_equal(
        new_counts,
        [3, 3, 3, 3, 0],
    )

    print("\nA1 Neighbor Count Prediction: PASS")

    # 5. A2: Filtering results
    expected_keep = np.array([
        True, True, True, True, False
    ])

    np.testing.assert_array_equal(
        old_keep,
        expected_keep,
    )

    np.testing.assert_array_equal(
        new_keep,
        expected_keep,
    )

    assert new_filtered.shape == (4, 3)

    np.testing.assert_allclose(
        new_filtered,
        points[:4],
        rtol=0,
        atol=1e-12,
    )

    print("A2 Filtering Prediction: PASS")

    # 6. A3: EXP029 statistics
    old_stats = compute_pointcloud_statistics(
        old_filtered
    )

    new_stats = compute_pointcloud_statistics(
        new_filtered
    )

    print("\nNew centroid:", new_stats["centroid"])
    print("New AABB extent:", new_stats["bbox_extent"])

    np.testing.assert_allclose(
        new_stats["centroid"],
        [0.05, 0.05, 1.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        new_stats["bbox_extent"],
        [0.1, 0.1, 0.0],
        rtol=0,
        atol=1e-12,
    )

    print("A3 Statistics Prediction: PASS")

    # 7. A4: Counts change, selected points do not
    assert np.any(new_counts != old_counts)

    np.testing.assert_array_equal(
        old_keep,
        new_keep,
    )

    np.testing.assert_allclose(
        old_filtered,
        new_filtered,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        old_stats["centroid"],
        new_stats["centroid"],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        old_stats["bbox_extent"],
        new_stats["bbox_extent"],
        rtol=0,
        atol=1e-12,
    )

    print("A4 Count vs Keep Mask: PASS")

    print("\nEXP030 Modification A: PASS")


if __name__ == "__main__":
    main()
