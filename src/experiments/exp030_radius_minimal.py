import numpy as np

from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


def radius_filter_minimal(points, radius, min_neighbors):
    """
    Minimal radius outlier removal experiment.

    Neighbor definition:
      Euclidean distance <= radius
      Self is NOT counted

    Returns
    -------
    filtered_points : [M, 3]
    neighbor_counts : [N]
    keep_mask       : [N]
    distance_matrix : [N, N]
    """

    # All pairwise XYZ differences
    differences = (
        points[:, None, :] - points[None, :, :]
    )

    # Euclidean distance of each point pair
    distance_matrix = np.linalg.norm(
        differences,
        axis=2,
    )

    # Determine neighbors within radius
    within_radius = distance_matrix <= radius

    # Exclude each point itself
    np.fill_diagonal(within_radius, False)

    # Count valid neighbors per point
    neighbor_counts = np.sum(
        within_radius,
        axis=1,
    )

    # Keep points with sufficient neighbors
    keep_mask = neighbor_counts >= min_neighbors

    filtered_points = points[keep_mask]

    return (
        filtered_points,
        neighbor_counts,
        keep_mask,
        distance_matrix,
    )


def main():
    print("=== EXP030 Radius Minimal Experiment ===")

    points = np.array([
        [0.0, 0.0, 1.0],  # A
        [0.1, 0.0, 1.0],  # B
        [0.0, 0.1, 1.0],  # C
        [0.1, 0.1, 1.0],  # D
        [1.0, 1.0, 5.0],  # E
    ], dtype=np.float64)

    radius = 0.11
    min_neighbors = 2

    # ----------------------------------
    # Experiment 1: Original parameters
    # ----------------------------------

    filtered, counts, keep, distances = (
        radius_filter_minimal(
            points,
            radius,
            min_neighbors,
        )
    )

    print("\nOriginal points:", points.shape)
    print("Radius:", radius)
    print("Min neighbors:", min_neighbors)

    print("\nPairwise distance matrix:")
    print(np.round(distances, 4))

    print("\nNeighbor counts:", counts)
    print("Keep mask:", keep)
    print("Filtered points:")
    print(filtered)

    # Independent predictions: P1 and P2
    expected_counts = np.array([2, 2, 2, 2, 0])

    expected_keep = np.array([
        True, True, True, True, False
    ])

    np.testing.assert_array_equal(
        counts, expected_counts
    )

    np.testing.assert_array_equal(
        keep, expected_keep
    )

    np.testing.assert_allclose(
        filtered,
        points[:4],
        rtol=0,
        atol=1e-12,
    )

    assert filtered.shape == (4, 3)

    print("\nP1 Neighbor Counts: PASS")
    print("P2 Filtering: PASS")

    # ----------------------------------
    # Experiment 2: EXP029 statistics
    # ----------------------------------

    before = compute_pointcloud_statistics(points)
    after = compute_pointcloud_statistics(filtered)

    print("\nBefore centroid:", before["centroid"])
    print("After centroid:", after["centroid"])

    print("Before AABB extent:", before["bbox_extent"])
    print("After AABB extent:", after["bbox_extent"])

    # Independent prediction: P3
    np.testing.assert_allclose(
        before["centroid"],
        [0.24, 0.24, 1.8],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        before["bbox_extent"],
        [1.0, 1.0, 4.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        after["centroid"],
        [0.05, 0.05, 1.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        after["bbox_extent"],
        [0.1, 0.1, 0.0],
        rtol=0,
        atol=1e-12,
    )

    print("P3 Statistics Comparison: PASS")

    # ----------------------------------
    # Experiment 3: Radius too small
    # ----------------------------------

    small_radius = 0.09

    filtered_small, counts_small, keep_small, _ = (
        radius_filter_minimal(
            points,
            small_radius,
            min_neighbors,
        )
    )

    print("\nSmall radius:", small_radius)
    print("Small-radius counts:", counts_small)
    print("Small-radius keep mask:", keep_small)
    print("Small-radius result:", filtered_small.shape)

    np.testing.assert_array_equal(
        counts_small,
        [0, 0, 0, 0, 0],
    )

    assert filtered_small.shape == (0, 3)
    assert not keep_small.any()

    # Business layer: do not call EXP029 on empty data
    if filtered_small.shape[0] == 0:
        location_result = None
    else:
        location_result = (
            compute_pointcloud_statistics(filtered_small)
        )

    assert location_result is None

    print("P4 Small Radius: PASS")
    print("Empty Result Routing: PASS")

    print("\nEXP030 Minimal Experiment: PASS")


if __name__ == "__main__":
    main()
