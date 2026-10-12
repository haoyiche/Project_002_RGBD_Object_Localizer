import numpy as np

from src.experiments.exp030_radius_minimal import (
    radius_filter_minimal,
)
from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


def filter_variant(
    points,
    radius,
    min_neighbors,
    *,
    include_boundary=True,
    exclude_self=True,
):
    """Compare correct and intentionally buggy rules."""

    differences = (
        points[:, None, :] - points[None, :, :]
    )
    distances = np.linalg.norm(differences, axis=2)

    if include_boundary:
        within_radius = distances <= radius
    else:
        within_radius = distances < radius

    if exclude_self:
        np.fill_diagonal(within_radius, False)

    counts = np.sum(within_radius, axis=1)
    keep = counts >= min_neighbors

    return points[keep], counts, keep, within_radius


def check_case(
    name,
    result,
    expected_counts,
    expected_keep,
    expected_points,
):
    filtered, counts, keep, _ = result

    print(f"\n--- {name} ---")
    print("Neighbor counts:", counts)
    print("Keep mask:", keep)
    print("Filtered shape:", filtered.shape)

    np.testing.assert_array_equal(
        counts,
        expected_counts,
    )

    np.testing.assert_array_equal(
        keep,
        expected_keep,
    )

    np.testing.assert_allclose(
        filtered,
        expected_points,
        rtol=0,
        atol=1e-12,
    )

    print(f"{name} Prediction: PASS")


def main():
    print("=== EXP030 Active Modification C ===")

    points = np.array([
        [0.0, 0.0, 1.0],  # A
        [1.0, 0.0, 1.0],  # B
        [0.0, 1.0, 1.0],  # C
        [3.0, 3.0, 1.0],  # D
    ], dtype=np.float64)

    radius = 1.0
    min_neighbors = 2

    # C1: Correct implementation
    correct = filter_variant(
        points, radius, min_neighbors,
        include_boundary=True,
        exclude_self=True,
    )

    check_case(
        "C1 Correct",
        correct,
        [2, 1, 1, 0],
        [True, False, False, False],
        points[:1],
    )

    # Compare with existing minimal implementation
    baseline = radius_filter_minimal(
        points, radius, min_neighbors
    )

    for i in range(3):
        np.testing.assert_array_equal(
            correct[i], baseline[i]
        )

    print("Minimal Implementation Consistency: PASS")

    # C2: Bug 1, strict comparison
    bug_boundary = filter_variant(
        points, radius, min_neighbors,
        include_boundary=False,
        exclude_self=True,
    )

    check_case(
        "C2 Boundary Bug",
        bug_boundary,
        [0, 0, 0, 0],
        [False, False, False, False],
        np.empty((0, 3)),
    )

    # C3: Bug 2, self counted
    bug_self = filter_variant(
        points, radius, min_neighbors,
        include_boundary=True,
        exclude_self=False,
    )

    check_case(
        "C3 Self-Counting Bug",
        bug_self,
        [3, 2, 2, 1],
        [True, True, True, False],
        points[:3],
    )

    # C4: Effects on EXP029
    correct_stats = compute_pointcloud_statistics(
        correct[0]
    )

    np.testing.assert_allclose(
        correct_stats["centroid"],
        [0.0, 0.0, 1.0],
        rtol=0, atol=1e-12,
    )

    np.testing.assert_allclose(
        correct_stats["bbox_extent"],
        [0.0, 0.0, 0.0],
        rtol=0, atol=1e-12,
    )

    try:
        compute_pointcloud_statistics(bug_boundary[0])
    except ValueError:
        print("\nC2 Empty Input Rejection: PASS")
    else:
        raise AssertionError(
            "EXP029 must reject empty input"
        )

    self_stats = compute_pointcloud_statistics(
        bug_self[0]
    )

    np.testing.assert_allclose(
        self_stats["centroid"],
        [1/3, 1/3, 1.0],
        rtol=0, atol=1e-12,
    )

    np.testing.assert_allclose(
        self_stats["bbox_extent"],
        [1.0, 1.0, 0.0],
        rtol=0, atol=1e-12,
    )

    print("C1 centroid:", correct_stats["centroid"])
    print("C1 extent:", correct_stats["bbox_extent"])
    print("C3 centroid:", self_stats["centroid"])
    print("C3 extent:", self_stats["bbox_extent"])
    print("C4 EXP029 Statistics: PASS")

    # C5: Independent assertions must catch both bugs
    expected_counts = np.array([2, 1, 1, 0])

    for name, result in [
        ("Boundary Bug", bug_boundary),
        ("Self-Counting Bug", bug_self),
    ]:
        try:
            np.testing.assert_array_equal(
                result[1],
                expected_counts,
            )
        except AssertionError:
            print(f"C5 {name} Detected: PASS")
        else:
            raise AssertionError(
                f"{name} was not detected"
            )

    # C5: Duplicate coordinates, different indices
    duplicate_points = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [3.0, 0.0, 1.0],
    ])

    duplicate_result = filter_variant(
        duplicate_points,
        radius=0.1,
        min_neighbors=1,
    )

    dup_filtered, dup_counts, dup_keep, dup_near = (
        duplicate_result
    )

    np.testing.assert_array_equal(
        dup_counts, [1, 1, 0]
    )

    np.testing.assert_array_equal(
        dup_keep, [True, True, False]
    )

    assert dup_filtered.shape == (2, 3)

    # Same coordinates, different indices remain neighbors
    assert dup_near[0, 1]
    assert dup_near[1, 0]

    # A point must not count itself
    assert not np.diag(dup_near).any()

    print("Duplicate-coordinate neighbors:", dup_counts)
    print("C5 Duplicate Index Semantics: PASS")

    print("\nEXP030 Modification C: PASS")


if __name__ == "__main__":
    main()
