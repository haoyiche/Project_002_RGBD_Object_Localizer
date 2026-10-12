import numpy as np

from src.experiments.exp030_radius_minimal import (
    radius_filter_minimal,
)
from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


def main():
    print("=== EXP030 Active Modification B ===")

    points = np.array([
        [0.0, 0.0, 1.0],  # A
        [0.1, 0.0, 1.0],  # B
        [0.0, 0.1, 1.0],  # C
        [0.1, 0.1, 1.0],  # D
        [1.0, 1.0, 5.0],  # E
    ], dtype=np.float64)

    radius = 0.15

    expected_counts = np.array([3, 3, 3, 3, 0])
    expected_keep_normal = np.array([
        True, True, True, True, False
    ])
    expected_keep_strict = np.zeros(5, dtype=bool)

    results = {}

    # Test three thresholds with identical input data
    for min_neighbors in [2, 3, 4]:
        filtered, counts, keep, _ = radius_filter_minimal(
            points,
            radius,
            min_neighbors,
        )

        results[min_neighbors] = {
            "points": filtered,
            "counts": counts,
            "keep": keep,
        }

        print(f"\n--- min_neighbors = {min_neighbors} ---")
        print("Neighbor counts:", counts)
        print("Keep mask:", keep)
        print("Filtered shape:", filtered.shape)

        # B1: Counts must not depend on threshold
        np.testing.assert_array_equal(
            counts,
            expected_counts,
        )

    print("\nB1 Neighbor Count Invariance: PASS")

    # B2: Threshold = 3
    result3 = results[3]

    np.testing.assert_array_equal(
        result3["keep"],
        expected_keep_normal,
    )

    assert result3["points"].shape == (4, 3)

    np.testing.assert_allclose(
        result3["points"],
        points[:4],
        rtol=0,
        atol=1e-12,
    )

    print("B2 Threshold 3: PASS")

    # B3: Threshold = 4
    result4 = results[4]

    np.testing.assert_array_equal(
        result4["keep"],
        expected_keep_strict,
    )

    assert result4["points"].shape == (0, 3)

    print("B3 Threshold 4: PASS")

    # Compare threshold 2 and threshold 3
    np.testing.assert_array_equal(
        results[2]["keep"],
        results[3]["keep"],
    )

    # B4: Non-empty statistics
    stats3 = compute_pointcloud_statistics(
        result3["points"]
    )

    np.testing.assert_allclose(
        stats3["centroid"],
        [0.05, 0.05, 1.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        stats3["bbox_extent"],
        [0.1, 0.1, 0.0],
        rtol=0,
        atol=1e-12,
    )

    print("\nThreshold 3 centroid:", stats3["centroid"])
    print("Threshold 3 extent:", stats3["bbox_extent"])

    # EXP029 must reject empty input
    try:
        compute_pointcloud_statistics(result4["points"])
    except ValueError:
        print("EXP029 empty input rejection: PASS")
    else:
        raise AssertionError(
            "EXP029 must reject empty points"
        )

    # Correct business-layer routing
    if result4["points"].shape[0] == 0:
        stats4 = None
    else:
        stats4 = compute_pointcloud_statistics(
            result4["points"]
        )

    assert stats4 is None

    print("B4 Statistics and Empty Routing: PASS")

    # B5: Diagnostic information
    original_count = points.shape[0]
    filtered_count = result4["points"].shape[0]

    diagnostic = {
        "original_count": original_count,
        "filtered_count": filtered_count,
        "removed_count": original_count - filtered_count,
        "radius": radius,
        "min_neighbors": 4,
        "neighbor_counts": result4["counts"].copy(),
        "keep_mask": result4["keep"].copy(),
        "status": (
            "ALL_FILTERED"
            if original_count > 0 and filtered_count == 0
            else "SUCCESS"
        ),
    }

    assert diagnostic["original_count"] == 5
    assert diagnostic["filtered_count"] == 0
    assert diagnostic["removed_count"] == 5
    assert diagnostic["status"] == "ALL_FILTERED"

    print("\nDiagnostic status:", diagnostic["status"])
    print("Original count:", diagnostic["original_count"])
    print("Filtered count:", diagnostic["filtered_count"])
    print("Removed count:", diagnostic["removed_count"])
    print("Radius:", diagnostic["radius"])
    print("Min neighbors:", diagnostic["min_neighbors"])

    print("B5 Diagnostic Routing: PASS")
    print("\nEXP030 Modification B: PASS")


if __name__ == "__main__":
    main()
