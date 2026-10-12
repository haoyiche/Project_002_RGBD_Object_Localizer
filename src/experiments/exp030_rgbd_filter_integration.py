import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_filter import radius_outlier_filter
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics


def run_pipeline(depth, rgb, mask, radius=0.11, min_neighbors=2):
    intrinsics = {
        "fx": 10.0,
        "fy": 10.0,
        "cx": 0.0,
        "cy": 0.0,
    }

    # EXP028: Mask -> XYZ and RGB
    points, colors, selected = mask_to_pointcloud(
        depth, rgb, mask, **intrinsics
    )

    # EXP030: XYZ -> filtered XYZ and original indices
    result = radius_outlier_filter(
        points, radius, min_neighbors
    )

    filtered_points = result["filtered_points"]
    kept_indices = result["kept_indices"]

    # The SAME indices must be applied to RGB.
    filtered_colors = colors[kept_indices]

    original_count = len(points)
    filtered_count = len(filtered_points)

    # Illustrative business-layer routing.
    if not mask.any():
        status = "EMPTY_MASK"
    elif original_count == 0:
        status = "NO_VALID_DEPTH"
    elif filtered_count == 0:
        status = "ALL_FILTERED"
    elif filtered_count < 2:
        status = "LOW_CONFIDENCE"
    else:
        status = "SUCCESS"

    # EXP029: only non-empty clouds are accepted.
    stats = (
        compute_pointcloud_statistics(filtered_points)
        if filtered_count > 0
        else None
    )

    return {
        "status": status,
        "selected": selected,
        "points": points,
        "colors": colors,
        "filtered_points": filtered_points,
        "filtered_colors": filtered_colors,
        "keep_mask": result["keep_mask"],
        "neighbor_counts": result["neighbor_counts"],
        "kept_indices": kept_indices,
        "stats": stats,
        "original_count": original_count,
        "filtered_count": filtered_count,
    }


def main():
    print("=== EXP030 RGB-D Filter Integration ===")

    depth = np.array([
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 5.0],
    ], dtype=np.float64)

    mask = np.array([
        [True, True, False],
        [True, True, True],
    ], dtype=bool)

    rgb = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 0]],
        [[0, 0, 255], [255, 255, 0], [0, 255, 255]],
    ], dtype=np.uint8)

    # ---------------------------------------
    # G1-G3: Normal pipeline with one outlier
    # ---------------------------------------
    normal = run_pipeline(depth, rgb, mask)

    expected_points = np.array([
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.0, 0.1, 1.0],
        [0.1, 0.1, 1.0],
        [1.0, 0.5, 5.0],
    ])

    expected_colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ])

    np.testing.assert_allclose(
        normal["points"], expected_points,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        normal["colors"], expected_colors,
        rtol=0, atol=1e-12
    )

    assert normal["points"].shape == (5, 3)
    assert normal["colors"].shape == (5, 3)

    print("\nG1 RGB-D Projection: PASS")

    np.testing.assert_array_equal(
        normal["neighbor_counts"],
        [2, 2, 2, 2, 0]
    )

    np.testing.assert_array_equal(
        normal["keep_mask"],
        [True, True, True, True, False]
    )

    np.testing.assert_array_equal(
        normal["kept_indices"],
        [0, 1, 2, 3]
    )

    np.testing.assert_allclose(
        normal["filtered_points"],
        expected_points[:4],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        normal["filtered_colors"],
        expected_colors[:4],
        rtol=0, atol=1e-12
    )

    print("Neighbor counts:", normal["neighbor_counts"])
    print("Kept indices:", normal["kept_indices"])
    print("Filtered points:", normal["filtered_count"])
    print("G2 Radius Filtering and RGB Alignment: PASS")

    before = compute_pointcloud_statistics(
        normal["points"]
    )

    after = normal["stats"]

    np.testing.assert_allclose(
        before["centroid"],
        [0.24, 0.14, 1.8],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        before["bbox_extent"],
        [1.0, 0.5, 4.0],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        after["centroid"],
        [0.05, 0.05, 1.0],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        after["bbox_extent"],
        [0.1, 0.1, 0.0],
        rtol=0, atol=1e-12
    )

    assert normal["status"] == "SUCCESS"

    print("\nBefore centroid:", before["centroid"])
    print("After centroid:", after["centroid"])
    print("Before extent:", before["bbox_extent"])
    print("After extent:", after["bbox_extent"])
    print("G3 Geometry Statistics: PASS")

    # ---------------------------------------
    # G4-A: Empty target mask
    # ---------------------------------------
    empty_mask = np.zeros_like(mask)

    empty = run_pipeline(depth, rgb, empty_mask)

    assert empty["status"] == "EMPTY_MASK"
    assert empty["points"].shape == (0, 3)
    assert empty["filtered_points"].shape == (0, 3)
    assert empty["filtered_colors"].shape == (0, 3)
    assert empty["keep_mask"].shape == (0,)
    assert empty["neighbor_counts"].shape == (0,)
    assert empty["kept_indices"].shape == (0,)
    assert empty["stats"] is None

    print("\nEmpty mask status:", empty["status"])
    print("G4 Empty Mask Routing: PASS")

    # ---------------------------------------
    # G4-B: Non-empty mask, invalid depth
    # ---------------------------------------
    invalid_depth = np.zeros_like(depth)

    invalid = run_pipeline(invalid_depth, rgb, mask)

    assert invalid["status"] == "NO_VALID_DEPTH"
    assert invalid["original_count"] == 0
    assert invalid["filtered_count"] == 0
    assert invalid["stats"] is None

    print("Invalid depth status:", invalid["status"])
    print("G4 Invalid Depth Routing: PASS")

    # ---------------------------------------
    # G4-C: Excessively strict filtering
    # ---------------------------------------
    strict = run_pipeline(
        depth, rgb, mask,
        radius=0.11,
        min_neighbors=3,
    )

    assert strict["status"] == "ALL_FILTERED"
    assert strict["original_count"] == 5
    assert strict["filtered_count"] == 0
    assert strict["stats"] is None

    np.testing.assert_array_equal(
        strict["neighbor_counts"],
        [2, 2, 2, 2, 0]
    )

    print("Strict filtering status:", strict["status"])
    print("G4 All Filtered Routing: PASS")

    # ---------------------------------------
    # G5: Only one point survives
    # ---------------------------------------
    sparse_mask = np.array([
        [True, True, False],
        [True, False, False],
    ], dtype=bool)

    low = run_pipeline(
        depth, rgb, sparse_mask,
        radius=0.11,
        min_neighbors=2,
    )

    assert low["status"] == "LOW_CONFIDENCE"
    assert low["original_count"] == 3
    assert low["filtered_count"] == 1

    np.testing.assert_array_equal(
        low["neighbor_counts"],
        [2, 1, 1]
    )

    np.testing.assert_array_equal(
        low["kept_indices"],
        [0]
    )

    np.testing.assert_allclose(
        low["stats"]["centroid"],
        [0.0, 0.0, 1.0],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        low["stats"]["bbox_extent"],
        [0.0, 0.0, 0.0],
        rtol=0, atol=1e-12
    )

    print("One-point status:", low["status"])
    print("G5 Low Confidence Routing: PASS")

    print("\nEXP030 RGB-D Integration: PASS")


if __name__ == "__main__":
    main()
