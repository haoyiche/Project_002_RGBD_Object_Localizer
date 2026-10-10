import numpy as np


def analyze(points):
    centroid = np.mean(points, axis=0)

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound

    median_center = np.median(points, axis=0)

    return {
        "centroid": centroid,
        "min_bound": min_bound,
        "max_bound": max_bound,
        "bbox_center": bbox_center,
        "bbox_extent": bbox_extent,
        "median_center": median_center,
    }


def main():
    print("=== EXP029 Active Modification B ===")

    # 1. Original point cloud
    original = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ], dtype=np.float64)

    # 2. Add depth outlier
    outlier = np.array([
        [1.0, 1.0, 10.0],
    ], dtype=np.float64)

    modified = np.vstack((original, outlier))

    old = analyze(original)
    new = analyze(modified)

    # 3. Print results
    print("Original points:", original.shape[0])
    print("Modified points:", modified.shape[0])

    print("Original centroid:", old["centroid"])
    print("New centroid:", new["centroid"])

    print("New AABB min:", new["min_bound"])
    print("New AABB max:", new["max_bound"])
    print("New AABB center:", new["bbox_center"])
    print("New AABB extent:", new["bbox_extent"])

    print("Median center:", new["median_center"])

    # 4. Calculate changes
    centroid_delta = (
        new["centroid"] - old["centroid"]
    )

    bbox_center_delta = (
        new["bbox_center"] - old["bbox_center"]
    )

    bbox_extent_delta = (
        new["bbox_extent"] - old["bbox_extent"]
    )

    print("Centroid delta:", centroid_delta)
    print("AABB center delta:", bbox_center_delta)
    print("AABB extent delta:", bbox_extent_delta)

    # 5. Independent expected values
    expected = {
        "centroid": [1.0, 1.0, 2.8],
        "min_bound": [0.0, 0.0, 1.0],
        "max_bound": [2.0, 2.0, 10.0],
        "bbox_center": [1.0, 1.0, 5.5],
        "bbox_extent": [2.0, 2.0, 9.0],
        "median_center": [1.0, 1.0, 1.0],
    }

    for key, value in expected.items():
        np.testing.assert_allclose(
            new[key],
            value,
            rtol=0,
            atol=1e-12,
            err_msg=f"Mismatch in {key}"
        )

    np.testing.assert_allclose(
        centroid_delta,
        [0.0, 0.0, 1.8],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        bbox_center_delta,
        [0.0, 0.0, 4.5],
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        bbox_extent_delta,
        [0.0, 0.0, 9.0],
        rtol=0, atol=1e-12
    )

    # In this example, AABB is more sensitive
    assert (
        abs(bbox_center_delta[2])
        > abs(centroid_delta[2])
    )

    print("Centroid prediction: PASS")
    print("AABB prediction: PASS")
    print("Delta prediction: PASS")
    print("Median robustness: PASS")
    print("EXP029 Modification B: PASS")


if __name__ == "__main__":
    main()
