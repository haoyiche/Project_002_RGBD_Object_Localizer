import numpy as np


def main():
    print("=== EXP029 Minimal Center & AABB ===")

    # 1. Input point cloud from EXP028
    points = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ], dtype=np.float64)

    # 2. Compute centroid
    centroid = np.mean(points, axis=0)

    # 3. Compute AABB
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound

    # 4. Independent expected values
    expected_centroid = np.array([
        2.0 / 3.0,
        2.0 / 3.0,
        1.0,
    ])

    expected_min = np.array([0.0, 0.0, 1.0])
    expected_max = np.array([2.0, 2.0, 1.0])
    expected_center = np.array([1.0, 1.0, 1.0])
    expected_extent = np.array([2.0, 2.0, 0.0])

    # 5. Verify predictions
    np.testing.assert_allclose(
        centroid, expected_centroid,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        min_bound, expected_min,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        max_bound, expected_max,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        bbox_center, expected_center,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        bbox_extent, expected_extent,
        rtol=0, atol=1e-12
    )

    assert centroid.shape == (3,)
    assert bbox_center.shape == (3,)
    assert bbox_extent.shape == (3,)

    # 6. Print results
    print("Input shape:", points.shape)
    print("Centroid:", centroid)
    print("AABB min:", min_bound)
    print("AABB max:", max_bound)
    print("AABB center:", bbox_center)
    print("AABB extent:", bbox_extent)

    print("EXP029 Predictions: PASS")
    print("EXP029 Minimal Experiment: PASS")


if __name__ == "__main__":
    main()
