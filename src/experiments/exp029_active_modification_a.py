import numpy as np


def analyze(points):
    centroid = np.mean(points, axis=0)

    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound

    return centroid, bbox_center, bbox_extent


def main():
    print("=== EXP029 Active Modification A ===")

    # 1. Original point cloud
    original = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ], dtype=np.float64)

    # 2. Modified point cloud
    modified = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ], dtype=np.float64)

    # 3. Compute geometry
    old_centroid, old_center, old_extent = analyze(original)
    new_centroid, new_center, new_extent = analyze(modified)

    # 4. Print actual results
    print("Original centroid:", old_centroid)
    print("Modified centroid:", new_centroid)

    print("Original AABB center:", old_center)
    print("Modified AABB center:", new_center)

    print("Original extent:", old_extent)
    print("Modified extent:", new_extent)

    axis1_mean = np.mean(modified, axis=1)
    print("Mean axis=1:", axis1_mean)

    # 5. Independent expected values
    expected_centroid = np.array([1.0, 1.0, 1.0])
    expected_center = np.array([1.0, 1.0, 1.0])
    expected_extent = np.array([2.0, 2.0, 0.0])

    expected_axis1 = np.array([
        1.0 / 3.0,
        1.0,
        1.0,
        5.0 / 3.0,
    ])

    # 6. Verify predictions
    np.testing.assert_allclose(
        new_centroid, expected_centroid,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        new_center, expected_center,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        new_extent, expected_extent,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        axis1_mean, expected_axis1,
        rtol=0, atol=1e-12
    )

    # AABB is unchanged
    np.testing.assert_allclose(
        old_center, new_center,
        rtol=0, atol=1e-12
    )

    np.testing.assert_allclose(
        old_extent, new_extent,
        rtol=0, atol=1e-12
    )

    # Centroid changes from 2/3 to 1 on X and Y
    expected_delta = np.array([
        1.0 / 3.0,
        1.0 / 3.0,
        0.0,
    ])

    np.testing.assert_allclose(
        new_centroid - old_centroid,
        expected_delta,
        rtol=0, atol=1e-12
    )

    print("Centroid change: PASS")
    print("AABB unchanged: PASS")
    print("Axis understanding: PASS")
    print("EXP029 Modification A: PASS")


if __name__ == "__main__":
    main()
