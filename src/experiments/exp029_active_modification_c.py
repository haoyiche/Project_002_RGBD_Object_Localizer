import numpy as np


def analyze(points):
    centroid = np.mean(points, axis=0)
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)

    return {
        "centroid": centroid,
        "min_bound": minimum,
        "max_bound": maximum,
        "bbox_center": (minimum + maximum) / 2,
        "bbox_extent": maximum - minimum,
    }


def main():
    print("=== EXP029 Active Modification C ===")

    original = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 2.0, 1.0],
    ], dtype=np.float64)

    translation = np.array([3.0, -2.0, 0.5])

    modified = original + translation

    old = analyze(original)
    new = analyze(modified)

    # 1. Independent XYZ reference
    expected_points = np.array([
        [3.0, -2.0, 1.5],
        [5.0, -2.0, 1.5],
        [3.0,  0.0, 1.5],
        [5.0,  0.0, 1.5],
    ])

    np.testing.assert_allclose(
        modified, expected_points, atol=1e-12
    )

    print("Modified points:")
    print(modified)
    print("Point translation: PASS")

    # 2. Centroid
    np.testing.assert_allclose(
        new["centroid"],
        [4.0, -1.0, 1.5],
        atol=1e-12
    )

    np.testing.assert_allclose(
        new["centroid"] - old["centroid"],
        translation,
        atol=1e-12
    )

    print("New centroid:", new["centroid"])
    print("Centroid translation: PASS")

    # 3. AABB
    expected = {
        "min_bound": [3.0, -2.0, 1.5],
        "max_bound": [5.0, 0.0, 1.5],
        "bbox_center": [4.0, -1.0, 1.5],
        "bbox_extent": [2.0, 2.0, 0.0],
    }

    for key, value in expected.items():
        np.testing.assert_allclose(
            new[key], value, atol=1e-12
        )

    for key in ["min_bound", "max_bound", "bbox_center"]:
        np.testing.assert_allclose(
            new[key] - old[key],
            translation,
            atol=1e-12
        )

    np.testing.assert_allclose(
        new["bbox_extent"],
        old["bbox_extent"],
        atol=1e-12
    )

    print("New AABB center:", new["bbox_center"])
    print("New AABB extent:", new["bbox_extent"])
    print("AABB translation: PASS")
    print("AABB extent invariant: PASS")

    # 4. All pairwise distances
    old_distances = np.linalg.norm(
        original[:, None, :] - original[None, :, :],
        axis=2
    )

    new_distances = np.linalg.norm(
        modified[:, None, :] - modified[None, :, :],
        axis=2
    )

    np.testing.assert_allclose(
        new_distances,
        old_distances,
        atol=1e-12
    )

    print("Pairwise distance invariant: PASS")
    print("EXP029 Modification C: PASS")


if __name__ == "__main__":
    main()
