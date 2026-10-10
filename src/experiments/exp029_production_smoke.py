import numpy as np

from src.geometry.pointcloud_statistics import (
    compute_pointcloud_statistics,
)


def main():
    print("=== EXP029 Production API Smoke Test ===")

    points = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
    ])

    result = compute_pointcloud_statistics(points)

    expected = {
        "centroid": [2/3, 2/3, 1.0],
        "min_bound": [0.0, 0.0, 1.0],
        "max_bound": [2.0, 2.0, 1.0],
        "bbox_center": [1.0, 1.0, 1.0],
        "bbox_extent": [2.0, 2.0, 0.0],
        "median_center": [0.0, 0.0, 1.0],
    }

    for key, value in expected.items():
        np.testing.assert_allclose(
            result[key],
            value,
            rtol=0,
            atol=1e-12,
        )
        assert result[key].shape == (3,)
        assert result[key].dtype == np.float64
        print(key, ":", result[key])

    print("Geometry: PASS")

    # Single-point contract
    single = compute_pointcloud_statistics(
        [[3.0, -2.0, 1.5]]
    )

    np.testing.assert_array_equal(
        single["bbox_extent"],
        [0.0, 0.0, 0.0],
    )

    print("Single point: PASS")

    # Invalid-input contracts
    invalid_cases = {
        "empty": np.empty((0, 3)),
        "wrong_shape": np.ones((4, 2)),
        "nan": [[0.0, np.nan, 1.0]],
        "inf": [[0.0, np.inf, 1.0]],
        "nonnumeric": [["bad", 0, 1]],
    }

    for name, bad_points in invalid_cases.items():
        try:
            compute_pointcloud_statistics(bad_points)
        except ValueError:
            print(name, ": correctly rejected")
        else:
            raise AssertionError(
                f"{name} should raise ValueError"
            )

    print("API Contract: PASS")
    print("EXP029 Production Smoke Test: PASS")


if __name__ == "__main__":
    main()
