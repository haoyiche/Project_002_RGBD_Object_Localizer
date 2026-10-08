import numpy as np

from src.geometry.depth_to_pointcloud import (
    depth_to_pointcloud,
)
from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


PARAMS = dict(
    fx=2.0,
    fy=2.0,
    cx=1.0,
    cy=0.0,
)


def compare_case(name, depth_m):
    points_v0 = depth_to_pointcloud(
        depth_m=depth_m,
        **PARAMS,
    )

    points_v1 = depth_to_pointcloud_vectorized(
        depth_m=depth_m,
        **PARAMS,
    )

    # Shape and point-order consistency
    assert points_v0.shape == points_v1.shape
    assert points_v1.ndim == 2
    assert points_v1.shape[1] == 3

    # Numerical consistency
    np.testing.assert_allclose(
        points_v1,
        points_v0,
        rtol=0,
        atol=1e-12,
    )

    print(
        f"{name}: PASS | "
        f"input={depth_m.shape} | "
        f"output={points_v1.shape}"
    )

    return points_v1


def main():
    print("=== EXP027 V0 vs V1 Comparison ===")

    # Case 1: known manual result
    depth_a = np.array([
        [1.0, 0.0, 3.0],
        [2.0, np.nan, 4.0],
    ])

    points_a = compare_case(
        "Known manual case",
        depth_a,
    )

    expected = np.array([
        [-0.5, 0.0, 1.0],
        [ 1.5, 0.0, 3.0],
        [-1.0, 1.0, 2.0],
        [ 2.0, 2.0, 4.0],
    ])

    np.testing.assert_allclose(
        points_a,
        expected,
        rtol=0,
        atol=1e-12,
    )

    # Case 2: invalid depth values
    depth_b = np.array([
        [np.nan, np.inf, -1.0],
        [0.0,    2.0,     3.0],
    ])

    compare_case("Invalid depths", depth_b)

    # Case 3: all invalid
    depth_c = np.zeros((2, 3))

    points_c = compare_case(
        "All invalid",
        depth_c,
    )

    assert points_c.shape == (0, 3)

    # Case 4: single row
    depth_d = np.array([
        [1.0, 2.0, 3.0, 0.0],
    ])

    compare_case("Single row", depth_d)

    # Case 5: single column
    depth_e = np.array([
        [1.0],
        [2.0],
        [0.0],
        [3.0],
    ])

    compare_case("Single column", depth_e)

    # Case 6: deterministic random data
    rng = np.random.default_rng(42)

    depth_f = rng.uniform(
        low=0.3,
        high=4.0,
        size=(7, 11),
    )

    depth_f[0, 1] = 0.0
    depth_f[2, 3] = np.nan
    depth_f[4, 5] = np.inf
    depth_f[6, 7] = -1.0

    compare_case("Random depth map", depth_f)

    print()
    print("All EXP027 V0/V1 comparisons PASSED.")


if __name__ == "__main__":
    main()
