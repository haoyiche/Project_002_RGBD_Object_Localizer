import numpy as np

from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


def faulty_depth_to_pointcloud(
    depth_m,
    fx,
    fy,
    cx,
    cy,
):
    # INTENTIONAL BUG:
    # np.indices() returns (v_grid, u_grid).
    # These two variables are deliberately swapped.
    u_grid, v_grid = np.indices(depth_m.shape)

    valid = (
        np.isfinite(depth_m)
        & (depth_m > 0)
    )

    u = u_grid[valid]
    v = v_grid[valid]
    z = depth_m[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.column_stack((x, y, z))


def main():
    print("=== EXP027 u/v Fault Injection ===")

    depth_m = np.array([
        [1.0, 0.0, 3.0],
        [2.0, np.nan, 4.0],
    ])

    params = dict(
        fx=2.0,
        fy=2.0,
        cx=1.0,
        cy=0.0,
    )

    # Independent hand-calculated reference
    expected = np.array([
        [-0.5, 0.0, 1.0],
        [ 1.5, 0.0, 3.0],
        [-1.0, 1.0, 2.0],
        [ 2.0, 2.0, 4.0],
    ])

    correct = depth_to_pointcloud_vectorized(
        depth_m=depth_m,
        **params,
    )

    faulty = faulty_depth_to_pointcloud(
        depth_m=depth_m,
        **params,
    )

    print("\nCorrect:")
    print(correct)

    print("\nFaulty:")
    print(faulty)

    # Check 1: independent geometry reference
    np.testing.assert_allclose(
        correct,
        expected,
        rtol=0,
        atol=1e-12,
    )

    print("\nCorrect geometry check: PASS")

    # Check 2: structural checks
    assert faulty.shape == (4, 3)

    np.testing.assert_array_equal(
        faulty[:, 2],
        expected[:, 2],
    )

    assert np.isfinite(faulty).all()

    print("Faulty shape check: PASS")
    print("Faulty Z check: PASS")
    print("Faulty finite check: PASS")

    # Check 3: independent XYZ comparison
    # This MUST detect the injected fault.
    try:
        np.testing.assert_allclose(
            faulty,
            expected,
            rtol=0,
            atol=1e-12,
        )

    except AssertionError:
        print(
            "Independent XYZ check: "
            "FAULT DETECTED"
        )

    else:
        raise AssertionError(
            "Fault injection was NOT detected!"
        )

    print("\nSpecific pixel (u=2, v=0, Z=3):")
    print("Correct:", correct[1])
    print("Faulty: ", faulty[1])

    print(
        "\nEXP027 u/v fault injection "
        "PASSED."
    )


if __name__ == "__main__":
    main()
