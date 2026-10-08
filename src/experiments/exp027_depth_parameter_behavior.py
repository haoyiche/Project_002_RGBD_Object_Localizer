import numpy as np

from src.geometry.depth_to_pointcloud import depth_to_pointcloud


def run(depth_m, cx=1.0):
    return depth_to_pointcloud(
        depth_m=depth_m,
        fx=2.0,
        fy=2.0,
        cx=cx,
        cy=0.0,
    )


def main():
    print("=== EXP027 Parameter Behavior ===")

    baseline_depth = np.array([
        [1.0, 1.0, 0.0],
        [2.0, 2.0, 2.0],
    ])

    baseline = run(baseline_depth)

    # ========================================
    # Case A: invalid -> valid
    # ========================================

    print("\n=== Case A ===")

    depth_a = baseline_depth.copy()
    depth_a[0, 2] = 3.0

    points_a = run(depth_a)

    print("Shape:", points_a.shape)
    print(points_a)

    expected_a = np.array([
        [-0.5, 0.0, 1.0],
        [ 0.0, 0.0, 1.0],
        [ 1.5, 0.0, 3.0],
        [-1.0, 1.0, 2.0],
        [ 0.0, 1.0, 2.0],
        [ 1.0, 1.0, 2.0],
    ])

    np.testing.assert_allclose(
        points_a, expected_a, rtol=0, atol=1e-12
    )

    assert points_a.shape == (6, 3)

    print("Case A PASS")

    # ========================================
    # Case B: invalid depth filtering
    # ========================================

    print("\n=== Case B ===")

    depth_b = np.array([
        [np.nan, np.inf, -1.0],
        [0.0,    2.0,     3.0],
    ])

    points_b = run(depth_b)

    print("Shape:", points_b.shape)
    print(points_b)

    expected_b = np.array([
        [0.0, 1.0, 2.0],
        [1.5, 1.5, 3.0],
    ])

    np.testing.assert_allclose(
        points_b, expected_b, rtol=0, atol=1e-12
    )

    assert points_b.shape == (2, 3)

    print("Case B PASS")

    # ========================================
    # Case C: cx = 1 -> 0
    # ========================================

    print("\n=== Case C ===")

    points_c = run(baseline_depth, cx=0.0)

    print("Baseline:")
    print(baseline)

    print("New points:")
    print(points_c)

    delta = points_c - baseline

    print("Delta:")
    print(delta)

    expected_delta_x = baseline[:, 2] / 2.0

    np.testing.assert_allclose(
        delta[:, 0],
        expected_delta_x,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        delta[:, 1:],
        np.zeros_like(delta[:, 1:]),
        rtol=0,
        atol=1e-12,
    )

    assert points_c.shape == baseline.shape

    print("Case C PASS")

    print("\nAll EXP027 parameter checks PASSED.")


if __name__ == "__main__":
    main()
