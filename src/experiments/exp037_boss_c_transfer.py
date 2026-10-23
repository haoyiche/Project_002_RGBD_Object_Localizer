"""
EXP037 Boss C: Transfer validation for Rx(90 degrees).

This experiment is fully independent of the B2/B3/B4 unit tests.
It verifies the X-axis rotation blind-spot derivation.
"""

import numpy as np

from exp037_multi_point_validation import compute_errors_mm


def main():
    # ---------------------------------------------------------
    # TODO 1: Create the three camera-frame points, shape (3, 3), unit: m
    # ---------------------------------------------------------
    points_camera = np.array([
        [0.50, 0.00, 0.00],   # P1: on the X axis (blind spot)
        [0.00, 0.10, 0.00],   # P2: off axis
        [0.00, 0.00, 0.20],   # P3: off axis
    ], dtype=np.float64)

    assert points_camera.shape == (3, 3)

    # ---------------------------------------------------------
    # TODO 2: Define reference and tested extrinsics
    # ---------------------------------------------------------
    R_true = np.eye(3, dtype=np.float64)
    t_true = np.zeros(3, dtype=np.float64)

    # Incorrect extrinsics: +90 degrees around X
    R_used = np.array([
        [1.0,  0.0,  0.0],
        [0.0,  0.0, -1.0],
        [0.0,  1.0,  0.0],
    ], dtype=np.float64)
    t_used = np.zeros(3, dtype=np.float64)

    # Sanity: R_used must be a valid rotation (orthogonal, det = +1)
    np.testing.assert_allclose(
        R_used.T @ R_used, np.eye(3), atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.det(R_used), 1.0, atol=1e-12
    )

    # ---------------------------------------------------------
    # TODO 3: Call compute_errors_mm()
    # ---------------------------------------------------------
    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t_true,
        R_used,
        t_used,
    )

    # ---------------------------------------------------------
    # TODO 4: Print signed error vectors and norms
    # ---------------------------------------------------------
    print("Error vectors (mm):")
    print(error_vectors_mm)

    print("Errors (mm):")
    print(errors_mm)

    print(f"Mean = {errors_mm.mean():.6f} mm")
    print(f"Max  = {errors_mm.max():.6f} mm")

    # ---------------------------------------------------------
    # TODO 5: Independent numeric assertions
    # ---------------------------------------------------------

    # --- Shape assertions ---
    assert error_vectors_mm.shape == (3, 3)
    assert errors_mm.shape == (3,)

    # --- Signed error vectors (mm) ---
    # These values are hand-derived from the X-axis rotation,
    # NOT copied from compute_errors_mm() output.
    expected_vectors_mm = np.array([
        [   0.0,    0.0,    0.0],   # P1: on X axis -> blind spot
        [   0.0, -100.0,  100.0],   # P2: rotated in YZ plane
        [   0.0, -200.0, -200.0],   # P3: rotated in YZ plane
    ], dtype=np.float64)

    np.testing.assert_allclose(
        error_vectors_mm,
        expected_vectors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # --- Euclidean errors (mm) ---
    expected_errors_mm = np.array([
        0.0,
        100.0 * np.sqrt(2.0),
        200.0 * np.sqrt(2.0),
    ], dtype=np.float64)

    np.testing.assert_allclose(
        errors_mm,
        expected_errors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # --- Mean and Max ---
    expected_mean_mm = (0.0 + 100.0 * np.sqrt(2.0) + 200.0 * np.sqrt(2.0)) / 3.0
    expected_max_mm = 200.0 * np.sqrt(2.0)

    np.testing.assert_allclose(
        errors_mm.mean(),
        expected_mean_mm,
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        errors_mm.max(),
        expected_max_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # --- Blind-spot structural assertion ---
    # P1 lies on the X axis; a rotation about X must not move it.
    np.testing.assert_allclose(
        error_vectors_mm[0],
        np.zeros(3),
        atol=1e-12,
    )

    # --- Off-axis points must have nonzero error ---
    assert errors_mm[1] > 0.0
    assert errors_mm[2] > 0.0

    # --- Distance-to-axis relationship ---
    # P2 distance to X axis = 0.10 m
    # P3 distance to X axis = 0.20 m
    # For 90-degree rotation: ||delta|| = r * sqrt(2) * 1000 mm
    r2 = np.linalg.norm(points_camera[1, 1:])
    r3 = np.linalg.norm(points_camera[2, 1:])
    np.testing.assert_allclose(errors_mm[1], r2 * np.sqrt(2.0) * 1000.0, atol=1e-9)
    np.testing.assert_allclose(errors_mm[2], r3 * np.sqrt(2.0) * 1000.0, atol=1e-9)

    print("\nAll independent assertions PASSED.")


if __name__ == "__main__":
    main()