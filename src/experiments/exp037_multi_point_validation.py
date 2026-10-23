
"""EXP037-C3: Multi-point extrinsic validation.

Goals:
1. Reproduce B2 / B3 / B4 experiments.
2. Compute per-point extrinsic position errors.
3. Make compute_errors_mm() independently importable.
4. Avoid running experiments during module import.

Coordinate transform: camera -> world
Input coordinate unit: meters
Output error unit: millimeters

Note:
R_true represents synthetic reference extrinsics,
not independently verified physical calibration.
"""

import numpy as np


# ============================================================
# 1. Reproducible experiment configurations
# ============================================================

CASES = {
    # B2: Three points on the Z axis
    "B2": [
        [0.00, 0.00, 1.0],
        [0.00, 0.00, 1.5],
        [0.00, 0.00, 2.0],
    ],

    # B3: Move one point away from the Z axis
    "B3": [
        [0.10, 0.00, 1.0],
        [0.00, 0.00, 1.5],
        [0.00, 0.00, 2.0],
    ],

    # B4: Add seven more points on the Z axis
    "B4": [
        [0.10, 0.00, 1.0],
        [0.00, 0.00, 1.5],
        [0.00, 0.00, 2.0],
        [0.00, 0.00, 2.1],
        [0.00, 0.00, 2.2],
        [0.00, 0.00, 2.3],
        [0.00, 0.00, 2.4],
        [0.00, 0.00, 2.5],
        [0.00, 0.00, 2.6],
        [0.00, 0.00, 2.7],
    ],
}


# Select experiment
CASE_NAME = "B4"


# ============================================================
# 2. Core computation function
# ============================================================

def compute_errors_mm(
    points_camera,
    R_true,
    t_true,
    R_used,
    t_used,
):
    """Compute extrinsic position errors for multiple points.

    Parameters
    ----------
    points_camera : ndarray, shape (N, 3)
        Camera-frame points in meters.

    R_true : ndarray, shape (3, 3)
        Reference rotation matrix.

    t_true : ndarray, shape (3,)
        Reference translation in meters.

    R_used : ndarray, shape (3, 3)
        Rotation matrix used by the program.

    t_used : ndarray, shape (3,)
        Translation used by the program, in meters.

    Returns
    -------
    error_vectors_mm : ndarray, shape (N, 3)
        Signed XYZ position errors in millimeters.

    errors_mm : ndarray, shape (N,)
        Euclidean position errors in millimeters.
    """

    # Transform points using reference extrinsics
    points_true = np.array([
        R_true @ p + t_true
        for p in points_camera
    ])

    # Transform points using tested extrinsics
    points_used = np.array([
        R_used @ p + t_used
        for p in points_camera
    ])

    # Signed 3D errors: (N, 3), unit mm
    error_vectors_mm = (
        points_used - points_true
    ) * 1000.0

    # Euclidean errors: (N,), unit mm
    errors_mm = np.linalg.norm(
        error_vectors_mm,
        axis=1
    )

    return error_vectors_mm, errors_mm


# ============================================================
# 3. Main experiment
# ============================================================

def main():

    # Load selected experiment data
    points_camera = np.array(
        CASES[CASE_NAME],
        dtype=np.float64
    )

    # Reference extrinsics
    R_true = np.eye(3, dtype=np.float64)
    t_true = np.zeros(3, dtype=np.float64)

    # Incorrect extrinsics: +90 degrees around Z
    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t_used = np.zeros(3, dtype=np.float64)

    # Compute per-point errors
    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t_true,
        R_used,
        t_used,
    )

    # Display results
    print("Input shape:", points_camera.shape)
    print("Errors shape:", errors_mm.shape)

    for i, error in enumerate(errors_mm):
        print(f"P{i + 1} error (mm): {error:.6f}")

    print("Mean error (mm):", errors_mm.mean())
    print("Max error (mm):", errors_mm.max())


# ============================================================
# 4. Program entry point
# ============================================================

if __name__ == "__main__":
    main()
