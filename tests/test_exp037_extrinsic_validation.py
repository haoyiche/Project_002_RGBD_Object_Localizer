
"""EXP037-C4: Unit tests for extrinsic validation."""

import sys
from pathlib import Path

import numpy as np

# Make the experiment module importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = PROJECT_ROOT / "src" / "experiments"

sys.path.insert(0, str(EXPERIMENT_DIR))

from exp037_multi_point_validation import (
    CASES,
    compute_errors_mm,
)


def test_b2_rotation_blind_spot():
    """Z-axis points cannot reveal this Z rotation error."""

    # Input points: shape (3, 3), unit meters
    points_camera = np.array(
        CASES["B2"],
        dtype=np.float64,
    )

    # Reference rotation
    R_true = np.eye(3, dtype=np.float64)

    # Incorrect rotation: +90 degrees around Z
    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t = np.zeros(3, dtype=np.float64)
def test_b3_nonzero_error_mm():
    """B3: Detect nonzero rotation error and verify mm units."""

    # Three camera-frame points, unit: meters
    points_camera = np.array(
        CASES["B3"],
        dtype=np.float64,
    )

    # Synthetic reference extrinsics
    R_true = np.eye(3, dtype=np.float64)
    t_true = np.zeros(3, dtype=np.float64)

    # Incorrect extrinsics: +90 degrees around Z
    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t_used = np.zeros(3, dtype=np.float64)

    # Call the function under test
    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t_true,
        R_used,
        t_used,
    )

    # Expected signed error vectors, unit: mm
    expected_vectors_mm = np.array([
        [-100.0, 100.0, 0.0],
        [   0.0,   0.0, 0.0],
        [   0.0,   0.0, 0.0],
    ], dtype=np.float64)

    # Expected Euclidean errors, unit: mm
    expected_errors_mm = np.array([
        100.0 * np.sqrt(2.0),
        0.0,
        0.0,
    ], dtype=np.float64)

    # Shape checks
    assert error_vectors_mm.shape == (3, 3)
    assert errors_mm.shape == (3,)

    # Signed error vector check
    np.testing.assert_allclose(
        error_vectors_mm,
        expected_vectors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # Euclidean error check
    np.testing.assert_allclose(
        errors_mm,
        expected_errors_mm,
        rtol=0.0,
        atol=1e-9,
    )



def test_b4_blind_spot_samples_dilute_mean():
    """B4: Extra blind-spot points dilute mean, not max."""

    points_camera = np.array(
        CASES["B4"],
        dtype=np.float64,
    )

    R_true = np.eye(3, dtype=np.float64)

    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t = np.zeros(3, dtype=np.float64)

    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t,
        R_used,
        t,
    )

    # 10 points, each with 3 error components.
    assert error_vectors_mm.shape == (10, 3)
    assert errors_mm.shape == (10,)

    # P1 has nonzero error; P2-P10 are on the Z axis.
    expected_vectors_mm = np.zeros(
        (10, 3),
        dtype=np.float64,
    )
    expected_vectors_mm[0] = [-100.0, 100.0, 0.0]

    np.testing.assert_allclose(
        error_vectors_mm,
        expected_vectors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    expected_errors_mm = np.zeros(
        10,
        dtype=np.float64,
    )
    expected_errors_mm[0] = 100.0 * np.sqrt(2.0)

    np.testing.assert_allclose(
        errors_mm,
        expected_errors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # Statistics must match the independent expectations.
    expected_mean_mm = (100.0 * np.sqrt(2.0)) / 10.0
    expected_max_mm = 100.0 * np.sqrt(2.0)

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
  # Call the function independently
    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t,
        R_used,
        t,
    )

    # Check shapes
    assert error_vectors_mm.shape == (3, 3)
    assert errors_mm.shape == (3,)

    # Check numerical values
    np.testing.assert_allclose(
        error_vectors_mm,
        np.zeros((3, 3)),
        atol=1e-12,
    )

    np.testing.assert_allclose(
        errors_mm,
        np.zeros(3),
        atol=1e-12,
    )




def test_b3_nonzero_error_mm():
    """B3: Detect nonzero rotation error and verify mm units."""

    # Three camera-frame points, unit: meters
    points_camera = np.array(
        CASES["B3"],
        dtype=np.float64,
    )

    # Synthetic reference extrinsics
    R_true = np.eye(3, dtype=np.float64)
    t_true = np.zeros(3, dtype=np.float64)

    # Incorrect extrinsics: +90 degrees around Z
    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t_used = np.zeros(3, dtype=np.float64)

    # Call the function under test
    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t_true,
        R_used,
        t_used,
    )

    # Expected signed error vectors, unit: mm
    expected_vectors_mm = np.array([
        [-100.0, 100.0, 0.0],
        [   0.0,   0.0, 0.0],
        [   0.0,   0.0, 0.0],
    ], dtype=np.float64)

    # Expected Euclidean errors, unit: mm
    expected_errors_mm = np.array([
        100.0 * np.sqrt(2.0),
        0.0,
        0.0,
    ], dtype=np.float64)

    # Shape checks
    assert error_vectors_mm.shape == (3, 3)
    assert errors_mm.shape == (3,)

    # Signed error vector check
    np.testing.assert_allclose(
        error_vectors_mm,
        expected_vectors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # Euclidean error check
    np.testing.assert_allclose(
        errors_mm,
        expected_errors_mm,
        rtol=0.0,
        atol=1e-9,
    )



def test_b4_blind_spot_samples_dilute_mean():
    """B4: Extra blind-spot points dilute mean, not max."""

    points_camera = np.array(
        CASES["B4"],
        dtype=np.float64,
    )

    R_true = np.eye(3, dtype=np.float64)

    R_used = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ], dtype=np.float64)

    t = np.zeros(3, dtype=np.float64)

    error_vectors_mm, errors_mm = compute_errors_mm(
        points_camera,
        R_true,
        t,
        R_used,
        t,
    )

    # 10 points, each with 3 error components.
    assert error_vectors_mm.shape == (10, 3)
    assert errors_mm.shape == (10,)

    # P1 has nonzero error; P2-P10 are on the Z axis.
    expected_vectors_mm = np.zeros(
        (10, 3),
        dtype=np.float64,
    )
    expected_vectors_mm[0] = [-100.0, 100.0, 0.0]

    np.testing.assert_allclose(
        error_vectors_mm,
        expected_vectors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    expected_errors_mm = np.zeros(
        10,
        dtype=np.float64,
    )
    expected_errors_mm[0] = 100.0 * np.sqrt(2.0)

    np.testing.assert_allclose(
        errors_mm,
        expected_errors_mm,
        rtol=0.0,
        atol=1e-9,
    )

    # Statistics must match the independent expectations.
    expected_mean_mm = (100.0 * np.sqrt(2.0)) / 10.0
    expected_max_mm = 100.0 * np.sqrt(2.0)

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
