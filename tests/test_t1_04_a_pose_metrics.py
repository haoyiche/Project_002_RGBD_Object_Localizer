"""T1-04-A: numeric pose diagnostics, units, frames, SO(3) and safety."""
import math

import numpy as np
import pytest

from src.evaluation.t1_04_a_pose_metrics import (
    PoseM2C, axis_angle_rotation, compare_pose_m2c, controlled_demo,
)


def test_identical_pose_has_zero_error():
    gt = PoseM2C(axis_angle_rotation([1, 0, 0], 22), [0.02, -0.01, 0.8])
    result = compare_pose_m2c(gt, gt)
    assert result["translation_error_mm"] == 0.0
    assert result["rotation_error_deg_raw"] == pytest.approx(0.0, abs=1e-10)
    assert result["symmetry_aware"] is False


def test_bop_mm_to_m_then_translate_five_mm():
    gt = PoseM2C.from_bop(np.eye(3), [10.0, 20.0, 500.0])
    np.testing.assert_allclose(gt.t_m, [0.01, 0.02, 0.5])
    estimate = PoseM2C(np.eye(3), gt.t_m + [0.005, 0, 0])
    got = compare_pose_m2c(estimate, gt)
    assert got["translation_error_mm"] == pytest.approx(5.0)
    assert got["rotation_error_deg_raw"] == pytest.approx(0.0)


def test_three_axis_translation_euclidean_not_component_max():
    gt = PoseM2C(np.eye(3), [0, 0, 0])
    estimate = PoseM2C(np.eye(3), [0.003, 0.004, 0])
    assert compare_pose_m2c(estimate, gt)["translation_error_mm"] == pytest.approx(5.0)


@pytest.mark.parametrize("degrees", [1e-6, 10.0, 90.0, 180.0])
def test_rotation_geodesic_handles_near_zero_and_pi(degrees):
    gt = PoseM2C(np.eye(3), [0, 0, 0.5])
    est = PoseM2C(axis_angle_rotation([0, 0, 1], degrees), [0, 0, 0.5])
    result = compare_pose_m2c(est, gt)
    assert result["rotation_error_deg_raw"] == pytest.approx(degrees, abs=2e-8)
    assert result["translation_error_mm"] == 0.0


def test_common_camera_left_rotation_preserves_relative_rotation_angle():
    base = PoseM2C(axis_angle_rotation([1, 0, 0], 40), [0.1, 0.2, 0.3])
    est = PoseM2C(axis_angle_rotation([0, 1, 0], 13) @ base.R, base.t_m)
    Q = axis_angle_rotation([0, 0, 1], 71)
    a = compare_pose_m2c(est, base)["rotation_error_deg_raw"]
    b = compare_pose_m2c(
        PoseM2C(Q @ est.R, est.t_m), PoseM2C(Q @ base.R, base.t_m)
    )["rotation_error_deg_raw"]
    assert a == pytest.approx(13.0)
    assert b == pytest.approx(a)


def test_inverse_frame_annotation_rejected_even_if_matrix_valid():
    with pytest.raises(ValueError, match="INVALID_POSE_FRAME_CONVENTION"):
        PoseM2C(np.eye(3), [0, 0, 1], source_frame="camera", target_frame="model")


def test_wrong_bop_translation_unit_fails_closed():
    with pytest.raises(ValueError, match="BOP_TRANSLATION_UNIT_MUST_BE_MM"):
        PoseM2C.from_bop(np.eye(3), [10, 20, 500], translation_unit="m")


@pytest.mark.parametrize("R,error", [
    (np.diag([1, 1, -1]), "INVALID_ROTATION_MATRIX"),
    (np.diag([1, 1, 2]), "INVALID_ROTATION_MATRIX"),
    (np.full((3, 3), float('nan')), "NONFINITE_ROTATION"),
    (np.eye(4), "INVALID_ROTATION_SHAPE"),
])
def test_bad_rotations_fail_closed(R, error):
    with pytest.raises(ValueError, match=error):
        PoseM2C(R, [0, 0, 0.5])


def test_bad_translation_shapes_and_values_fail_closed():
    with pytest.raises(ValueError, match="INVALID_TRANSLATION_SHAPE"):
        PoseM2C(np.eye(3), [[0], [0], [1]])
    with pytest.raises(ValueError, match="NONFINITE_TRANSLATION"):
        PoseM2C(np.eye(3), [0, 0, float('inf')])


def test_arrays_are_copied_and_read_only():
    R = np.eye(3)
    t = np.array([0.1, 0.2, 0.3])
    p = PoseM2C(R, t)
    R[0, 0] = 9
    t[0] = 9
    assert p.R[0, 0] == 1
    assert p.t_m[0] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        p.t_m[0] = 0


def test_raw_180_rotation_is_not_symmetry_aware():
    gt = PoseM2C(np.eye(3), [0, 0, 0.5])
    est = PoseM2C(axis_angle_rotation([0, 0, 1], 180), [0, 0, 0.5])
    got = compare_pose_m2c(est, gt)
    assert got["rotation_error_deg_raw"] == pytest.approx(180.0)
    assert got["symmetry_aware"] is False
    with pytest.raises(ValueError, match="UNSUPPORTED_SYMMETRY_POLICY"):
        compare_pose_m2c(est, gt, symmetry_policy="symmetry_aware")


def test_rejects_untyped_pose_dict():
    gt = PoseM2C(np.eye(3), [0, 0, 0.5])
    with pytest.raises(ValueError, match="POSE_M2C_REQUIRED"):
        compare_pose_m2c({"R": np.eye(3)}, gt)


def test_gt_derived_demo_never_claims_real_pose_estimation():
    r = controlled_demo()
    assert r["translation_error_mm"] == pytest.approx(math.sqrt(29))
    assert r["rotation_error_deg_raw"] == pytest.approx(10.0)
    assert r["gt_used_to_construct_candidate"] is True
    assert r["estimation_performed"] is False
    assert r["pose_accuracy_claimed"] is False
    assert r["physical_accuracy_claimed"] is False
    assert r["publishable"] is False
    assert r["position_m"] is None
