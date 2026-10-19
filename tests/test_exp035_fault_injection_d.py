"""EXP035-D: isolated five-fault RED/PATCH/GREEN regression tests.

All mutant implementations are intentional failures and are never installed
into src/geometry/rigid_transform.py.
"""
import importlib.util

import numpy as np
import pytest

from src.experiments.exp035_se3_minimal import CAMERA_FRAME, BASE_FRAME, EXPECTED_CAMERA_M
from src.experiments.exp035_se3_contract_c import make_demo_chain
from src.experiments.exp035_fault_injection_d import (
    TOL, ORACLE_BASE_M, ORACLE_WORLD_M,
    ORACLE_D1_BAD_M, ORACLE_D2_BAD_M,
    ORACLE_D3_BAD_T_M, ORACLE_D3_BAD_WORLD_M,
    mutant_d1_batch_wrong_transpose, mutant_d2_inverse_wrong_t,
    mutant_d3_compose_wrong_t, mutant_d4_ignore_frame,
    mutant_d5_publish_on_bool_claim, verify_red, verify_green,
    expect_red_failure, compare_with_open3d,
)
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position


@pytest.fixture
def setup():
    bc, wb = make_demo_chain()
    return EXPECTED_CAMERA_M.copy(), bc, wb


def test_mutant_red_detects_all_five(setup):
    p, bc, wb = setup
    assert verify_red(p, bc, wb) == {f"D{i}": True for i in range(1, 6)}


@pytest.mark.parametrize("tag", ["D1", "D2", "D3", "D4", "D5"])
def test_each_red_oracle_is_present(tag, setup):
    p, bc, wb = setup
    assert verify_red(p, bc, wb)[tag] is True


def test_red_helper_rejects_undetected_mutation():
    with pytest.raises(AssertionError, match="not detected"):
        expect_red_failure("undetected", lambda: None)


def test_red_helper_catches_expected_assertion():
    assert expect_red_failure("detected", lambda: (_ for _ in ()).throw(AssertionError()))


def test_d1_bad_output_equals_hand_oracle(setup):
    p, bc, _ = setup
    bad = mutant_d1_batch_wrong_transpose(bc, p[None, :])
    assert bad.shape == (1, 3)
    np.testing.assert_allclose(bad[0], ORACLE_D1_BAD_M, rtol=0, atol=TOL)


def test_d1_correct_output_and_delta(setup):
    p, bc, _ = setup
    right = bc.transform_points(p[None, :], frame_id=CAMERA_FRAME)[0]
    bad = mutant_d1_batch_wrong_transpose(bc, p[None, :])[0]
    np.testing.assert_allclose(right, ORACLE_BASE_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(bad - right, [0.02, -1.02, 0.0], rtol=0, atol=TOL)


@pytest.mark.parametrize("n", [1, 3, 100])
def test_d1_wrong_batch_for_any_n(n, setup):
    p, bc, _ = setup
    pts = np.repeat(p[None, :], n, axis=0)
    wrong = mutant_d1_batch_wrong_transpose(bc, pts)
    right = bc.transform_points(pts, frame_id=CAMERA_FRAME)
    assert wrong.shape == right.shape == (n, 3)
    assert not np.allclose(wrong, right, rtol=0, atol=TOL)


def test_d2_bad_inverse_oracle(setup):
    p, bc, _ = setup
    base = bc.transform_point(p, frame_id=CAMERA_FRAME)
    bad = mutant_d2_inverse_wrong_t(bc).transform_point(base, frame_id=BASE_FRAME)
    np.testing.assert_allclose(bad, ORACLE_D2_BAD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(bad - p, [-0.40, -0.20, 0.0], rtol=0, atol=TOL)


def test_d2_correct_roundtrip(setup):
    p, bc, _ = setup
    base = bc.transform_point(p, frame_id=CAMERA_FRAME)
    back = bc.inverse().transform_point(base, frame_id=BASE_FRAME)
    np.testing.assert_allclose(back, p, rtol=0, atol=TOL)
    np.testing.assert_allclose(bc.inverse().t_m, [0.10, 0.30, -0.20], rtol=0, atol=TOL)


def test_d2_bad_inverse_still_orthogonal(setup):
    _, bc, _ = setup
    wrong = mutant_d2_inverse_wrong_t(bc)
    np.testing.assert_allclose(wrong.R.T @ wrong.R, np.eye(3), rtol=0, atol=TOL)
    assert np.linalg.det(wrong.R) == pytest.approx(1.0)


def test_d3_bad_translation_oracle(setup):
    _, bc, wb = setup
    np.testing.assert_allclose(mutant_d3_compose_wrong_t(wb, bc).t_m,
                               ORACLE_D3_BAD_T_M, rtol=0, atol=TOL)


def test_d3_bad_world_oracle_and_delta(setup):
    p, bc, wb = setup
    bad = mutant_d3_compose_wrong_t(wb, bc).transform_point(p, frame_id=CAMERA_FRAME)
    good = wb.compose(bc).transform_point(p, frame_id=CAMERA_FRAME)
    np.testing.assert_allclose(bad, ORACLE_D3_BAD_WORLD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(good, ORACLE_WORLD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(bad - good, [0.20, -0.40, 0.00], rtol=0, atol=TOL)


def test_d3_rotation_alone_cannot_detect_mutant(setup):
    _, bc, wb = setup
    wrong = mutant_d3_compose_wrong_t(wb, bc)
    good = wb.compose(bc)
    np.testing.assert_allclose(wrong.R, good.R, rtol=0, atol=TOL)
    assert not np.allclose(wrong.t_m, good.t_m, rtol=0, atol=TOL)


def test_d3_correct_sequential_equals_composed(setup):
    p, bc, wb = setup
    sequential = wb.transform_point(bc.transform_point(p, frame_id=CAMERA_FRAME),
                                    frame_id=BASE_FRAME)
    composite = wb.compose(bc).transform_point(p, frame_id=CAMERA_FRAME)
    np.testing.assert_allclose(sequential, composite, rtol=0, atol=TOL)


def test_d4_mutant_silently_accepts_wrong_frame(setup):
    p, bc, _ = setup
    bad = mutant_d4_ignore_frame(bc, p, frame_id="world")
    np.testing.assert_allclose(bad, ORACLE_BASE_M, rtol=0, atol=TOL)


@pytest.mark.parametrize("frame", ["world", "robot_base", "tool_frame"])
def test_d4_core_rejects_wrong_frames(frame, setup):
    p, bc, _ = setup
    with pytest.raises(ValueError, match="frame mismatch"):
        bc.transform_point(p, frame_id=frame)


def test_d4_core_rejects_wrong_batch_frame(setup):
    p, bc, _ = setup
    with pytest.raises(ValueError, match="frame mismatch"):
        bc.transform_points(p[None, :], frame_id="world")


def test_d5_mutant_flag_is_unsafe(setup):
    p, bc, _ = setup
    claim = RigidTransform(CAMERA_FRAME, BASE_FRAME, bc.R, bc.t_m,
                           calibration_verified=True)
    bad = mutant_d5_publish_on_bool_claim(claim, p)
    assert bad["publishable"] is True
    np.testing.assert_allclose(bad["position_m"], ORACLE_BASE_M, rtol=0, atol=TOL)


def test_d5_production_numeric_inspector_blocks_true_claim(setup):
    p, bc, _ = setup
    claim = RigidTransform(CAMERA_FRAME, BASE_FRAME, bc.R, bc.t_m,
                           calibration_verified=True)
    good = inspect_numeric_position(p, frame_id=CAMERA_FRAME, transform=claim)
    assert good["publishable"] is False and good["position_m"] is None
    assert good["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert "GLOBAL_VALIDATION_REQUIRED" in good["reasons"]
    assert good["diagnostics"]["demo_only"] is True


def test_d5_false_claim_is_blocked(setup):
    p, bc, _ = setup
    good = inspect_numeric_position(p, frame_id=CAMERA_FRAME, transform=bc)
    assert good["publishable"] is False and good["position_m"] is None
    assert good["status"] == "BLOCKED_UNCALIBRATED"


def test_green_checks_all_five(setup):
    p, bc, wb = setup
    assert verify_green(p, bc, wb) is True


def test_mutants_never_modify_source_transforms(setup):
    p, bc, wb = setup
    R0, t0 = bc.R.copy(), bc.t_m.copy()
    R1, t1 = wb.R.copy(), wb.t_m.copy()
    _ = mutant_d1_batch_wrong_transpose(bc, p[None, :])
    _ = mutant_d2_inverse_wrong_t(bc)
    _ = mutant_d3_compose_wrong_t(wb, bc)
    _ = mutant_d4_ignore_frame(bc, p, frame_id="world")
    _ = mutant_d5_publish_on_bool_claim(bc, p)
    np.testing.assert_array_equal(bc.R, R0)
    np.testing.assert_array_equal(bc.t_m, t0)
    np.testing.assert_array_equal(wb.R, R1)
    np.testing.assert_array_equal(wb.t_m, t1)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None,
                    reason="Open3D is available in user's ai_3d Windows environment")
def test_open3d_correct_two_stage_comparison(setup):
    p, bc, wb = setup
    assert compare_with_open3d(p, bc, wb) is True
