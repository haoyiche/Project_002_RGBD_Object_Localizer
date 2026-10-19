"""EXP035-C: independent SE(3) interface and publication-safety tests."""
import importlib.util
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.experiments.exp035_se3_minimal import EXPECTED_CAMERA_M
from src.experiments.exp035_se3_contract_c import (
    ORACLE_BASE_M, ORACLE_WORLD_M, ORACLE_INVERSE_T_M,
    compare_with_open3d, make_demo_chain, verify_independent_oracles,
)
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position


@pytest.fixture
def chain():
    return make_demo_chain()


def test_input_source_target_frames(chain):
    bc, wb = chain
    assert (bc.source_frame_id, bc.target_frame_id) == ("camera_optical_frame", "robot_base")
    assert (wb.source_frame_id, wb.target_frame_id) == ("robot_base", "world")


def test_synthetic_calibration_false(chain):
    assert all(not t.calibration_verified for t in chain)


def test_hand_computed_oracle(chain):
    bc, _ = chain
    np.testing.assert_allclose(bc.transform_point(EXPECTED_CAMERA_M, frame_id="camera_optical_frame"),
                               ORACLE_BASE_M, rtol=0, atol=1e-12)


def test_composed_world_oracle(chain):
    bc, wb = chain
    wc = wb.compose(bc)
    np.testing.assert_allclose(wc.transform_point(EXPECTED_CAMERA_M, frame_id="camera_optical_frame"),
                               ORACLE_WORLD_M, rtol=0, atol=1e-12)


def test_sequential_vs_composed(chain):
    bc, wb = chain
    seq = wb.transform_point(bc.transform_point(EXPECTED_CAMERA_M, frame_id="camera_optical_frame"),
                             frame_id="robot_base")
    cmp_ = wb.compose(bc).transform_point(EXPECTED_CAMERA_M, frame_id="camera_optical_frame")
    np.testing.assert_allclose(seq, cmp_, rtol=0, atol=1e-12)


def test_expected_compound_rotation_and_translation(chain):
    bc, wb = chain
    wc = wb.compose(bc)
    np.testing.assert_allclose(wc.R, np.diag([-1, -1, 1]), rtol=0, atol=1e-12)
    np.testing.assert_allclose(wc.t_m, [-0.10, 0.70, 0.20], rtol=0, atol=1e-12)


def test_homogeneous_convention(chain):
    bc, wb = chain
    wc = wb.compose(bc)
    np.testing.assert_allclose(wc.as_matrix(), wb.as_matrix() @ bc.as_matrix(),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(wc.as_matrix()[3], [0, 0, 0, 1], rtol=0, atol=1e-12)


def test_inverse_translation_hand_oracle(chain):
    bc, _ = chain
    np.testing.assert_allclose(bc.inverse().t_m, ORACLE_INVERSE_T_M, rtol=0, atol=1e-12)


def test_inverse_frame_ids(chain):
    bc, _ = chain
    inv = bc.inverse()
    assert (inv.source_frame_id, inv.target_frame_id) == ("robot_base", "camera_optical_frame")


def test_inverse_round_trip(chain):
    bc, wb = chain
    wc = wb.compose(bc)
    pos = wc.transform_point(EXPECTED_CAMERA_M, frame_id="camera_optical_frame")
    np.testing.assert_allclose(wc.inverse().transform_point(pos, frame_id="world"),
                               EXPECTED_CAMERA_M, rtol=0, atol=1e-12)


def test_t_inverse_not_negative_t(chain):
    bc, _ = chain
    assert not np.allclose(bc.inverse().t_m, -bc.t_m, rtol=0, atol=1e-12)


def test_batch_hand_oracle(chain):
    bc, _ = chain
    points = np.repeat(EXPECTED_CAMERA_M[None, :], 100, axis=0)
    out = bc.transform_points(points, frame_id="camera_optical_frame")
    assert out.shape == (100, 3)
    np.testing.assert_allclose(out, np.repeat([ORACLE_BASE_M], 100, axis=0), rtol=0, atol=1e-12)


def test_batch_each_row_independently(chain):
    bc, _ = chain
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], dtype=float)
    actual = bc.transform_points(points, frame_id="camera_optical_frame")
    expected = np.array([[0.30, -0.10, 0.20], [0.30, 0.90, 0.20],
                         [-1.70, -0.10, 0.20]])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_batch_empty_allowed(chain):
    bc, _ = chain
    assert bc.transform_points(np.empty((0, 3)), frame_id="camera_optical_frame").shape == (0, 3)


def test_input_numpy_arrays_not_aliased(chain):
    R = np.eye(3)
    t = np.array([1., 2., 3.])
    obj = RigidTransform("a", "b", R, t)
    R[0, 0] = 9
    t[0] = 99
    np.testing.assert_array_equal(obj.R, np.eye(3))
    np.testing.assert_array_equal(obj.t_m, [1, 2, 3])


def test_properties_are_defensive_copies(chain):
    bc, _ = chain
    R = bc.R
    R[0, 0] = 999
    t = bc.t_m
    t[0] = 999
    assert bc.R[0, 0] == 0 and bc.t_m[0] == pytest.approx(0.30)


def test_frozen_frame_and_verification(chain):
    bc, _ = chain
    with pytest.raises(FrozenInstanceError):
        bc.calibration_verified = True
    with pytest.raises(FrozenInstanceError):
        bc.source_frame_id = "robot_base"


def test_reflection_is_orthogonal_but_rejected():
    R = np.diag([1., 1., -1.])
    np.testing.assert_array_equal(R.T @ R, np.eye(3))
    assert np.linalg.det(R) == pytest.approx(-1)
    with pytest.raises(ValueError, match="determinant"):
        RigidTransform("a", "b", R, [0, 0, 0])


@pytest.mark.parametrize("R", [
    np.diag([2., 1., 1.]),
    np.zeros((3, 3)),
    np.array([[1., .1, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_nonorthogonal_rejected(R):
    with pytest.raises(ValueError, match="orthogonal"):
        RigidTransform("a", "b", R, [0, 0, 0])


@pytest.mark.parametrize("R", [
    np.eye(2),
    np.zeros((4, 4)),
    [[1, 0], [0, 1]],
    [[1, 0, 0], [0, 1]],
])
def test_bad_rotation_shapes_rejected(R):
    with pytest.raises(ValueError, match="shape"):
        RigidTransform("a", "b", R, [0, 0, 0])


@pytest.mark.parametrize("t", [
    [0, 0],
    [[0, 0, 0]],
    [float("nan"), 0, 0],
    [0, float("inf"), 0],
    [True, 0, 0],
    ["0", 0, 0],
])
def test_bad_translation_rejected(t):
    with pytest.raises(ValueError):
        RigidTransform("a", "b", np.eye(3), t)


@pytest.mark.parametrize("frame", [None, "", "  ", 123, []])
def test_bad_frame_ids_rejected(frame):
    with pytest.raises(ValueError):
        RigidTransform(frame, "valid", np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError):
        RigidTransform("valid", frame, np.eye(3), [0, 0, 0])


@pytest.mark.parametrize("flag", [None, 0, 1, "true", np.bool_(True)])
def test_bad_verification_flag_rejected(flag):
    with pytest.raises(ValueError, match="bool"):
        RigidTransform("a", "b", np.eye(3), [0, 0, 0], calibration_verified=flag)


def test_disconnected_composition_rejected(chain):
    bc, _ = chain
    world_tool = RigidTransform("tool_frame", "world", np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError, match="frame mismatch"):
        world_tool.compose(bc)


def test_wrong_composition_order_rejected(chain):
    bc, wb = chain
    with pytest.raises(ValueError, match="frame mismatch"):
        bc.compose(wb)


def test_composition_rejects_non_transform(chain):
    bc, _ = chain
    with pytest.raises(ValueError):
        bc.compose(np.eye(4))


def test_wrong_single_point_frame_rejected(chain):
    bc, _ = chain
    with pytest.raises(ValueError, match="frame mismatch"):
        bc.transform_point([0, 0, 0], frame_id="world")


def test_wrong_batch_frame_rejected(chain):
    bc, _ = chain
    with pytest.raises(ValueError, match="frame mismatch"):
        bc.transform_points(np.zeros((5, 3)), frame_id="robot_base")


@pytest.mark.parametrize("bad", [
    [1, 2],
    [[1, 2, 3]],
    [0, float("nan"), 1],
    [0, 1, float("inf")],
    [True, 0, 0],
    ["1", 0, 0],
])
def test_bad_single_point_rejected(bad, chain):
    bc, _ = chain
    with pytest.raises(ValueError):
        bc.transform_point(bad, frame_id="camera_optical_frame")


@pytest.mark.parametrize("bad", [
    [1, 2, 3],
    np.ones((3, 1)),
    [[0, 0, float("nan")]],
    [[0, float("inf"), 0]],
    [[True, 0, 0]],
    [["1", 0, 0]],
])
def test_bad_batch_rejected(bad, chain):
    bc, _ = chain
    with pytest.raises(ValueError):
        bc.transform_points(bad, frame_id="camera_optical_frame")


def test_mixed_verified_chain_is_unverified(chain):
    bc, wb = chain
    only_bc = RigidTransform(bc.source_frame_id, bc.target_frame_id, bc.R, bc.t_m,
                             calibration_verified=True)
    assert wb.compose(only_bc).calibration_verified is False


def test_both_verified_claims_propagate(chain):
    bc, wb = chain
    bc = RigidTransform(bc.source_frame_id, bc.target_frame_id, bc.R, bc.t_m,
                        calibration_verified=True)
    wb = RigidTransform(wb.source_frame_id, wb.target_frame_id, wb.R, wb.t_m,
                        calibration_verified=True)
    assert wb.compose(bc).calibration_verified is True
    # Claims alone do not grant publication rights.
    rep = inspect_numeric_position(EXPECTED_CAMERA_M, frame_id="camera_optical_frame",
                                   transform=wb.compose(bc))
    assert rep["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert rep["publishable"] is False and rep["position_m"] is None


def test_unverified_inverse_remains_unverified(chain):
    bc, _ = chain
    assert bc.inverse().calibration_verified is False


def test_unverified_numeric_diagnostics_not_published(chain):
    bc, _ = chain
    rep = inspect_numeric_position(EXPECTED_CAMERA_M, frame_id="camera_optical_frame",
                                   transform=bc)
    assert rep["status"] == "BLOCKED_UNCALIBRATED"
    assert rep["publishable"] is False and rep["position_m"] is None
    np.testing.assert_allclose(rep["diagnostics"]["p_target_demo_m"], ORACLE_BASE_M,
                               rtol=0, atol=1e-12)
    assert rep["diagnostics"]["demo_only"] is True


def test_independent_oracle_verifier(chain):
    bc, wb, wc, base, world = verify_independent_oracles(EXPECTED_CAMERA_M)
    np.testing.assert_allclose(world, ORACLE_WORLD_M, rtol=0, atol=1e-12)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None,
                    reason="Open3D not installed in this environment")
def test_open3d_agrees_with_numpy(chain):
    bc, wb = chain
    assert compare_with_open3d(EXPECTED_CAMERA_M, bc, wb.compose(bc)) is True
