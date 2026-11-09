"""T1-06-A: CPU-only unit/centered-vs-original/mode smoke tests."""
import numpy as np
import pytest

from src.evaluation.t1_04_a_pose_metrics import axis_angle_rotation
from src.evaluation.t1_06_a_pose_adapter_contract import (
    assert_single_frame_comparison,
    bbox_center_m,
    bop_mesh_to_m,
    pose_4x4_to_original_m2c,
    smoke_demo,
)


def tf(R=None, t=(0.1, 0.2, 0.5)):
    T = np.eye(4)
    T[:3, :3] = np.eye(3) if R is None else R
    T[:3, 3] = t
    return T


def test_a_identity_center_compensation_and_wrong_shift():
    c = [0.02, -0.01, 0.04]
    orig = pose_4x4_to_original_m2c(tf(), origin='centered_model', model_center_m=c)
    np.testing.assert_allclose(orig.t_m, [0.08, 0.21, 0.46], atol=1e-12)
    assert np.linalg.norm(c) * 1000 == pytest.approx(45.8257569495584)


def test_a_active_modification_rotation_90_deg_is_R_times_center():
    R = axis_angle_rotation([0, 0, 1], 90)
    c = np.array([0.02, -0.01, 0.04])
    orig = pose_4x4_to_original_m2c(tf(R), origin='centered_model', model_center_m=c)
    np.testing.assert_allclose(orig.t_m, [0.09, 0.18, 0.46], atol=1e-12)
    assert np.linalg.norm(orig.t_m - np.array([0.1, 0.2, 0.5])) * 1000 == pytest.approx(45.8257569495584)


def test_register_public_pose_must_not_be_shifted_again():
    p = pose_4x4_to_original_m2c(tf(t=[0.08, 0.21, 0.46]), origin='original_model')
    np.testing.assert_allclose(p.t_m, [0.08, 0.21, 0.46])
    with pytest.raises(ValueError, match='DO_NOT_CENTER_REGISTER_OUTPUT_AGAIN'):
        pose_4x4_to_original_m2c(tf(), origin='original_model', model_center_m=[0.02, 0, 0])


def test_bbox_center_not_mean_vertices():
    points = np.array([[0., 0., 0.], [0.10, 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    np.testing.assert_allclose(bbox_center_m(points), [0.05, 0., 0.])
    assert not np.allclose(bbox_center_m(points), points.mean(axis=0))


@pytest.mark.parametrize('unit', ['mm', 'm'])
def test_b_correct_units_are_normalized(unit):
    raw = np.array([[0,0,0],[80,0,0],[0,70,90]], dtype=float)
    vertices = raw if unit == 'mm' else raw / 1000.0
    meters = bop_mesh_to_m(vertices, input_unit=unit, size_mm=[80,70,90])
    np.testing.assert_allclose(np.ptp(meters, axis=0), [0.08,0.07,0.09])


def test_b_raw_mm_falsely_claimed_as_m_is_rejected():
    raw = np.array([[0,0,0],[80,0,0],[0,70,90]], dtype=float)
    with pytest.raises(ValueError, match='MODEL_SIZE_MM_MISMATCH'):
        bop_mesh_to_m(raw, input_unit='m', size_mm=[80,70,90])


def test_b_unknown_input_unit_rejected():
    with pytest.raises(ValueError, match='UNKNOWN_MODEL_INPUT_UNIT'):
        bop_mesh_to_m([[0,0,0],[1,0,0],[0,1,1]],input_unit='cm',size_mm=[1,1,1])


@pytest.mark.parametrize('bad', [
    np.eye(3),
    np.diag([1.,1.,1.,2.]),
    np.full((4,4),np.nan),
])
def test_invalid_pose_4x4_rejected(bad):
    with pytest.raises(ValueError, match='INVALID_POSE_4X4'):
        pose_4x4_to_original_m2c(bad, origin='original_model')


def test_so3_reflection_rejected():
    T = tf()
    T[:3,:3] = np.diag([1,1,-1])
    with pytest.raises(ValueError, match='INVALID_ROTATION_MATRIX'):
        pose_4x4_to_original_m2c(T, origin='original_model')


def same_frame_kwargs():
    return dict(baseline_ids=[15, 16], candidate_ids=[15, 16],
                baseline_mask_sha={15:'a',16:'b'},candidate_mask_sha={15:'a',16:'b'},
                baseline_mesh_sha='mesh',candidate_mesh_sha='mesh',
                baseline_mode='register',candidate_mode='register')


def test_c_register_register_is_declared_single_frame():
    assert assert_single_frame_comparison(**same_frame_kwargs()) is True


def test_c_track_rejected():
    d = same_frame_kwargs()
    d['candidate_mode'] = 'track'
    with pytest.raises(ValueError, match='TEMPORAL_MODE_NOT_SINGLE_FRAME'):
        assert_single_frame_comparison(**d)


def test_c_gt_leakage_rejected():
    with pytest.raises(ValueError, match='GT_POSE_LEAKAGE_IN_PREDICTION'):
        assert_single_frame_comparison(**same_frame_kwargs(), candidate_reads_gt=True)


def test_c_mismatched_frame_or_mask_rejected():
    d = same_frame_kwargs()
    d['candidate_mask_sha'] = {15:'a',16:'c'}
    with pytest.raises(ValueError, match='MISMATCHED_COMPARISON_INPUTS'):
        assert_single_frame_comparison(**d)
    d = same_frame_kwargs()
    d['candidate_ids'] = [15,17]
    with pytest.raises(ValueError, match='MISMATCHED_COMPARISON_INPUTS'):
        assert_single_frame_comparison(**d)


def test_smoke_is_formula_only_not_real_inference():
    r=smoke_demo()
    np.testing.assert_allclose(r['centered_to_original_t_m'],[.08,.21,.46])
    assert r['foundationpose_inference_performed'] is False
    assert r['gt_pose_read'] is False
    assert r['pose_accuracy_claimed'] is False
    assert r['publishable'] is False
