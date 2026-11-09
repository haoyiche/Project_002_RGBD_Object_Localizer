"""T1-06-A: CPU-only FoundationPose/BOP coordinate and fair-input contract.

No FoundationPose inference, model registration, scene_gt access, or GPU usage.

FoundationPose's public register() return is already in ORIGINAL mesh coordinates:
    best_pose = poses[0] @ tf_to_centered_mesh
This module must NOT apply another origin correction to that public result.
"""
from __future__ import annotations

import json
import numpy as np

from src.evaluation.t1_04_a_pose_metrics import PoseM2C


def _points(x):
    try:
        a = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('INVALID_MODEL_POINTS') from exc
    if a.ndim != 2 or a.shape[1] != 3 or len(a) < 3 or not np.isfinite(a).all():
        raise ValueError('INVALID_MODEL_POINTS')
    return a


def bop_mesh_to_m(vertices, *, input_unit: str, size_mm):
    """Check BOP model extent against models_info size in mm, output Nx3 meters.

    Explicitly reject raw mm vertices falsely declared as meters. The tolerance
    of 1 mm is for checking metadata agreement, not measuring pose accuracy.
    """
    points = _points(vertices)
    try:
        size = np.asarray(size_mm, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('INVALID_MODEL_METADATA') from exc
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError('INVALID_MODEL_METADATA')
    if input_unit == 'mm':
        points_m = points / 1000.0
    elif input_unit == 'm':
        points_m = points.copy()
    else:
        raise ValueError('UNKNOWN_MODEL_INPUT_UNIT')
    actual_mm = np.ptp(points_m, axis=0) * 1000.0
    if not np.allclose(actual_mm, size, rtol=0, atol=1.0):
        raise ValueError('MODEL_SIZE_MM_MISMATCH')
    return points_m


def bbox_center_m(vertices_m):
    """FoundationPose uses (axis-wise min + max) / 2, not the mean vertex."""
    a = _points(vertices_m)
    return (np.min(a, axis=0) + np.max(a, axis=0)) / 2.0


def pose_4x4_to_original_m2c(T, *, origin: str, model_center_m=None) -> PoseM2C:
    """Turn a 4x4 model->camera pose into an original-model PoseM2C.

    origin='original_model': public FoundationPose register() result; NO shift.
    origin='centered_model': *internal* pose; subtract R@bbox_center once.
    """
    try:
        mat = np.asarray(T, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('INVALID_POSE_4X4') from exc
    if (mat.shape != (4, 4) or not np.isfinite(mat).all()
            or not np.allclose(mat[3], [0., 0., 0., 1.], rtol=0, atol=1e-10)):
        raise ValueError('INVALID_POSE_4X4')
    R = mat[:3, :3]
    t = mat[:3, 3]
    if origin == 'centered_model':
        if model_center_m is None:
            raise ValueError('MODEL_CENTER_REQUIRED')
        center = np.asarray(model_center_m, dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('INVALID_MODEL_CENTER')
        t = t - R @ center
    elif origin == 'original_model':
        if model_center_m is not None:
            raise ValueError('DO_NOT_CENTER_REGISTER_OUTPUT_AGAIN')
    else:
        raise ValueError('UNSUPPORTED_POSE_ORIGIN')
    return PoseM2C(R, t)


def assert_single_frame_comparison(*, baseline_ids, candidate_ids,
                                   baseline_mask_sha, candidate_mask_sha,
                                   baseline_mesh_sha, candidate_mesh_sha,
                                   baseline_mode, candidate_mode,
                                   baseline_reads_gt=False, candidate_reads_gt=False):
    """Preflight declarative information-equivalence check, not proof of behavior.

    Hash maps should originate from independently verified frozen manifests. A
    static configuration check cannot guarantee a third-party library never
    reads GT: runtime file-access/code audit is still required.
    """
    if baseline_reads_gt or candidate_reads_gt:
        raise ValueError('GT_POSE_LEAKAGE_IN_PREDICTION')
    if baseline_mode != 'register' or candidate_mode != 'register':
        raise ValueError('TEMPORAL_MODE_NOT_SINGLE_FRAME')
    if (list(baseline_ids) != list(candidate_ids)
            or dict(baseline_mask_sha) != dict(candidate_mask_sha)
            or baseline_mesh_sha != candidate_mesh_sha):
        raise ValueError('MISMATCHED_COMPARISON_INPUTS')
    if len(set(baseline_ids)) != len(baseline_ids) or not baseline_ids:
        raise ValueError('INVALID_COMPARISON_FRAME_IDS')
    if set(baseline_mask_sha) != set(baseline_ids):
        raise ValueError('INVALID_COMPARISON_MASK_HASH_MAP')
    return True


def smoke_demo():
    center = np.array([0.02, -0.01, 0.04])
    T = np.eye(4)
    T[:3, 3] = [0.10, 0.20, 0.50]
    a = pose_4x4_to_original_m2c(T, origin='centered_model', model_center_m=center)
    original = pose_4x4_to_original_m2c(
        np.array([[1,0,0,0.08],[0,1,0,0.21],[0,0,1,0.46],[0,0,0,1.0]]),
        origin='original_model')
    assert np.allclose(a.t_m, original.t_m)
    return {
        'schema': 't1-06-a-cpu-contract-smoke-v1',
        'centered_to_original_t_m': a.t_m.tolist(),
        'wrong_origin_shift_mm': float(np.linalg.norm(center) * 1000.0),
        'official_register_output_already_original_model': True,
        'foundationpose_inference_performed': False,
        'gt_pose_read': False,
        'pose_accuracy_claimed': False,
        'publishable': False,
        'status': 'MATH_CONTRACT_ONLY',
    }


if __name__ == '__main__':
    print(json.dumps(smoke_demo(), indent=2))
