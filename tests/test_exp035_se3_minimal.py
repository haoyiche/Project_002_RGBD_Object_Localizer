"""EXP035-A: independent numerical oracle, 4x4 matrix, direction, provenance."""
import copy
import importlib.util
import json

import numpy as np
import pytest

from src.experiments.exp035_se3_minimal import (
    CAMERA_FRAME,
    EXPECTED_BASE_M,
    EXPECTED_CAMERA_M,
    EXPECTED_ROTATED_M,
    EXPECTED_WRONG_M,
    calculate_numpy,
    compare_with_open3d,
    load_exp034_published_position,
    make_homogeneous,
    make_synthetic_transform,
    verify_hand_oracle,
)


def make_report(tmp_path):
    root = tmp_path
    path = root / "results/exp034/exp034_C_validated_localization.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "PARTIAL_INVALID",
        "published_positions": [{
            "cluster_label": 1,
            "frame_id": CAMERA_FRAME,
            "position_m": [0.51, 0.01, 1.152],
        }],
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return root, path, report


def test_shape_and_dtype():
    p = EXPECTED_CAMERA_M
    R, t = make_synthetic_transform()
    T = make_homogeneous(R, t)
    assert (p.shape, R.shape, t.shape, T.shape) == ((3,), (3, 3), (3,), (4, 4))
    assert p.dtype == np.float64 and R.dtype == np.float64 and t.dtype == np.float64


def test_a1_rotated_oracle():
    R, t = make_synthetic_transform()
    rotated, _, _, _ = calculate_numpy(EXPECTED_CAMERA_M, R, t)
    np.testing.assert_allclose(rotated, EXPECTED_ROTATED_M, atol=1e-12, rtol=0)


def test_a2_transformed_oracle():
    R, t = make_synthetic_transform()
    _, p_base, _, _ = calculate_numpy(EXPECTED_CAMERA_M, R, t)
    np.testing.assert_allclose(p_base, EXPECTED_BASE_M, atol=1e-12, rtol=0)


def test_a3_wrong_order_oracle():
    R, t = make_synthetic_transform()
    _, p_base, _, wrong = calculate_numpy(EXPECTED_CAMERA_M, R, t)
    np.testing.assert_allclose(wrong, EXPECTED_WRONG_M, atol=1e-12, rtol=0)
    np.testing.assert_allclose(wrong - p_base, [-0.20, 0.40, 0], atol=1e-12, rtol=0)


def test_a4_inverse_round_trip():
    R, t = make_synthetic_transform()
    _, _, p_back, _ = calculate_numpy(EXPECTED_CAMERA_M, R, t)
    np.testing.assert_allclose(p_back, EXPECTED_CAMERA_M, atol=1e-12, rtol=0)


def test_homogeneous_and_inverses():
    R, t = make_synthetic_transform()
    T, T_inv = verify_hand_oracle(EXPECTED_CAMERA_M, R, t)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), atol=1e-12, rtol=0)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=1e-12, rtol=0)


def test_direction_ignores_translation():
    R, t = make_synthetic_transform()
    T = make_homogeneous(R, t)
    np.testing.assert_allclose(T @ [1.0, 0, 0, 0], [0, 1, 0, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(T @ [1.0, 0, 0, 1], [0.3, 0.9, 0.2, 1], atol=1e-12, rtol=0)


def test_rotation_matrix_properties():
    R, _ = make_synthetic_transform()
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-12, rtol=0)


def test_saved_exp034_provenance(tmp_path):
    root, _, _ = make_report(tmp_path)
    np.testing.assert_allclose(load_exp034_published_position(root), EXPECTED_CAMERA_M, atol=1e-12, rtol=0)


@pytest.mark.parametrize("change", [
    {"status": "ALL_INVALID"},
    {"published_positions": []},
    {"published_positions": [{"cluster_label": 0, "frame_id": CAMERA_FRAME, "position_m": [0.51, 0.01, 1.152]}]},
    {"published_positions": [{"cluster_label": 1, "frame_id": "robot_base", "position_m": [0.51, 0.01, 1.152]}]},
    {"published_positions": [{"cluster_label": 1, "frame_id": CAMERA_FRAME, "position_m": [float("nan"), 0.01, 1.152]}]},
])
def test_reject_unexpected_exp034_source(tmp_path, change):
    root, path, original = make_report(tmp_path)
    changed = copy.deepcopy(original)
    changed.update(change)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises((ValueError, AssertionError)):
        load_exp034_published_position(root)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None, reason="Open3D not installed")
def test_open3d_comparison():
    R, t = make_synthetic_transform()
    T, T_inv = verify_hand_oracle(EXPECTED_CAMERA_M, R, t)
    base, back = compare_with_open3d(EXPECTED_CAMERA_M, T, T_inv)
    np.testing.assert_allclose(base, EXPECTED_BASE_M, atol=1e-12, rtol=0)
    np.testing.assert_allclose(back, EXPECTED_CAMERA_M, atol=1e-12, rtol=0)
