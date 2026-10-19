"""EXP035-A: hand prediction -> NumPy SE(3) -> Open3D verification.

Synthetic camera-to-base transform ONLY. No calibration; do not publish robot poses.
Run from the P02 project root: python -m src.experiments.exp035_se3_minimal
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMERA_FRAME = "camera_optical_frame"
BASE_FRAME = "robot_base"
EXPECTED_CAMERA_M = np.array([0.51, 0.01, 1.152], dtype=np.float64)
EXPECTED_ROTATED_M = np.array([-0.01, 0.51, 1.152], dtype=np.float64)
EXPECTED_BASE_M = np.array([0.29, 0.41, 1.352], dtype=np.float64)
EXPECTED_WRONG_M = np.array([0.09, 0.81, 1.352], dtype=np.float64)
CALIBRATION_VERIFIED = False  # The R/t below are synthetic teaching data.


def make_synthetic_transform():
    """R maps camera-axis coordinates to base axes; t is expressed in BASE_FRAME."""
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]], dtype=np.float64)
    t = np.array([0.30, -0.10, 0.20], dtype=np.float64)
    return R, t


def make_homogeneous(R, t):
    """Return T_base_from_camera; input shapes R=(3,3), t=(3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_exp034_published_position(project_root):
    """Read previously validated EXP034-C report, not a new raw cluster position."""
    file_path = Path(project_root) / "results/exp034/exp034_C_validated_localization.json"
    report = json.loads(file_path.read_text(encoding="utf-8"))
    if report.get("status") != "PARTIAL_INVALID":
        raise ValueError("Expected EXP034-C demonstration report with PARTIAL_INVALID")
    items = report.get("published_positions")
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError("EXP034-C should contain exactly one publishable object")
    obj = items[0]
    if obj.get("cluster_label") != 1 or obj.get("frame_id") != CAMERA_FRAME:
        raise ValueError("Unexpected cluster_label or frame_id in EXP034-C result")
    p = np.asarray(obj.get("position_m"), dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("EXP034-C position_m must be finite (3,) in meters")
    np.testing.assert_allclose(p, EXPECTED_CAMERA_M, rtol=0, atol=1e-12)
    return p


def calculate_numpy(p_camera_m, R, t):
    """Only the EXP035-A minimal calculation; strict general APIs come later."""
    p_base = R @ p_camera_m + t
    p_camera_back = R.T @ (p_base - t)
    rotated = R @ p_camera_m
    wrong_order = R @ (p_camera_m + t)
    return rotated, p_base, p_camera_back, wrong_order


def verify_hand_oracle(p_camera_m, R, t):
    """Explicit independently hand-computed constants from A1-A4."""
    np.testing.assert_allclose(p_camera_m, EXPECTED_CAMERA_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(R.T @ R, np.eye(3), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, rtol=0, atol=1e-12)
    rotated, p_base, p_back, wrong_order = calculate_numpy(p_camera_m, R, t)
    np.testing.assert_allclose(rotated, EXPECTED_ROTATED_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_base, EXPECTED_BASE_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_back, EXPECTED_CAMERA_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(wrong_order, EXPECTED_WRONG_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(wrong_order - p_base, [-0.20, 0.40, 0.00], rtol=0, atol=1e-12)
    T = make_homogeneous(R, t)
    T_inv = make_homogeneous(R.T, -R.T @ t)
    np.testing.assert_allclose(T_inv @ T, np.eye(4), rtol=0, atol=1e-12)
    np.testing.assert_allclose(T @ T_inv, np.eye(4), rtol=0, atol=1e-12)
    np.testing.assert_allclose(T @ np.r_[p_camera_m, 1], np.r_[EXPECTED_BASE_M, 1], rtol=0, atol=1e-12)
    # A direction vector has homogeneous w=0 and must not be translated.
    np.testing.assert_allclose(T @ [1, 0, 0, 0], [0, 1, 0, 0], rtol=0, atol=1e-12)
    return T, T_inv


def compare_with_open3d(p_camera_m, T, T_inv):
    """Independent implementation using Open3D's PointCloud.transform()."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p_camera_m.reshape(1, 3))
    cloud.transform(T)
    transformed = np.asarray(cloud.points).copy()[0]
    np.testing.assert_allclose(transformed, EXPECTED_BASE_M, rtol=0, atol=1e-12)
    cloud.transform(T_inv)
    recovered = np.asarray(cloud.points).copy()[0]
    np.testing.assert_allclose(recovered, EXPECTED_CAMERA_M, rtol=0, atol=1e-12)
    print("Open3D:", o3d.__version__)
    print("Open3D forward/inverse comparison: PASS")
    return transformed, recovered


def run(project_root):
    root = Path(project_root)
    p_camera_m = load_exp034_published_position(root)
    R, t = make_synthetic_transform()
    rotated, p_base, p_back, wrong_order = calculate_numpy(p_camera_m, R, t)
    T, T_inv = verify_hand_oracle(p_camera_m, R, t)
    print("EXP034 published-position provenance: PASS | cluster_label=1")
    print("Independent hand-computed oracle: PASS")
    print("R orthogonality / det / homogeneous inverse: PASS")
    print("A1 rotated XYZ:", rotated.tolist())
    print("A2 p_base XYZ:", p_base.tolist())
    print("A3 wrong-order XYZ:", wrong_order.tolist())
    print("A3 wrong minus correct (m):", (wrong_order - p_base).tolist())
    print("A4 inverse round-trip XYZ:", p_back.tolist())

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("Open3D comparison: SKIPPED (not installed in this environment)")
        open3d_status = "SKIPPED"
    else:
        compare_with_open3d(p_camera_m, T, T_inv)
        open3d_status = "PASS"

    assert CALIBRATION_VERIFIED is False, "This experiment must never claim a calibrated transform"
    result = {
        "experiment": "EXP035-A",
        "source_report": "results/exp034/exp034_C_validated_localization.json",
        "source_cluster_label": 1,
        "source_frame_id": CAMERA_FRAME,
        "target_frame_id": BASE_FRAME,
        "unit": "m",
        "calibration_verified": False,
        "publishable": False,
        "warning": "SYNTHETIC R/t ONLY; base-frame coordinates are for education, not robot control.",
        "R_base_from_camera": R.tolist(),
        "t_base_from_camera_m": t.tolist(),
        "T_base_from_camera": T.tolist(),
        "T_camera_from_base": T_inv.tolist(),
        "p_camera_m": p_camera_m.tolist(),
        "p_rotated_m": rotated.tolist(),
        "p_base_demo_m": p_base.tolist(),
        "p_camera_roundtrip_m": p_back.tolist(),
        "wrong_order_demo_m": wrong_order.tolist(),
        "wrong_minus_correct_m": (wrong_order - p_base).tolist(),
        "open3d_comparison": open3d_status,
    }
    out = root / "results/exp035/exp035_A_se3_minimal.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out)
    print("EXP035-A NUMERICAL EXPERIMENT VERIFIED; EXP035 UNIT COMPLETION: PENDING")
    return result


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
