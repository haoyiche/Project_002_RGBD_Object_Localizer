"""EXP035-B: controlled SE(3) changes with independent hand-computed oracles.

Run from Project_002_RGBD_Object_Localizer root:
    python -m src.experiments.exp035_transform_ablation_b

All transforms in this file are SYNTHETIC FOR EDUCATION, NOT CALIBRATED.
Never publish their numeric coordinates as physical robot poses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.exp035_se3_minimal import (
    BASE_FRAME,
    CAMERA_FRAME,
    EXPECTED_CAMERA_M,
    load_exp034_published_position,
    make_homogeneous,
    make_synthetic_transform,
)

WORLD_FRAME = "world"
TOL = 1e-12

# Independently hand-computed predictions from EXP035-B, not values derived
# from the implementation under test.
ORACLE_B0_BASE_M = np.array([0.29, 0.41, 1.352], dtype=np.float64)
ORACLE_B1_BASE_M = np.array([0.81, -0.09, 1.352], dtype=np.float64)
ORACLE_B1_DELTA_M = np.array([0.52, -0.50, 0.0], dtype=np.float64)
ORACLE_B2_BASE_M = np.array([-0.01, 0.51, 1.152], dtype=np.float64)
ORACLE_NORM_SQUARED_M2 = 1.587304
ORACLE_WORLD_M = np.array([-0.61, 0.69, 1.352], dtype=np.float64)
ORACLE_WORLD_FROM_CAMERA_R = np.diag([-1.0, -1.0, 1.0])
ORACLE_WORLD_FROM_CAMERA_T_M = np.array([-0.10, 0.70, 0.20], dtype=np.float64)
ORACLE_REVERSED_DEMO_M = np.array([-0.61, -0.31, 1.352], dtype=np.float64)


def make_world_from_base():
    """Synthetic T_world_from_base with another +90 degree Z rotation."""
    R = np.array([[0.0, -1.0, 0.0],
                  [1.0,  0.0, 0.0],
                  [0.0,  0.0, 1.0]], dtype=np.float64)
    t = np.array([-0.20, 0.40, 0.00], dtype=np.float64)
    return R, t


def compute_variants(p_camera_m):
    """Compare three controlled SE(3) interventions without changing the point.

    Returns NumPy arrays so tests can check exact shapes and computations.
    This is a minimal learning experiment; a strict general API is EXP035-C.
    """
    p = np.asarray(p_camera_m, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError("p_camera_m must be a finite (3,) vector in meters")

    R_bc, t_bc = make_synthetic_transform()
    R_wb, t_wb = make_world_from_base()

    # B0 (unchanged baseline from EXP035-A).
    T_bc = make_homogeneous(R_bc, t_bc)
    p_base_b0 = R_bc @ p + t_bc

    # B1: identity R and unchanged t; inverse is subtraction of t.
    R_b1 = np.eye(3, dtype=np.float64)
    T_b1 = make_homogeneous(R_b1, t_bc)
    p_base_b1 = R_b1 @ p + t_bc
    p_camera_back_b1 = p_base_b1 - t_bc

    # B2: unchanged R and zero t; Euclidean norm about the origin is invariant.
    t_b2 = np.zeros(3, dtype=np.float64)
    T_b2 = make_homogeneous(R_bc, t_b2)
    p_base_b2 = R_bc @ p + t_b2
    p_camera_back_b2 = R_bc.T @ p_base_b2

    # B3/B4: camera -> base -> world, composition right-to-left.
    T_wb = make_homogeneous(R_wb, t_wb)
    p_world_sequential = R_wb @ p_base_b0 + t_wb
    R_wc = R_wb @ R_bc
    t_wc = R_wb @ t_bc + t_wb
    T_wc = T_wb @ T_bc
    p_world_composed = R_wc @ p + t_wc
    p_world_homogeneous = (T_wc @ np.r_[p, 1.0])[:3]
    p_camera_back_b3 = (np.linalg.inv(T_wc) @ np.r_[p_world_sequential, 1.0])[:3]

    # Deliberately wrong: coordinate frames do not compose in this order.
    # Numeric result is shown ONLY as a debugging counterexample.
    T_reversed_wrong = T_bc @ T_wb
    p_world_reversed_wrong = (T_reversed_wrong @ np.r_[p, 1.0])[:3]

    return {
        "p_camera_m": p.copy(),
        "B0": {"p_base_m": p_base_b0, "T_base_from_camera": T_bc},
        "B1": {
            "p_base_m": p_base_b1,
            "p_camera_roundtrip_m": p_camera_back_b1,
            "delta_vs_A_m": p_base_b1 - p_base_b0,
            "T_base_from_camera": T_b1,
        },
        "B2": {
            "p_base_m": p_base_b2,
            "p_camera_roundtrip_m": p_camera_back_b2,
            "norm_squared_camera_m2": float(p @ p),
            "norm_squared_rotated_m2": float(p_base_b2 @ p_base_b2),
            "T_base_from_camera": T_b2,
        },
        "B3": {
            "p_base_m": p_base_b0,
            "p_world_sequential_m": p_world_sequential,
            "p_world_composed_m": p_world_composed,
            "p_world_homogeneous_m": p_world_homogeneous,
            "p_camera_roundtrip_m": p_camera_back_b3,
            "R_world_from_camera": R_wc,
            "t_world_from_camera_m": t_wc,
            "T_world_from_base": T_wb,
            "T_world_from_camera": T_wc,
            "T_reversed_wrong_demo": T_reversed_wrong,
            "p_reversed_wrong_demo_m": p_world_reversed_wrong,
        },
    }


def verify_hand_oracles(case):
    """Check against independent predictions and SE(3) invariants."""
    p = case["p_camera_m"]
    b0, b1, b2, b3 = (case[k] for k in ("B0", "B1", "B2", "B3"))
    np.testing.assert_allclose(p, EXPECTED_CAMERA_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b0["p_base_m"], ORACLE_B0_BASE_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b1["p_base_m"], ORACLE_B1_BASE_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b1["delta_vs_A_m"], ORACLE_B1_DELTA_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b1["p_camera_roundtrip_m"], p, rtol=0, atol=TOL)
    np.testing.assert_allclose(b2["p_base_m"], ORACLE_B2_BASE_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b2["norm_squared_camera_m2"], ORACLE_NORM_SQUARED_M2, rtol=0, atol=TOL)
    np.testing.assert_allclose(b2["norm_squared_rotated_m2"], ORACLE_NORM_SQUARED_M2, rtol=0, atol=TOL)
    np.testing.assert_allclose(b2["p_camera_roundtrip_m"], p, rtol=0, atol=TOL)
    for key in ("p_world_sequential_m", "p_world_composed_m", "p_world_homogeneous_m"):
        np.testing.assert_allclose(b3[key], ORACLE_WORLD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["R_world_from_camera"], ORACLE_WORLD_FROM_CAMERA_R, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["t_world_from_camera_m"], ORACLE_WORLD_FROM_CAMERA_T_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["p_camera_roundtrip_m"], p, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["T_world_from_camera"],
                               b3["T_world_from_base"] @ b0["T_base_from_camera"],
                               rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["T_world_from_camera"][:3, :3],
                               ORACLE_WORLD_FROM_CAMERA_R, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["T_world_from_camera"][:3, 3],
                               ORACLE_WORLD_FROM_CAMERA_T_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["p_reversed_wrong_demo_m"], ORACLE_REVERSED_DEMO_M,
                               rtol=0, atol=TOL)
    assert not np.allclose(b3["T_reversed_wrong_demo"], b3["T_world_from_camera"],
                           rtol=0, atol=TOL)
    # The two Z rotations commute, but the full SE(3) transforms do not.
    np.testing.assert_allclose(b3["T_reversed_wrong_demo"][:3, :3],
                               b3["T_world_from_camera"][:3, :3], rtol=0, atol=TOL)
    np.testing.assert_allclose(b3["p_reversed_wrong_demo_m"] - b3["p_world_sequential_m"],
                               [0.0, -1.0, 0.0], rtol=0, atol=TOL)
    return True


def compare_with_open3d(case):
    """Cross-check three full transformations using Open3D PointCloud.transform."""
    import open3d as o3d

    p = case["p_camera_m"]
    for key, transform_key, expected in (
        ("B1", "T_base_from_camera", ORACLE_B1_BASE_M),
        ("B2", "T_base_from_camera", ORACLE_B2_BASE_M),
        ("B3", "T_world_from_camera", ORACLE_WORLD_M),
    ):
        T = case[key][transform_key]
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(p.reshape(1, 3))
        cloud.transform(T)
        np.testing.assert_allclose(np.asarray(cloud.points)[0], expected, rtol=0, atol=TOL)
        cloud.transform(np.linalg.inv(T))
        np.testing.assert_allclose(np.asarray(cloud.points)[0], p, rtol=0, atol=TOL)
    print("Open3D:", o3d.__version__)
    print("Open3D B1/B2/B3 forward + inverse comparison: PASS")
    return True


def serialize_case(case):
    """JSON-friendly results. No field in the results confers publication rights."""
    p = case["p_camera_m"]
    b0, b1, b2, b3 = (case[k] for k in ("B0", "B1", "B2", "B3"))
    return {
        "experiment": "EXP035-B",
        "source_report": "results/exp034/exp034_C_validated_localization.json",
        "source_cluster_label": 1,
        "unit": "m",
        "source_frame_id": CAMERA_FRAME,
        "demo_target_frame_ids": [BASE_FRAME, WORLD_FRAME],
        "calibration_verified": False,
        "publishable": False,
        "position_m": None,
        "warning": "SYNTHETIC TRANSFORMS: numeric demonstration only, not verified physical robot poses.",
        "B0_baseline": {"p_base_demo_m": b0["p_base_m"].tolist()},
        "B1_translation_only": {
            "p_base_demo_m": b1["p_base_m"].tolist(),
            "delta_vs_A_m": b1["delta_vs_A_m"].tolist(),
            "p_camera_roundtrip_m": b1["p_camera_roundtrip_m"].tolist(),
        },
        "B2_rotation_only": {
            "p_base_demo_m": b2["p_base_m"].tolist(),
            "norm_camera_m": float(np.linalg.norm(p)),
            "norm_rotated_m": float(np.linalg.norm(b2["p_base_m"])),
            "norm_squared_m2": b2["norm_squared_camera_m2"],
            "p_camera_roundtrip_m": b2["p_camera_roundtrip_m"].tolist(),
        },
        "B3_two_stage": {
            "p_base_demo_m": b3["p_base_m"].tolist(),
            "p_world_sequential_demo_m": b3["p_world_sequential_m"].tolist(),
            "p_world_composed_demo_m": b3["p_world_composed_m"].tolist(),
            "p_world_homogeneous_demo_m": b3["p_world_homogeneous_m"].tolist(),
            "R_world_from_camera": b3["R_world_from_camera"].tolist(),
            "t_world_from_camera_m": b3["t_world_from_camera_m"].tolist(),
            "p_reversed_wrong_demo_m": b3["p_reversed_wrong_demo_m"].tolist(),
            "p_camera_roundtrip_m": b3["p_camera_roundtrip_m"].tolist(),
        },
        "B5_hypothetical_chain": {
            "note": "HYPOTHETICAL ONLY: suppose first edge verified; second edge unverified.",
            "transforms": [
                {"source_frame_id": CAMERA_FRAME, "target_frame_id": BASE_FRAME,
                 "verified": True},
                {"source_frame_id": BASE_FRAME, "target_frame_id": WORLD_FRAME,
                 "verified": False},
            ],
            "target_frame_id": WORLD_FRAME,
            "status": "BLOCKED_UNCALIBRATED",
            "publishable": False,
            "position_m": None,
            "reasons": ["UNVERIFIED_TRANSFORM: robot_base -> world"],
        },
        "note": "B0-B3 actual synthetic transforms are ALL unverified; B5 is a separate hypothetical example.",
    }


def run(project_root):
    root = Path(project_root)
    p = load_exp034_published_position(root)
    case = compute_variants(p)
    verify_hand_oracles(case)
    print("EXP034 published-position provenance: PASS | cluster_label=1")
    print("B1 translation-only oracle and inverse: PASS | base =", case["B1"]["p_base_m"].tolist())
    print("B2 rotation-only norm invariance and inverse: PASS | base =", case["B2"]["p_base_m"].tolist())
    print("B3 two-stage/compound/oracle/roundtrip: PASS | world =", case["B3"]["p_world_sequential_m"].tolist())
    print("B4 rotation blocks commute here, full SE(3) does not: PASS | reversed =",
          case["B3"]["p_reversed_wrong_demo_m"].tolist())
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("Open3D comparison: SKIPPED (not installed in this environment)")
        open3d_status = "SKIPPED"
    else:
        compare_with_open3d(case)
        open3d_status = "PASS"

    result = serialize_case(case)
    assert result["publishable"] is False and result["position_m"] is None
    assert result["B5_hypothetical_chain"]["publishable"] is False
    result["open3d_comparison"] = open3d_status
    out = root / "results/exp035/exp035_B_transform_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("B5 unverified chain blocks world-position publication: PASS")
    print("Saved:", out)
    print("EXP035-B NUMERICAL EXPERIMENT VERIFIED; EXP035 UNIT COMPLETION: PENDING")
    return result


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
