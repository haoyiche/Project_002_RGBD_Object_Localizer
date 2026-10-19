"""EXP035-C: strict SE(3) contract, using EXP034's validated B position.

All R/t values are synthetic. This experiment never publishes robot poses.
Run from P02: python -m src.experiments.exp035_se3_contract_c
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.exp035_se3_minimal import (
    BASE_FRAME, CAMERA_FRAME, EXPECTED_CAMERA_M, load_exp034_published_position,
    make_synthetic_transform,
)
from src.experiments.exp035_transform_ablation_b import (
    WORLD_FRAME, make_world_from_base,
)
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position

ORACLE_BASE_M = [0.29, 0.41, 1.352]
ORACLE_WORLD_M = [-0.61, 0.69, 1.352]
ORACLE_INVERSE_T_M = [0.10, 0.30, -0.20]


def make_demo_chain():
    R_bc, t_bc = make_synthetic_transform()
    R_wb, t_wb = make_world_from_base()
    bc = RigidTransform(CAMERA_FRAME, BASE_FRAME, R_bc, t_bc, calibration_verified=False)
    wb = RigidTransform(BASE_FRAME, WORLD_FRAME, R_wb, t_wb, calibration_verified=False)
    return bc, wb


def verify_independent_oracles(p):
    bc, wb = make_demo_chain()
    wc = wb.compose(bc)
    base = bc.transform_point(p, frame_id=CAMERA_FRAME)
    world_seq = wb.transform_point(base, frame_id=BASE_FRAME)
    world_comp = wc.transform_point(p, frame_id=CAMERA_FRAME)
    recovered = wc.inverse().transform_point(world_comp, frame_id=WORLD_FRAME)
    np.testing.assert_allclose(base, ORACLE_BASE_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(world_seq, ORACLE_WORLD_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(world_comp, ORACLE_WORLD_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(recovered, EXPECTED_CAMERA_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(bc.inverse().t_m, ORACLE_INVERSE_T_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(wc.as_matrix(), wb.as_matrix() @ bc.as_matrix(),
                               rtol=0, atol=1e-12)
    assert (bc.calibration_verified, wb.calibration_verified,
            wc.calibration_verified) == (False, False, False)
    assert bc.source_frame_id == CAMERA_FRAME and wc.target_frame_id == WORLD_FRAME
    return bc, wb, wc, base, world_comp


def compare_with_open3d(p, bc, wc):
    import open3d as o3d
    for transform, expected in ((bc, ORACLE_BASE_M), (wc, ORACLE_WORLD_M)):
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(np.asarray(p).reshape(1, 3))
        cloud.transform(transform.as_matrix())
        np.testing.assert_allclose(np.asarray(cloud.points)[0], expected, rtol=0, atol=1e-12)
        cloud.transform(transform.inverse().as_matrix())
        np.testing.assert_allclose(np.asarray(cloud.points)[0], EXPECTED_CAMERA_M,
                                   rtol=0, atol=1e-12)
    print("Open3D:", o3d.__version__)
    print("Open3D forward/inverse: PASS")
    return True


def run(project_root):
    root = Path(project_root)
    p = load_exp034_published_position(root)
    bc, wb, wc, base, world = verify_independent_oracles(p)
    print("EXP034 published-position provenance: PASS | cluster_label=1")
    print("C1-C4 strict SE(3), composition and independent oracles: PASS")
    reflection = np.diag([1.0, 1.0, -1.0])
    try:
        RigidTransform(CAMERA_FRAME, BASE_FRAME, reflection, bc.t_m)
    except ValueError:
        print("C1 reflection det=-1 rejected: PASS")
    else:
        raise AssertionError("reflection must be rejected")
    mismatched = RigidTransform("tool_frame", WORLD_FRAME, np.eye(3), np.zeros(3))
    try:
        mismatched.compose(bc)
    except ValueError:
        print("C2 disconnected frame chain rejected: PASS")
    else:
        raise AssertionError("mismatched frames must be rejected")
    batch = bc.transform_points(np.repeat(p[None, :], 100, axis=0), frame_id=CAMERA_FRAME)
    assert batch.shape == (100, 3)
    np.testing.assert_allclose(batch, np.repeat(base[None, :], 100, axis=0), atol=1e-12, rtol=0)
    print("C3 batch (100,3): PASS")
    np.testing.assert_allclose(bc.inverse().t_m, ORACLE_INVERSE_T_M, rtol=0, atol=1e-12)
    print("C4 inverse translation: PASS")
    report = inspect_numeric_position(p, frame_id=CAMERA_FRAME, transform=wc)
    assert report["status"] == "BLOCKED_UNCALIBRATED"
    assert report["publishable"] is False and report["position_m"] is None
    np.testing.assert_allclose(report["diagnostics"]["p_target_demo_m"], ORACLE_WORLD_M,
                               atol=1e-12, rtol=0)
    print("C5 synthetic transform blocks publication: PASS")
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("Open3D: SKIPPED (missing in current environment)")
        open3d_status = "SKIPPED"
    else:
        compare_with_open3d(p, bc, wc)
        open3d_status = "PASS"
    result = {
        "experiment": "EXP035-C",
        "source_cluster_label": 1,
        "source_frame_id": CAMERA_FRAME,
        "target_frame_id": WORLD_FRAME,
        "unit": "m",
        "transform_chain": [
            {"from": bc.source_frame_id, "to": bc.target_frame_id,
             "calibration_verified": bc.calibration_verified},
            {"from": wb.source_frame_id, "to": wb.target_frame_id,
             "calibration_verified": wb.calibration_verified},
        ],
        "p_camera_m": p.tolist(),
        "p_base_demo_m": base.tolist(),
        "p_world_demo_m": world.tolist(),
        "R_world_from_camera": wc.R.tolist(),
        "t_world_from_camera_m": wc.t_m.tolist(),
        "publication_report": report,
        "open3d_comparison": open3d_status,
        "warning": "Synthetic R/t are NOT physical extrinsics. Never publish as robot coordinates.",
    }
    output = root / "results/exp035/exp035_C_rigid_transform_contract.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved:", output)
    print("EXP035-C NUMERICAL CONTRACT VERIFIED; EXP035 UNIT COMPLETION: PENDING")
    return result


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
