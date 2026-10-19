"""EXP035-D: isolated SE(3) fault injection, RED -> PATCH -> GREEN.

The five MUTANT functions below are deliberately wrong and only for controlled
training. They never modify src/geometry/rigid_transform.py or issue robot poses.

Run in P02: python -m src.experiments.exp035_fault_injection_d
All R/t are SYNTHETIC and NOT CALIBRATED for real-world deployment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.exp035_se3_minimal import (
    CAMERA_FRAME, BASE_FRAME, EXPECTED_CAMERA_M, load_exp034_published_position,
)
from src.experiments.exp035_se3_contract_c import make_demo_chain
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position

TOL = 1e-12
ORACLE_BASE_M = [0.29, 0.41, 1.352]
ORACLE_WORLD_M = [-0.61, 0.69, 1.352]
ORACLE_D1_BAD_M = [0.31, -0.61, 1.352]
ORACLE_D2_BAD_M = [0.11, -0.19, 1.152]
ORACLE_D3_BAD_T_M = [0.10, 0.30, 0.20]
ORACLE_D3_BAD_WORLD_M = [-0.41, 0.29, 1.352]


def mutant_d1_batch_wrong_transpose(transform: RigidTransform, points_m):
    """MUTANT: mistakenly treat row vectors as column vectors (wrong R)."""
    return np.asarray(points_m, dtype=np.float64) @ transform.R + transform.t_m


def mutant_d2_inverse_wrong_t(transform: RigidTransform):
    """MUTANT: fails to rotate inverse translation into inverse target frame."""
    return RigidTransform(transform.target_frame_id, transform.source_frame_id,
                          transform.R.T, -transform.t_m,
                          calibration_verified=transform.calibration_verified)


def mutant_d3_compose_wrong_t(outer: RigidTransform, inner: RigidTransform):
    """MUTANT: retains correct rotation but adds differently framed translations."""
    return RigidTransform(inner.source_frame_id, outer.target_frame_id,
                          outer.R @ inner.R, inner.t_m + outer.t_m,
                          calibration_verified=(outer.calibration_verified
                                                and inner.calibration_verified))


def mutant_d4_ignore_frame(transform: RigidTransform, point_m, *, frame_id):
    """MUTANT: ignores claimed frame by replacing it with expected source frame."""
    del frame_id
    return transform.transform_point(point_m, frame_id=transform.source_frame_id)


def mutant_d5_publish_on_bool_claim(transform: RigidTransform, point_m):
    """MUTANT: falsely treats mutable-in-construction bool claim as proof."""
    report = inspect_numeric_position(point_m, frame_id=transform.source_frame_id,
                                      transform=transform)
    report["publishable"] = transform.calibration_verified
    report["position_m"] = (report["diagnostics"]["p_target_demo_m"]
                            if report["publishable"] else None)
    return report


def _must_reject_wrong_frame(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError("frame mismatch silently accepted")


def _must_block_release(report):
    assert report["publishable"] is False, "unsafe position publication"
    assert report["position_m"] is None, "unsafe non-null position"


def expect_red_failure(label, oracle_assertion):
    """RED means the intentionally wrong implementation FAILS its expected oracle.

    Catch each expected AssertionError so all five failure modes can be tested
    in a single run. A mutant unexpectedly passing is itself an experiment error.
    """
    try:
        oracle_assertion()
    except AssertionError:
        print(f"RED {label}: expected oracle failure observed")
        return True
    raise AssertionError(f"{label}: mutant was not detected by the oracle")


def verify_red(p_camera, bc, wb):
    """Confirm exactly five faults are detectable without changing production code."""
    p_base = bc.transform_point(p_camera, frame_id=CAMERA_FRAME)
    bc_claim = RigidTransform(CAMERA_FRAME, BASE_FRAME, bc.R, bc.t_m,
                              calibration_verified=True)
    checks = {
        "D1": lambda: np.testing.assert_allclose(
            mutant_d1_batch_wrong_transpose(bc, p_camera.reshape(1, 3))[0],
            ORACLE_BASE_M, rtol=0, atol=TOL),
        "D2": lambda: np.testing.assert_allclose(
            mutant_d2_inverse_wrong_t(bc).transform_point(p_base, frame_id=BASE_FRAME),
            p_camera, rtol=0, atol=TOL),
        "D3": lambda: np.testing.assert_allclose(
            mutant_d3_compose_wrong_t(wb, bc).transform_point(p_camera, frame_id=CAMERA_FRAME),
            ORACLE_WORLD_M, rtol=0, atol=TOL),
        "D4": lambda: _must_reject_wrong_frame(lambda: mutant_d4_ignore_frame(
            bc, p_camera, frame_id="world")),
        "D5": lambda: _must_block_release(mutant_d5_publish_on_bool_claim(bc_claim, p_camera)),
    }
    return {name: expect_red_failure(name, operation) for name, operation in checks.items()}


def verify_green(p_camera, bc, wb):
    """GREEN: existing validated production interface passes all five oracles."""
    p_base = bc.transform_point(p_camera, frame_id=CAMERA_FRAME)
    np.testing.assert_allclose(bc.transform_points(p_camera.reshape(1, 3),
                                                    frame_id=CAMERA_FRAME)[0],
                               ORACLE_BASE_M, rtol=0, atol=TOL)
    print("GREEN D1: correct R.T for batches: PASS")

    np.testing.assert_allclose(bc.inverse().transform_point(p_base, frame_id=BASE_FRAME),
                               p_camera, rtol=0, atol=TOL)
    print("GREEN D2: inverse round-trip: PASS")

    world = wb.compose(bc).transform_point(p_camera, frame_id=CAMERA_FRAME)
    np.testing.assert_allclose(world, ORACLE_WORLD_M, rtol=0, atol=TOL)
    print("GREEN D3: translation composition oracle: PASS")

    _must_reject_wrong_frame(lambda: bc.transform_point(p_camera, frame_id="world"))
    print("GREEN D4: frame mismatch rejected: PASS")

    bc_claim = RigidTransform(CAMERA_FRAME, BASE_FRAME, bc.R, bc.t_m,
                              calibration_verified=True)
    report = inspect_numeric_position(p_camera, frame_id=CAMERA_FRAME,
                                      transform=bc_claim)
    _must_block_release(report)
    assert report["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert "GLOBAL_VALIDATION_REQUIRED" in report["reasons"]
    print("GREEN D5: unauthenticated True claim still cannot publish: PASS")
    return True


def compare_with_open3d(p_camera, bc, wb):
    """Cross-check only correct transforms, never the synthetic publication claim."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p_camera.reshape(1, 3))
    cloud.transform(bc.as_matrix())
    np.testing.assert_allclose(np.asarray(cloud.points)[0], ORACLE_BASE_M,
                               rtol=0, atol=TOL)
    cloud.transform(wb.as_matrix())
    np.testing.assert_allclose(np.asarray(cloud.points)[0], ORACLE_WORLD_M,
                               rtol=0, atol=TOL)
    print("Open3D:", o3d.__version__)
    print("Open3D correct two-stage transformation: PASS")
    return True


def run(project_root):
    root = Path(project_root)
    p = load_exp034_published_position(root)
    bc, wb = make_demo_chain()
    assert not bc.calibration_verified and not wb.calibration_verified
    print("EXP034 published-position provenance: PASS | cluster_label=1")
    red = verify_red(p, bc, wb)
    assert len(red) == 5 and all(red.values())
    verify_green(p, bc, wb)

    try:
        import open3d  # noqa: F401
    except ImportError:
        print("Open3D comparison: SKIPPED (not installed)")
        o3d_result = "SKIPPED"
    else:
        compare_with_open3d(p, bc, wb)
        o3d_result = "PASS"

    base = bc.transform_point(p, frame_id=CAMERA_FRAME)
    report = {
        "experiment": "EXP035-D",
        "source_report": "results/exp034/exp034_C_validated_localization.json",
        "source_cluster_label": 1,
        "source_frame_id": CAMERA_FRAME,
        "synthetic_extrinsics": True,
        "calibration_verified": False,
        "publishable": False,
        "position_m": None,
        "red_oracles_failed_as_expected": red,
        "green_oracles_passed": True,
        "d1_bad_base_m": mutant_d1_batch_wrong_transpose(bc, p.reshape(1, 3))[0].tolist(),
        "d1_correct_base_m": base.tolist(),
        "d2_bad_camera_m": mutant_d2_inverse_wrong_t(bc).transform_point(
            base, frame_id=BASE_FRAME).tolist(),
        "d3_bad_t_world_camera_m": mutant_d3_compose_wrong_t(wb, bc).t_m.tolist(),
        "d3_bad_world_m": mutant_d3_compose_wrong_t(wb, bc).transform_point(
            p, frame_id=CAMERA_FRAME).tolist(),
        "d3_correct_world_m": wb.compose(bc).transform_point(
            p, frame_id=CAMERA_FRAME).tolist(),
        "d4_wrong_frame_would_silently_produce_m": mutant_d4_ignore_frame(
            bc, p, frame_id="world").tolist(),
        "d5_claim_true_safe_status": inspect_numeric_position(
            p, frame_id=CAMERA_FRAME,
            transform=RigidTransform(CAMERA_FRAME, BASE_FRAME, bc.R, bc.t_m,
                                     calibration_verified=True))["status"],
        "open3d_comparison": o3d_result,
        "warning": "ALL TRANSFORMS SYNTHETIC. Diagnostics only; never robot-control poses.",
    }
    np.testing.assert_allclose(report["d1_bad_base_m"], ORACLE_D1_BAD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(report["d2_bad_camera_m"], ORACLE_D2_BAD_M, rtol=0, atol=TOL)
    np.testing.assert_allclose(report["d3_bad_t_world_camera_m"], ORACLE_D3_BAD_T_M,
                               rtol=0, atol=TOL)
    np.testing.assert_allclose(report["d3_bad_world_m"], ORACLE_D3_BAD_WORLD_M,
                               rtol=0, atol=TOL)
    out = root / "results/exp035/exp035_D_fault_injection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Saved:", out)
    print("EXP035-D RED/PATCH/GREEN VERIFIED; EXP035 UNIT COMPLETION: PENDING")
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
