"""EXP035-E: independent EXP034 report audit + diagnostic-only SE(3) chain.

Run from P02 root: python -m src.experiments.exp035_safe_pipeline_e
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.experiments.exp035_se3_contract_c import make_demo_chain
from src.geometry.localization_report_audit import audit_and_extract
from src.geometry.rigid_transform import RigidTransform
from src.geometry.validated_position_pipeline import run_safe_pipeline

CAMERA_ORACLE = [0.51, 0.01, 1.152]
BASE_ORACLE = [0.29, 0.41, 1.352]
WORLD_ORACLE = [-0.61, 0.69, 1.352]
SOURCE_FILE = "results/exp034/exp034_C_validated_localization.json"


def load_exp034_report(root):
    path = Path(root) / SOURCE_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def with_calibration_claim(transform, claim):
    """Clone for synthetic claim-intervention; NOT trusted authentication."""
    return RigidTransform(
        transform.source_frame_id, transform.target_frame_id,
        transform.R, transform.t_m, calibration_verified=claim,
    )


def verify_e1_to_e5(report):
    bc, wb = make_demo_chain()
    source = audit_and_extract(report)
    assert [p["cluster_label"] for p in source] == [1]
    np.testing.assert_allclose(source[0]["position_m"], CAMERA_ORACLE, rtol=0, atol=1e-12)
    print("E1 EXP034 valid B only; invalid A excluded: PASS")

    e2 = run_safe_pipeline(report, [bc])
    assert e2["candidate_count"] == 1 and e2["rejected_source_cluster_labels"] == [0]
    assert e2["cache_audit"] == "VERIFIED_AGAINST_OBJECTS"
    candidate2 = e2["objects"][0]
    assert candidate2["status"] == "BLOCKED_UNCALIBRATED"
    assert candidate2["reasons"] == ["UNVERIFIED_EXTRINSIC",
                                      "UNVERIFIED_TRANSFORM: camera_optical_frame -> robot_base"]
    assert candidate2["position_m"] is None and candidate2["publishable"] is False
    np.testing.assert_allclose(candidate2["diagnostics"]["p_target_demo_m"],
                               BASE_ORACLE, rtol=0, atol=1e-12)
    assert e2["published_positions"] == []
    print("E2 unverified extrinsic: demo numeric PASS, publication BLOCKED")

    bc_claim = with_calibration_claim(bc, True)
    e3 = run_safe_pipeline(report, [bc_claim])
    candidate3 = e3["objects"][0]
    assert (candidate3["status"], candidate3["publishable"], candidate3["position_m"]) == (
        "READY_FOR_GLOBAL_VALIDATION", False, None
    )
    assert candidate3["reasons"] == ["GLOBAL_VALIDATION_REQUIRED"]
    assert not e3["published_positions"]
    print("E3 claimed verification without evidence cannot publish: PASS")

    bad_frame = copy.deepcopy(report)
    bad_frame["frame_id"] = "world"
    bad_frame["objects"][1]["frame_id"] = "world"
    bad_frame["published_positions"][0]["frame_id"] = "world"
    try:
        run_safe_pipeline(bad_frame, [bc])
    except ValueError as exc:
        assert "frame mismatch" in str(exc)
    else:
        raise AssertionError("E4 wrong frame must be rejected")

    tampered_cache = copy.deepcopy(report)
    tampered_cache["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.160],
    })
    # The legacy audit ignores this cache: it only trusts objects.
    assert [v["cluster_label"] for v in audit_and_extract(tampered_cache)] == [1]
    try:
        run_safe_pipeline(tampered_cache, [bc])
    except ValueError as exc:
        assert "published_positions mismatch" in str(exc)
    else:
        raise AssertionError("E4 corrupted cache must be rejected")
    print("E4 wrong frame and injected invalid-cluster cache rejected: PASS")

    e5 = run_safe_pipeline(report, [bc_claim, wb])
    candidate5 = e5["objects"][0]
    assert e5["status"] == "BLOCKED_UNCALIBRATED"
    assert candidate5["reasons"] == [
        "UNVERIFIED_EXTRINSIC", "UNVERIFIED_TRANSFORM: robot_base -> world"
    ]
    assert candidate5["position_m"] is None and e5["published_positions"] == []
    np.testing.assert_allclose(candidate5["diagnostics"]["steps"][0]["p_target_demo_m"],
                               BASE_ORACLE, rtol=0, atol=1e-12)
    np.testing.assert_allclose(candidate5["diagnostics"]["p_target_demo_m"],
                               WORLD_ORACLE, rtol=0, atol=1e-12)
    assert all(not record["trusted_calibration_record_available"]
               for record in e5["transform_chain"])
    print("E5 partial trust claim: world demo PASS, publication BLOCKED")
    return {"E2_unverified_base": e2, "E3_claimed_base": e3,
            "E5_partial_chain": e5}


def compare_open3d(report):
    import open3d as o3d
    bc, wb = make_demo_chain()
    canonical = audit_and_extract(report)
    p = np.asarray(canonical[0]["position_m"], dtype=np.float64)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(p.reshape(1, 3))
    cloud.transform(bc.as_matrix())
    np.testing.assert_allclose(np.asarray(cloud.points)[0], BASE_ORACLE,
                               atol=1e-12, rtol=0)
    cloud.transform(wb.as_matrix())
    np.testing.assert_allclose(np.asarray(cloud.points)[0], WORLD_ORACLE,
                               atol=1e-12, rtol=0)
    print("Open3D:", o3d.__version__)
    print("Open3D E2/E5 numeric-only chain comparison: PASS")


def run(project_root):
    root = Path(project_root)
    report = load_exp034_report(root)
    result = verify_e1_to_e5(report)
    try:
        import open3d  # noqa: F401
    except ImportError:
        print("Open3D comparison: SKIPPED (not available locally)")
        open3d_status = "SKIPPED"
    else:
        compare_open3d(report)
        open3d_status = "PASS"
    output = {
        "experiment": "EXP035-E",
        "source_report": SOURCE_FILE,
        "source_cluster_label": 1,
        "source_invalid_cluster_label": 0,
        "open3d_comparison": open3d_status,
        "scenarios": result,
        "conclusion": "Even an all-True calibration claim is not authenticated evidence. "
                      "No transformed robot position is published.",
    }
    path = root / "results/exp035/exp035_E_safe_pipeline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    print("Saved:", path)
    print("EXP035-E EXPERIMENT VERIFIED; EXP035 UNIT COMPLETION: PENDING")
    return output


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
