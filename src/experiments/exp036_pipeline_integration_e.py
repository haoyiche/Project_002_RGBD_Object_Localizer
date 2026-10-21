"""EXP036-E: real EXP034/EXP035 audit APIs + EXP036-C strict synthetic metrics.

Run from P02 root: python -m src.experiments.exp036_pipeline_integration_e
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.geometry.rigid_transform import RigidTransform
from src.geometry.localization_evaluation_adapter import (
    evaluate_localization_pipeline, INDEPENDENT, DERIVED,
)

CAMERA = "camera_optical_frame"
BASE = "robot_base"


def make_report():
    return {
        "sample_id": "FRAME_DEMO_001",
        "frame_id": CAMERA,
        "status": "PARTIAL_INVALID",
        "valid_count": 1,
        "invalid_count": 1,
        "objects": [
            {"cluster_label": 2, "frame_id": CAMERA, "status": "VALID",
             "publishable": True, "position_m": [0.102, 0.051, 1.02]},
            {"cluster_label": 7, "frame_id": CAMERA, "status": "INVALID_CLUSTER",
             "publishable": False, "position_m": None,
             "reasons": ["EXTENT_EXCEEDED", "CENTROID_BBOX_GAP"]},
        ],
        "published_positions": [
            {"cluster_label": 2, "frame_id": CAMERA, "position_m": [0.102, 0.051, 1.02]}
        ],
    }


def make_exact_camera_report():
    """E3-E5 oracle uses p_camera=[0.10, 0.05, 1.00] exactly."""
    r = make_report()
    r["objects"][0]["position_m"] = [0.10, 0.05, 1.00]
    r["published_positions"][0]["position_m"] = [0.10, 0.05, 1.00]
    return r


def identity_camera_transform():
    return RigidTransform(CAMERA, CAMERA, np.eye(3), [0, 0, 0],
                          calibration_verified=False)


def camera_to_base(*, nominal_verified_claim=False):
    return RigidTransform(CAMERA, BASE, np.eye(3), [0.3, -0.1, 0.2],
                          calibration_verified=nominal_verified_claim)


def synthetic_reference(*, frame_id=CAMERA, gt=None, gt_source=INDEPENDENT):
    if gt is None:
        gt = [0.1, 0.05, 1.0]
    return {
        "sample_id": "FRAME_DEMO_001:cluster:2",
        "source_sample_id": "FRAME_DEMO_001",
        "reference_kind": "synthetic_gt",
        "gt_source": gt_source,
        "gt_independent": gt_source == INDEPENDENT,
        "frame_id": frame_id,
        "unit": "m",
        "synthetic_gt_m": list(gt),
    }


def run(project_root: str | Path = "."):
    report = make_report()

    # E1/E2: geometric-validity is not zero-error; invalid cluster not counted.
    e1e2 = evaluate_localization_pipeline(report, [identity_camera_transform()],
                                           {2: synthetic_reference()})
    np.testing.assert_allclose(e1e2["comparison_rows"][0]["error_mm"],
                               np.sqrt(405.0), atol=1e-10, rtol=0)
    assert e1e2["evaluated_count"] == 1 and e1e2["excluded_count"] == 1
    assert e1e2["comparison_rows"][1]["reason"] == "INVALID_CLUSTER"
    print("E1/E2 geometry-audited synthetic error + invalid cluster excluded: PASS")

    # E3: unverified extrinsics allow EXP035-only demo diagnostics; no comparison
    # statistic in the adapter. This is stricter than manual mathematical demos.
    exact_report = make_exact_camera_report()
    e3 = evaluate_localization_pipeline(exact_report, [camera_to_base()],
         {2: synthetic_reference(frame_id=BASE, gt=[0.4, -0.05, 1.2])})
    assert e3["synthetic_statistics"] is None
    assert e3["comparison_rows"][0]["reason"] == "UNVERIFIED_TRANSFORM_FOR_CROSS_FRAME_GT"
    np.testing.assert_allclose(e3["comparison_rows"][0]["demo_only_position_m"],
                               [0.4, -0.05, 1.2], atol=1e-12, rtol=0)
    print("E3 unverified extrinsic -> demo only, official evaluation blocked: PASS")

    # E4: even when bool verified=True, EXP035 explicitly does not authenticate
    # real-world provenance or grant publishing; synthetic-only comparison.
    e4 = evaluate_localization_pipeline(exact_report, [camera_to_base(nominal_verified_claim=True)],
         {2: synthetic_reference(frame_id=BASE, gt=[0.4, -0.05, 1.2])})
    np.testing.assert_allclose(e4["synthetic_statistics"]["mean_mm"], 0.0,
                               atol=1e-10, rtol=0)
    assert e4["publishable"] is False and e4["calibration_authenticated"] is False
    print("E4 nominal verified CLAIM permits synthetic comparison, never publication: PASS")

    # E5: cyclic GT derived using same unverified transform is self-referential.
    same_pipeline_gt = [0.4, -0.05, 1.2]
    e5 = evaluate_localization_pipeline(exact_report, [camera_to_base()], {
        2: synthetic_reference(frame_id=BASE, gt=same_pipeline_gt, gt_source=DERIVED)
    })
    assert e5["synthetic_statistics"] is None
    assert e5["comparison_rows"][0]["reason"] == "SELF_DERIVED_REFERENCE"
    np.testing.assert_allclose(e5["comparison_rows"][0]["demo_numerical_error_mm"],
                               0.0, atol=1e-10, rtol=0)
    print("E5 self-derived 0mm GT is diagnostic only and never official precision: PASS")

    output = {
        "experiment": "EXP036-E",
        "reference_kind": "synthetic_gt",
        "E1_E2": e1e2, "E3": e3, "E4": e4, "E5": e5,
        "publishable": False,
        "published_positions": [],
        "physical_accuracy_verified": False,
    }
    p = Path(project_root) / "results" / "exp036" / "exp036_E_pipeline_integration.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                 encoding="utf-8")
    print("Saved:", p)
    print("EXP036-E SYNTHETIC PIPELINE INTEGRATION VERIFIED; UNIT COMPLETION: PENDING")
    return output


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
