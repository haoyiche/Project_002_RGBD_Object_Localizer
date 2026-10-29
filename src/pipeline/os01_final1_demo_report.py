"""Level2 Final-1: same-capture RGB-D geometry & coordinate-chain demo.

All positions and extents are meters. This is deterministic synthetic evidence,
NOT a calibrated pose, robot command, or authenticated validation service.
"""

import json
from pathlib import Path

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.pipeline import os01_d2_plane_background as d2
from src.pipeline import os01_d3_coordinate_chain as d3
from src.pipeline import os01_d4_same_capture as d4


REPORT_PATH = Path("outputs/level2_final/demo_report.json")


def _aabb(points):
    """Tight axis-aligned box of the supplied (N,3) points, in THAT frame."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have nonempty shape (N,3)")
    if not np.isfinite(points).all():
        raise ValueError("AABB points must be finite")
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    return {
        "min": lo.tolist(),
        "max": hi.tolist(),
        "extent": (hi - lo).tolist(),
        "center": ((lo + hi) / 2).tolist(),
    }


def make_demo_capture():
    depth_m, rgb, target_mask = d2.create_scene()
    return d4.Capture(
        frame_uid="level2-final-demo-frame-0001",
        frame_id=d3.CAMERA_FRAME,
        capture_timestamp_ns=1_000_000_000,
        depth_m=depth_m,
        rgb=rgb,
        target_mask=target_mask,
    )


def build_demo_report(
    *,
    capture=None,
    candidate=None,
    transform=None,
    validated_at_ns=2_000_000_000,
    now_ns=2_100_000_000,
):
    """Validate D2 and consume only D4's bound snapshot; fail closed on rejection."""
    if capture is None:
        capture = make_demo_capture()
    if candidate is None:
        candidate = capture
    if transform is None:
        transform = d3.demo_transform()

    ticket = d4.validate_capture(capture, now_ns=validated_at_ns)
    diagnostic = d4.analyze_validated_capture(
        ticket, candidate=candidate, now_ns=now_ns, transform=transform
    )

    report = {
        "schema_version": "os01_level2_final1_v1",
        "sample_id": "os01-level2-demo-001",
        "frame_uid": diagnostic["frame_uid"],
        "status": diagnostic["status"],
        "reasons": list(diagnostic["reasons"]),
        "source_frame_id": diagnostic["source_frame_id"],
        "target_frame_id": diagnostic["target_frame_id"],
        "target_point_count": None,
        "centroid_camera_m": None,
        "bbox_min_camera_m": None,
        "bbox_max_camera_m": None,
        "bbox_extent_camera_m": None,
        "bbox_center_camera_m": None,
        "centroid_world_demo_m": None,
        "bbox_min_world_demo_m": None,
        "bbox_max_world_demo_m": None,
        "bbox_extent_world_demo_m": None,
        "bbox_center_world_demo_m": None,
        "input_sha256": ticket.input_sha256,
        "reference_kind": "deterministic_synthetic_demo",
        "physical_accuracy_claimed": False,
        "roundtrip_residual_mm": None,
        "position_m": None,
        "publishable": False,
    }
    if diagnostic["status"] != "DIAGNOSTIC_ONLY":
        return report

    # Only the D2-validated snapshot can supply geometry, not the optional candidate.
    shot = ticket.snapshot
    points_camera, _colors, _valid_pixels = mask_to_pointcloud(
        shot.depth_m, shot.rgb, shot.target_mask, **d2.INTRINSICS
    )
    if (len(points_camera) != diagnostic["diagnostics"]["target_point_count"]
            or len(points_camera) != ticket.d2_report["remaining_point_count"]):
        report["status"] = "INTEGRATION_MISMATCH"
        report["reasons"] = ["TARGET_POINT_COUNT_CHANGED"]
        return report

    points_world = transform.transform_points(points_camera, frame_id=d3.CAMERA_FRAME)
    camera_box = _aabb(points_camera)
    world_box = _aabb(points_world)
    c_camera = points_camera.mean(axis=0)
    c_world = points_world.mean(axis=0)
    existing = diagnostic["diagnostics"]

    # Check the two computation routes without treating agreement as physical accuracy.
    if (not np.allclose(c_camera, existing["centroid_camera_m"], rtol=0, atol=1e-12)
            or not np.allclose(c_world, existing["centroid_world_demo_m"], rtol=0, atol=1e-12)):
        report["status"] = "INTEGRATION_MISMATCH"
        report["reasons"] = ["CENTROID_COMPUTATION_MISMATCH"]
        return report

    report.update({
        "target_point_count": int(len(points_camera)),
        "centroid_camera_m": c_camera.tolist(),
        "bbox_min_camera_m": camera_box["min"],
        "bbox_max_camera_m": camera_box["max"],
        "bbox_extent_camera_m": camera_box["extent"],
        "bbox_center_camera_m": camera_box["center"],
        "centroid_world_demo_m": c_world.tolist(),
        "bbox_min_world_demo_m": world_box["min"],
        "bbox_max_world_demo_m": world_box["max"],
        "bbox_extent_world_demo_m": world_box["extent"],
        "bbox_center_world_demo_m": world_box["center"],
        "roundtrip_residual_mm": existing["roundtrip_residual_mm"],
    })
    return report


def main():
    report = build_demo_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("status:", report["status"])
    print("target_point_count:", report["target_point_count"])
    print("centroid_camera_m:", report["centroid_camera_m"])
    print("bbox_extent_camera_m:", report["bbox_extent_camera_m"])
    print("centroid_world_demo_m:", report["centroid_world_demo_m"])
    print("bbox_extent_world_demo_m:", report["bbox_extent_world_demo_m"])
    print("publishable:", report["publishable"])
    print("position_m:", report["position_m"])
    print("report:", REPORT_PATH)
    if report["status"] != "DIAGNOSTIC_ONLY":
        raise SystemExit("Level 2 Final-1 demo rejected; inspect report")
    assert report["publishable"] is False and report["position_m"] is None
    print("Level 2 Final-1 geometry report: PASS")


if __name__ == "__main__":
    main()
