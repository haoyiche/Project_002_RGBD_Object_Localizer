"""OS-01-D3: synthetic camera -> world_demo coordinate-chain diagnostic.

Reuses OS-01-D2 segmentation validation, EXP028 back-projection, and
EXP035 RigidTransform. This is NOT calibrated physical localization.
All positions / translations are in meters. No robot-publishable output.
"""

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.rigid_transform import RigidTransform
from src.pipeline import os01_d2_plane_background as d2


CAMERA_FRAME = "camera_optical_frame"
WORLD_DEMO_FRAME = "world_demo"


def demo_transform():
    """Synthetic T_world_demo_from_camera, NOT authenticated calibration."""
    return RigidTransform(
        source_frame_id=CAMERA_FRAME,
        target_frame_id=WORLD_DEMO_FRAME,
        R=np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ], dtype=np.float64),
        t_m=np.array([0.10, -0.20, 0.30], dtype=np.float64),
        calibration_verified=False,
    )


def analyze_coordinate_chain(*, transform=None):
    """Produce only diagnostics; fail closed for upstream or frame mismatch.

    D2 and this stage each regenerate the same deterministic EXP028 synthetic
    scene. For real data these stages MUST consume the same versioned capture.
    """
    if transform is None:
        transform = demo_transform()
    if not isinstance(transform, RigidTransform):
        raise ValueError("transform must be RigidTransform")

    result = {
        "status": None,
        "reasons": [],
        "source_frame_id": CAMERA_FRAME,
        "target_frame_id": WORLD_DEMO_FRAME,
        "publishable": False,
        "position_m": None,
        "diagnostics": {},
    }

    upstream = d2.analyze_scene()
    if not (
        upstream["status"] == "DIAGNOSTIC_ONLY"
        and upstream["target_membership_matches"] is True
        and upstream["target_xyz_matches"] is True
        and upstream["target_rgb_matches"] is True
    ):
        result["status"] = "UPSTREAM_REJECTED"
        result["reasons"] = ["D2_SEGMENTATION_NOT_VERIFIED"]
        result["diagnostics"]["d2_status"] = upstream["status"]
        return result

    if (transform.source_frame_id != CAMERA_FRAME
            or transform.target_frame_id != WORLD_DEMO_FRAME):
        result["status"] = "FRAME_MISMATCH"
        result["reasons"] = ["TRANSFORM_DIRECTION_OR_FRAME_MISMATCH"]
        result["diagnostics"]["received_transform_frames"] = {
            "source": transform.source_frame_id,
            "target": transform.target_frame_id,
        }
        return result

    depth_m, rgb, target_mask = d2.create_scene()
    points_camera, _colors, _valid_pixels = mask_to_pointcloud(
        depth_m, rgb, target_mask, **d2.INTRINSICS
    )
    if not len(points_camera):
        result["status"] = "NO_VALID_TARGET_POINTS"
        result["reasons"] = ["EMPTY_TARGET_POINTCLOUD"]
        return result

    centroid_camera_m = points_camera.mean(axis=0)
    centroid_world_demo_m = transform.transform_point(
        centroid_camera_m, frame_id=CAMERA_FRAME
    )
    reconstructed_camera_m = transform.inverse().transform_point(
        centroid_world_demo_m, frame_id=WORLD_DEMO_FRAME
    )
    roundtrip_residual_mm = float(
        np.linalg.norm(reconstructed_camera_m - centroid_camera_m) * 1000.0
    )

    result["status"] = "DIAGNOSTIC_ONLY"
    result["reasons"] = ["SYNTHETIC_DEMO_NOT_PHYSICAL_VALIDATION"]
    if not transform.calibration_verified:
        result["reasons"].append("UNVERIFIED_EXTRINSIC")
    result["diagnostics"] = {
        "d2_status": upstream["status"],
        "target_point_count": int(len(points_camera)),
        "centroid_camera_m": centroid_camera_m.tolist(),
        "centroid_world_demo_m": centroid_world_demo_m.tolist(),
        "roundtrip_residual_mm": roundtrip_residual_mm,
        "calibration_verified_claim": transform.calibration_verified,
        "reference_kind": "deterministic_synthetic_demo",
    }
    return result


def main():
    report = analyze_coordinate_chain()
    for key, value in report.items():
        print(f"{key}: {value}")
    if report["status"] != "DIAGNOSTIC_ONLY":
        raise SystemExit("D3 coordinate-chain diagnostic FAILED")
    assert report["publishable"] is False
    assert report["position_m"] is None
    print("D3 synthetic coordinate-chain diagnostic: PASS")


if __name__ == "__main__":
    main()
