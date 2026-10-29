"""OS-01-D3: synthetic coordinate-chain acceptance and fail-closed tests.

No claim of calibrated camera/world alignment or robot publication.
Run: python -m pytest tests/test_os01_d3_coordinate_chain.py -v
"""

import numpy as np

from src.geometry.rigid_transform import RigidTransform
from src.pipeline import os01_d3_coordinate_chain as d3


def test_d3_synthetic_forward_transform_and_roundtrip():
    report = d3.analyze_coordinate_chain()

    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["source_frame_id"] == "camera_optical_frame"
    assert report["target_frame_id"] == "world_demo"
    assert report["publishable"] is False
    assert report["position_m"] is None
    assert "UNVERIFIED_EXTRINSIC" in report["reasons"]

    diagnostics = report["diagnostics"]
    assert diagnostics["target_point_count"] == 1500
    assert diagnostics["reference_kind"] == "deterministic_synthetic_demo"
    np.testing.assert_allclose(
        diagnostics["centroid_camera_m"],
        [0.004, 0.004, 1.2],
        rtol=0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        diagnostics["centroid_world_demo_m"],
        [0.096, -0.196, 1.5],
        rtol=0,
        atol=1e-10,
    )
    assert diagnostics["roundtrip_residual_mm"] < 1e-8


def test_d3_inverse_transform_direction_fails_closed():
    # T_camera_from_world is mathematically valid, but has wrong frame direction.
    inverse = d3.demo_transform().inverse()
    report = d3.analyze_coordinate_chain(transform=inverse)

    assert report["status"] == "FRAME_MISMATCH"
    assert report["reasons"] == ["TRANSFORM_DIRECTION_OR_FRAME_MISMATCH"]
    assert report["diagnostics"]["received_transform_frames"] == {
        "source": "world_demo",
        "target": "camera_optical_frame",
    }
    assert "centroid_world_demo_m" not in report["diagnostics"]
    assert report["publishable"] is False
    assert report["position_m"] is None


def test_d3_upstream_rejection_blocks_pointcloud_conversion(monkeypatch):
    # Guard must short-circuit before reprojecting any target points.
    monkeypatch.setattr(
        d3.d2, "analyze_scene", lambda: {
            "status": "SEGMENTATION_MISMATCH",
            "target_membership_matches": False,
            "target_xyz_matches": False,
            "target_rgb_matches": False,
        }
    )

    def forbidden_conversion(*args, **kwargs):
        raise AssertionError("Back-projection must not run after D2 rejects")

    monkeypatch.setattr(d3, "mask_to_pointcloud", forbidden_conversion)
    report = d3.analyze_coordinate_chain()

    assert report["status"] == "UPSTREAM_REJECTED"
    assert report["reasons"] == ["D2_SEGMENTATION_NOT_VERIFIED"]
    assert report["diagnostics"] == {"d2_status": "SEGMENTATION_MISMATCH"}
    assert report["publishable"] is False
    assert report["position_m"] is None


def test_d3_claimed_calibration_does_not_authorize_publication():
    base = d3.demo_transform()
    # The bool is merely a user-provided claim, never authenticated evidence.
    claimed = RigidTransform(
        source_frame_id=base.source_frame_id,
        target_frame_id=base.target_frame_id,
        R=base.R,
        t_m=base.t_m,
        calibration_verified=True,
    )
    report = d3.analyze_coordinate_chain(transform=claimed)

    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["diagnostics"]["calibration_verified_claim"] is True
    assert "SYNTHETIC_DEMO_NOT_PHYSICAL_VALIDATION" in report["reasons"]
    assert report["publishable"] is False
    assert report["position_m"] is None
