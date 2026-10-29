"""OS-01-D2: full-scene RANSAC / mask correspondence regression tests.

These are synthetic-scene diagnostic tests, not physical calibration checks.
"""

import pytest

from src.pipeline import os01_d2_plane_background as d2


def test_d2_full_scene_background_and_target_membership():
    report = d2.analyze_scene()

    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["scene_point_count"] == 7900
    assert report["plane_inlier_count"] == 6400
    assert report["remaining_point_count"] == 1500
    assert report["mask_target_point_count"] == 1500
    assert report["invalid_depth_count"] == 100
    assert report["target_membership_matches"] is True
    assert report["target_xyz_matches"] is True
    assert report["target_rgb_matches"] is True
    assert report["reference_plane_error_m"] == pytest.approx(0.0, abs=1e-8)
    assert report["publishable"] is False
    assert report["position_m"] is None


def test_d2_equal_counts_but_swapped_membership_fails_closed(monkeypatch):
    # Deliberately swap the depth-plane identity of one target pixel and one
    # background pixel. Valid pixel count and both group counts remain the
    # same, but one background point contaminates the retained target cloud.
    original_create_scene = d2.create_scene

    def swapped_scene():
        depth_m, rgb, target_mask = original_create_scene()
        depth_m = depth_m.copy()

        assert target_mask[20, 30]
        assert not target_mask[0, 0]
        assert depth_m[20, 30] == 1.2
        assert depth_m[0, 0] == 2.0

        depth_m[20, 30] = 2.0  # One mask target falls on background plane.
        depth_m[0, 0] = 1.2   # One non-target leaves background plane.
        return depth_m, rgb, target_mask

    monkeypatch.setattr(d2, "create_scene", swapped_scene)
    report = d2.analyze_scene()

    # Pure counts still look perfect. This is the point of the fault injection.
    assert report["scene_point_count"] == 7900
    assert report["plane_inlier_count"] == 6400
    assert report["remaining_point_count"] == 1500
    assert report["mask_target_point_count"] == 1500
    assert report["invalid_depth_count"] == 100

    # Row-level correspondence catches the identity swap and blocks output.
    assert report["target_membership_matches"] is False
    assert report["target_xyz_matches"] is False
    assert report["target_rgb_matches"] is False
    assert report["status"] == "SEGMENTATION_MISMATCH"
    assert report["publishable"] is False
    assert report["position_m"] is None
