"""Final-1: verify actual AABB geometry and retain D4's fail-closed contract."""
from dataclasses import replace
import json

import numpy as np

from src.pipeline import os01_d3_coordinate_chain as d3
from src.pipeline import os01_final1_demo_report as final1


def test_final1_same_capture_centroids_aabbs_and_safety():
    report = final1.build_demo_report()
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["frame_uid"] == "level2-final-demo-frame-0001"
    assert report["target_point_count"] == 1500
    np.testing.assert_allclose(report["centroid_camera_m"], [.004, .004, 1.2], atol=1e-12)
    np.testing.assert_allclose(report["bbox_min_camera_m"], [-.234, -.234, 1.2], atol=1e-12)
    np.testing.assert_allclose(report["bbox_max_camera_m"], [.234, .234, 1.2], atol=1e-12)
    np.testing.assert_allclose(report["bbox_extent_camera_m"], [.468, .468, 0], atol=1e-12)
    np.testing.assert_allclose(report["bbox_center_camera_m"], [0, 0, 1.2], atol=1e-12)
    np.testing.assert_allclose(report["centroid_world_demo_m"], [.096, -.196, 1.5], atol=1e-12)
    np.testing.assert_allclose(report["bbox_min_world_demo_m"], [-.134, -.434, 1.5], atol=1e-12)
    np.testing.assert_allclose(report["bbox_max_world_demo_m"], [.334, .034, 1.5], atol=1e-12)
    np.testing.assert_allclose(report["bbox_extent_world_demo_m"], [.468, .468, 0], atol=1e-12)
    np.testing.assert_allclose(report["bbox_center_world_demo_m"], [.1, -.2, 1.5], atol=1e-12)
    assert report["physical_accuracy_claimed"] is False
    assert report["publishable"] is False and report["position_m"] is None
    json.dumps(report, allow_nan=False)  # valid finite JSON


def test_final1_frame_b_cannot_reuse_frame_a_validation():
    a = final1.make_demo_capture()
    b = replace(a, frame_uid="different-frame-B")
    report = final1.build_demo_report(capture=a, candidate=b)
    assert report["status"] == "FRAME_UID_MISMATCH"
    assert report["centroid_world_demo_m"] is None
    assert report["bbox_extent_world_demo_m"] is None
    assert report["publishable"] is False and report["position_m"] is None


def test_final1_rejects_inverse_transform():
    report = final1.build_demo_report(transform=d3.demo_transform().inverse())
    assert report["status"] == "FRAME_MISMATCH"
    assert report["centroid_camera_m"] is None
    assert report["publishable"] is False and report["position_m"] is None


def test_final1_rejects_wrong_membership_even_when_counts_match():
    a = final1.make_demo_capture()
    corrupted_depth = a.depth_m.copy()
    corrupted_depth[20, 30] = 2.0
    corrupted_depth[0, 0] = 1.2
    bad_capture = replace(a, depth_m=corrupted_depth)
    report = final1.build_demo_report(capture=bad_capture)
    assert report["status"] == "UPSTREAM_REJECTED"
    assert report["bbox_extent_camera_m"] is None
    assert report["centroid_world_demo_m"] is None
    assert report["publishable"] is False and report["position_m"] is None


def test_final1_aabb_depends_on_frame_for_non_axis_aligned_rotation():
    # Small independent counterexample: 45-degree rotation does change extent.
    points = np.array([[1., 0., 0.], [0., 1., 0.]])
    aabb_a = final1._aabb(points)
    angle = np.pi / 4
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    aabb_b = final1._aabb(points @ R.T)
    np.testing.assert_allclose(aabb_a["extent"], [1., 1., 0.], atol=1e-12)
    np.testing.assert_allclose(aabb_b["extent"], [np.sqrt(2), 0., 0.], atol=1e-12)
