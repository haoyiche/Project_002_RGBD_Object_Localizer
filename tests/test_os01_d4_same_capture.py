"""OS-01-D4 same-capture identity, content and safety contract tests."""
from dataclasses import replace

import numpy as np

from src.geometry.rigid_transform import RigidTransform
from src.pipeline import os01_d4_same_capture as d4


NOW = 2_000_000_000


def frame(uid="frame-A"):
    depth, rgb, mask = d4.d2.create_scene()
    return d4.Capture(
        frame_uid=uid,
        frame_id="camera_optical_frame",
        capture_timestamp_ns=1_000_000_000,
        depth_m=depth,
        rgb=rgb,
        target_mask=mask,
    )


def issue(capture):
    return d4.validate_capture(capture, now_ns=NOW, ttl_ns=1_000_000_000)


def test_d4_same_capture_succeeds_diagnostic_only():
    capture = frame()
    ticket = issue(capture)
    report = d4.analyze_validated_capture(ticket, candidate=capture, now_ns=NOW + 1)
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["diagnostics"]["target_point_count"] == 1500
    np.testing.assert_allclose(report["diagnostics"]["centroid_camera_m"], [.004, .004, 1.2], atol=1e-10)
    np.testing.assert_allclose(report["diagnostics"]["centroid_world_demo_m"], [.096, -.196, 1.5], atol=1e-10)
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_rejects_different_frame_uid_before_projection(monkeypatch):
    a = frame()
    ticket = issue(a)
    b = frame("frame-B")
    def forbidden(*args, **kwargs):
        raise AssertionError("Must not project frame B")
    monkeypatch.setattr(d4, "mask_to_pointcloud", forbidden)
    report = d4.analyze_validated_capture(ticket, candidate=b, now_ns=NOW + 1)
    assert report["status"] == "FRAME_UID_MISMATCH"
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_rejects_modified_depth_same_uid():
    a = frame()
    ticket = issue(a)
    changed = a.depth_m.copy()
    changed[20, 30] += 0.5
    tampered = replace(a, depth_m=changed)
    report = d4.analyze_validated_capture(ticket, candidate=tampered, now_ns=NOW + 1)
    assert report["status"] == "INPUT_DATA_HASH_MISMATCH"
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_rejects_expired_ticket():
    a = frame()
    ticket = issue(a)
    report = d4.analyze_validated_capture(ticket, candidate=a,
        now_ns=ticket.expires_at_ns + 1)
    assert report["status"] == "STALE_VALIDATION_TICKET"
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_upstream_mismatch_blocks_projection(monkeypatch):
    a = frame()
    altered = a.depth_m.copy()
    altered[20, 30] = 2.0   # target pixel becomes background plane
    altered[0, 0] = 1.2     # background pixel becomes target plane
    broken = replace(a, depth_m=altered)
    ticket = issue(broken)
    assert ticket.d2_report["status"] == "SEGMENTATION_MISMATCH"
    def forbidden(*args, **kwargs):
        raise AssertionError("No projection after D2 rejection")
    monkeypatch.setattr(d4, "mask_to_pointcloud", forbidden)
    report = d4.analyze_validated_capture(ticket, candidate=broken, now_ns=NOW + 1)
    assert report["status"] == "UPSTREAM_REJECTED"
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_inverse_transform_is_rejected():
    a = frame()
    ticket = issue(a)
    report = d4.analyze_validated_capture(ticket, candidate=a, now_ns=NOW + 1,
        transform=d4.d3.demo_transform().inverse())
    assert report["status"] == "FRAME_MISMATCH"
    assert report["publishable"] is False and report["position_m"] is None


def test_d4_calibration_claim_never_publishes():
    a = frame()
    ticket = issue(a)
    base = d4.d3.demo_transform()
    claimed = RigidTransform(base.source_frame_id, base.target_frame_id,
                             base.R, base.t_m, calibration_verified=True)
    report = d4.analyze_validated_capture(ticket, candidate=a, now_ns=NOW + 1,
                                          transform=claimed)
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["diagnostics"]["calibration_verified_claim"] is True
    assert report["publishable"] is False and report["position_m"] is None
