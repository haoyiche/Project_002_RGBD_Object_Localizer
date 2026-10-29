"""OS-01-D4: bind D2 validation and D3-style coordinate diagnostics to ONE capture.

Educational SAME-PROCESS consistency demonstration, NOT an authenticated
validation ticket or a production robot-publication gateway.

A SHA-256 digest binds capture identity, numeric inputs and demo intrinsics.
SHA-256 does not authenticate the issuer: a malicious party can forge both
capture and a plain Python ticket. No physical calibration is established.
"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.rigid_transform import RigidTransform
from src.pipeline import os01_d2_plane_background as d2
from src.pipeline import os01_d3_coordinate_chain as d3


@dataclass(frozen=True)
class Capture:
    frame_uid: str
    frame_id: str
    capture_timestamp_ns: int
    depth_m: np.ndarray
    rgb: np.ndarray
    target_mask: np.ndarray


@dataclass(frozen=True)
class ValidatedCapture:
    # Arrays below are defensive snapshots, not cryptographic proof of trust.
    snapshot: Capture
    input_sha256: str
    d2_report: dict
    validated_at_ns: int
    expires_at_ns: int


def _snapshot(capture):
    if not isinstance(capture, Capture):
        raise ValueError("capture must be Capture")
    if not isinstance(capture.frame_uid, str) or not capture.frame_uid:
        raise ValueError("frame_uid must be nonempty string")
    if not isinstance(capture.frame_id, str) or not capture.frame_id:
        raise ValueError("frame_id must be nonempty string")
    if type(capture.capture_timestamp_ns) is not int or capture.capture_timestamp_ns < 0:
        raise ValueError("capture_timestamp_ns must be a nonnegative int")
    # Copy once; both the validator and the consumer use these frozen-in-time bytes.
    return Capture(
        frame_uid=capture.frame_uid,
        frame_id=capture.frame_id,
        capture_timestamp_ns=capture.capture_timestamp_ns,
        depth_m=np.array(capture.depth_m, copy=True),
        rgb=np.array(capture.rgb, copy=True),
        target_mask=np.array(capture.target_mask, copy=True),
    )


def _digest(snapshot):
    hasher = hashlib.sha256()
    meta = {
        "frame_uid": snapshot.frame_uid,
        "frame_id": snapshot.frame_id,
        "capture_timestamp_ns": snapshot.capture_timestamp_ns,
        "intrinsics": d2.INTRINSICS,
    }
    hasher.update(json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for name in ("depth_m", "rgb", "target_mask"):
        array = np.asarray(getattr(snapshot, name))
        hasher.update(name.encode("ascii"))
        hasher.update(array.dtype.str.encode("ascii"))
        hasher.update(repr(array.shape).encode("ascii"))
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def validate_capture(capture, *, now_ns, ttl_ns=5_000_000_000):
    """Run D2 against an immutable-in-practice snapshot; issue local evidence.

    now_ns is caller-supplied for reproducible tests. In production a trusted
    clock and trusted issuance/authentication system would be required.
    """
    if type(now_ns) is not int or type(ttl_ns) is not int or ttl_ns <= 0:
        raise ValueError("now_ns and positive ttl_ns must be int nanoseconds")
    shot = _snapshot(capture)
    if shot.frame_id != d3.CAMERA_FRAME:
        raise ValueError("capture frame must be camera_optical_frame")
    upstream = d2.analyze_scene(
        scene=(shot.depth_m, shot.rgb, shot.target_mask)
    )
    return ValidatedCapture(
        snapshot=shot,
        input_sha256=_digest(shot),
        d2_report=upstream,
        validated_at_ns=now_ns,
        expires_at_ns=now_ns + ttl_ns,
    )


def analyze_validated_capture(ticket, *, candidate=None, now_ns, transform=None):
    """D3-style diagnostic consuming only data bound to the D2 result.

    Optional candidate demonstrates that B-frame substitution gets rejected.
    No raw candidate is ever projected: the ticket's own D2-validated snapshot
    is used after checks pass. All paths keep publication disabled.
    """
    if not isinstance(ticket, ValidatedCapture):
        raise ValueError("ticket must be ValidatedCapture")
    if type(now_ns) is not int:
        raise ValueError("now_ns must be int nanoseconds")
    if transform is None:
        transform = d3.demo_transform()
    if not isinstance(transform, RigidTransform):
        raise ValueError("transform must be RigidTransform")

    result = {
        "status": None,
        "reasons": [],
        "frame_uid": ticket.snapshot.frame_uid,
        "source_frame_id": d3.CAMERA_FRAME,
        "target_frame_id": d3.WORLD_DEMO_FRAME,
        "publishable": False,
        "position_m": None,
        "diagnostics": {},
    }

    def block(status, reason):
        result["status"] = status
        result["reasons"] = [reason]
        return result

    # Validate source before *any* projection or coordinate transform.
    upstream = ticket.d2_report
    if not (
        upstream.get("status") == "DIAGNOSTIC_ONLY"
        and upstream.get("target_membership_matches") is True
        and upstream.get("target_xyz_matches") is True
        and upstream.get("target_rgb_matches") is True
    ):
        return block("UPSTREAM_REJECTED", "D2_SEGMENTATION_NOT_VERIFIED")

    if now_ns < ticket.validated_at_ns or now_ns > ticket.expires_at_ns:
        return block("STALE_VALIDATION_TICKET", "VALIDATION_WINDOW_EXPIRED")

    # Detect modifications to the stored snapshot itself.
    if _digest(ticket.snapshot) != ticket.input_sha256:
        return block("INPUT_DATA_HASH_MISMATCH", "VALIDATED_SNAPSHOT_MODIFIED")

    if candidate is not None:
        candidate_shot = _snapshot(candidate)
        if candidate_shot.frame_uid != ticket.snapshot.frame_uid:
            return block("FRAME_UID_MISMATCH", "VALIDATED_FRAME_UID_REQUIRED")
        if (candidate_shot.frame_id != ticket.snapshot.frame_id
                or candidate_shot.capture_timestamp_ns != ticket.snapshot.capture_timestamp_ns):
            return block("CAPTURE_METADATA_MISMATCH", "FRAME_METADATA_CHANGED")
        if _digest(candidate_shot) != ticket.input_sha256:
            return block("INPUT_DATA_HASH_MISMATCH", "CAPTURE_CONTENT_CHANGED")

    if (transform.source_frame_id != d3.CAMERA_FRAME
            or transform.target_frame_id != d3.WORLD_DEMO_FRAME):
        return block("FRAME_MISMATCH", "TRANSFORM_DIRECTION_OR_FRAME_MISMATCH")

    shot = ticket.snapshot
    points_camera, _colors, _valid = mask_to_pointcloud(
        shot.depth_m, shot.rgb, shot.target_mask, **d2.INTRINSICS
    )
    if len(points_camera) == 0:
        return block("NO_VALID_TARGET_POINTS", "EMPTY_TARGET_POINTCLOUD")

    p_c = points_camera.mean(axis=0)
    p_w = transform.transform_point(p_c, frame_id=d3.CAMERA_FRAME)
    roundtrip = transform.inverse().transform_point(p_w, frame_id=d3.WORLD_DEMO_FRAME)
    result["status"] = "DIAGNOSTIC_ONLY"
    result["reasons"] = ["SYNTHETIC_DEMO_NOT_PHYSICAL_VALIDATION"]
    if not transform.calibration_verified:
        result["reasons"].append("UNVERIFIED_EXTRINSIC")
    result["diagnostics"] = {
        "input_sha256": ticket.input_sha256,
        "d2_status": upstream["status"],
        "target_point_count": int(len(points_camera)),
        "centroid_camera_m": p_c.tolist(),
        "centroid_world_demo_m": p_w.tolist(),
        "roundtrip_residual_mm": float(np.linalg.norm(roundtrip - p_c) * 1000),
        "calibration_verified_claim": transform.calibration_verified,
        "reference_kind": "deterministic_synthetic_demo",
    }
    return result


def main():
    depth_m, rgb, mask = d2.create_scene()
    capture = Capture(
        frame_uid="os01-d4-demo-frame-0001",
        frame_id=d3.CAMERA_FRAME,
        capture_timestamp_ns=1_000_000_000,
        depth_m=depth_m,
        rgb=rgb,
        target_mask=mask,
    )
    ticket = validate_capture(capture, now_ns=2_000_000_000)
    report = analyze_validated_capture(ticket, candidate=capture, now_ns=2_100_000_000)
    for name, value in report.items():
        print(f"{name}: {value}")
    if report["status"] != "DIAGNOSTIC_ONLY":
        raise SystemExit("D4 same-capture diagnostic FAILED")
    assert report["publishable"] is False and report["position_m"] is None
    print("D4 same-capture diagnostic: PASS")


if __name__ == "__main__":
    main()
