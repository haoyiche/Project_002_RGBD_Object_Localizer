"""EXP035-E: fail-closed EXP034 -> SE(3) *diagnostic-only* integration.

Input contract: EXP034 report plus nonempty, connected list of RigidTransform.
Transforms are synthetic teaching extrinsics. No trusted external calibration verifier
exists in EXP035, so this module NEVER publishes transformed robot positions.

The precomputed EXP034 `published_positions` field is treated as an optional
cache. If supplied, it must agree with an independent audit of `objects`.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

from src.geometry.localization_report_audit import audit_and_extract
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position


CACHE_ATOL_M = 1e-12


def _position_array(value, label):
    """Parse exact XYZ meters, refusing bool, complex, strings, NaN and Inf."""
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{label} must be a length-3 XYZ list")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
           for v in value):
        raise ValueError(f"{label} must contain real non-bool coordinates")
    try:
        xyz = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} cannot be converted to float64") from exc
    if not np.isfinite(xyz).all():
        raise ValueError(f"{label} coordinates must be finite")
    return xyz


def _validate_published_cache(report, canonical):
    """Missing cache is permitted; present cache is verified, never trusted."""
    if "published_positions" not in report:
        return "ABSENT_RECOMPUTED"
    cached = report["published_positions"]
    if not isinstance(cached, list) or len(cached) != len(canonical):
        raise ValueError("published_positions mismatch: count differs from audited objects")
    expected_by_label = {item["cluster_label"]: item for item in canonical}
    seen = set()
    for item in cached:
        if not isinstance(item, dict):
            raise ValueError("published_positions entries must be dicts")
        label = item.get("cluster_label")
        if type(label) is not int or label < 0 or label in seen or label not in expected_by_label:
            raise ValueError("published_positions mismatch: invalid/duplicate/unknown cluster_label")
        seen.add(label)
        reference = expected_by_label[label]
        if item.get("frame_id") != reference["frame_id"]:
            raise ValueError("published_positions mismatch: frame_id differs")
        actual_xyz = _position_array(item.get("position_m"), "published_positions.position_m")
        expected_xyz = _position_array(reference["position_m"], "audited.position_m")
        if not np.allclose(actual_xyz, expected_xyz, rtol=0, atol=CACHE_ATOL_M):
            raise ValueError("published_positions mismatch: position differs")
    return "VERIFIED_AGAINST_OBJECTS"


def _validate_chain(transforms, source_frame_id):
    if not isinstance(transforms, (tuple, list)) or not transforms:
        raise ValueError("transforms must be a nonempty list/tuple of RigidTransform")
    current = source_frame_id
    for index, transform in enumerate(transforms):
        if not isinstance(transform, RigidTransform):
            raise ValueError(f"transforms[{index}] must be RigidTransform")
        if current != transform.source_frame_id:
            raise ValueError(
                f"frame mismatch at transform {index}: expected source {current!r}, "
                f"got {transform.source_frame_id!r}"
            )
        current = transform.target_frame_id
    return current


def run_safe_pipeline(report, transforms):
    """Audit EXP034 report, compute *demo-only* positions, publish nothing.

    Raises ValueError for invalid report, tampered cache or disconnected frames.
    ``calibration_verified`` is a self-declared boolean and cannot authorize
    publication. A future trusted calibration verifier must be independently
    integrated and tested; none is simulated here.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be a dict")
    report_frame = report.get("frame_id")
    if not isinstance(report_frame, str) or not report_frame.strip():
        raise ValueError("report.frame_id must be a nonempty string")
    # EXP034's audited `objects` are the source of truth, not the cached list.
    # The existing audit validates per-object status, count and coordinates.
    canonical = audit_and_extract(report)
    for obj in report["objects"]:
        if obj.get("frame_id") != report_frame:
            raise ValueError("frame mismatch between report and object")
    cache_status = _validate_published_cache(report, canonical)
    target_frame = _validate_chain(transforms, report_frame)

    chain = [
        {
            "source_frame_id": tr.source_frame_id,
            "target_frame_id": tr.target_frame_id,
            "calibration_verified_claim": tr.calibration_verified,
            "trusted_calibration_record_available": False,
        }
        for tr in transforms
    ]
    results = []
    for item in canonical:
        source_position = _position_array(item["position_m"], "audited.position_m")
        current_position = source_position
        current_frame = item["frame_id"]
        diagnostics_steps = []
        for tr in transforms:
            diagnostic = inspect_numeric_position(
                current_position, frame_id=current_frame, transform=tr
            )
            next_position = _position_array(
                diagnostic["diagnostics"]["p_target_demo_m"], "p_target_demo_m"
            )
            diagnostics_steps.append({
                "source_frame_id": current_frame,
                "target_frame_id": tr.target_frame_id,
                "p_target_demo_m": next_position.tolist(),
                "calibration_verified_claim": tr.calibration_verified,
                "numeric_status": diagnostic["status"],
            })
            current_position = next_position
            current_frame = tr.target_frame_id

        all_claimed = all(tr.calibration_verified for tr in transforms)
        reasons = (
            ["GLOBAL_VALIDATION_REQUIRED"] if all_claimed
            else ["UNVERIFIED_EXTRINSIC"]
        )
        if not all_claimed:
            reasons.extend(
                f"UNVERIFIED_TRANSFORM: {tr.source_frame_id} -> {tr.target_frame_id}"
                for tr in transforms if not tr.calibration_verified
            )
        results.append({
            "cluster_label": item["cluster_label"],
            "source_frame_id": item["frame_id"],
            "target_frame_id": current_frame,
            "status": "READY_FOR_GLOBAL_VALIDATION" if all_claimed else "BLOCKED_UNCALIBRATED",
            "publishable": False,
            "position_m": None,
            "reasons": reasons,
            "diagnostics": {
                "p_source_m": source_position.tolist(),
                "p_target_demo_m": current_position.tolist(),
                "steps": diagnostics_steps,
                "demo_only": True,
            },
        })

    if not report["objects"]:
        status = "NO_OBJECTS"
    elif not canonical:
        status = "ALL_INVALID"
    elif all(tr.calibration_verified for tr in transforms):
        status = "READY_FOR_GLOBAL_VALIDATION"
    else:
        status = "BLOCKED_UNCALIBRATED"

    return {
        "experiment": "EXP035-E",
        "source_frame_id": report_frame,
        "target_frame_id": target_frame,
        "unit": "m",
        "source_report_status": report["status"],
        "source_valid_count": report["valid_count"],
        "source_invalid_count": report["invalid_count"],
        "cache_audit": cache_status,
        "status": status,
        "publishable": False,
        "published_positions": [],
        "candidate_count": len(canonical),
        "rejected_source_cluster_labels": [
            item["cluster_label"] for item in report["objects"]
            if item["status"] == "INVALID_CLUSTER"
        ],
        "transform_chain": chain,
        "objects": results,
        "note": "DIAGNOSTIC ONLY. No authenticated calibration verifier exists. "
                "Never forward p_target_demo_m as a robot position.",
    }
