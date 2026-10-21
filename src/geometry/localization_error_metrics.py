"""EXP036-C: strict per-sample 3D localization error evaluator.

All evaluated positions are in meters and in the same camera frame;
error magnitudes and signed differences are in millimeters.
This module only evaluates synthetic reference data, not physical accuracy.

API:
    evaluate_synthetic_samples(records, expected_frame_id=...) -> dict

Invalid samples are recorded in ``rejected_samples`` and excluded from metrics.
A deliberately injected large but valid error is NOT automatically excluded.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SYNTHETIC_GT = "synthetic_gt"


def _nonempty_string(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _vec3_m(value, name: str) -> np.ndarray:
    """Validate shape (3,), dtype/units contract and finite scalar entries."""
    if value is None:
        raise ValueError(f"{name} is missing")
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be shape (3,)") from exc
    if raw.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    floats = []
    for v in raw:
        if isinstance(v, (bool, np.bool_)) or not isinstance(
            v, (int, float, np.integer, np.floating)
        ):
            raise ValueError(f"{name} contains a non-numeric or boolean value")
        f = float(v)
        if not np.isfinite(f):
            raise ValueError(f"{name} must contain finite coordinates")
        floats.append(f)
    result = np.array(floats, dtype=np.float64)
    return result


def _rejection_reason(item: Mapping, expected_frame_id: str) -> str | None:
    """Return an explicit diagnostic code instead of making invalid errors zero."""
    if item.get("reference_kind") != SYNTHETIC_GT:
        return "REFERENCE_KIND_MISMATCH"
    if item.get("synthetic_gt_m") is None:
        return "MISSING_GROUND_TRUTH"
    if item.get("estimate_m") is None:
        return "MISSING_ESTIMATE"
    if item.get("gt_unit") != "m" or item.get("estimate_unit") != "m":
        return "UNCONFIRMED_UNITS"
    if (
        item.get("frame_id") != expected_frame_id
        or item.get("gt_frame_id") != expected_frame_id
        or item.get("estimate_frame_id") != expected_frame_id
    ):
        return "FRAME_MISMATCH"
    try:
        _vec3_m(item["synthetic_gt_m"], "synthetic_gt_m")
    except ValueError:
        return "INVALID_GROUND_TRUTH"
    try:
        _vec3_m(item["estimate_m"], "estimate_m")
    except ValueError:
        return "INVALID_ESTIMATE"
    if type(item.get("injected_outlier")) is not bool:
        return "INVALID_OUTLIER_METADATA"
    return None


def evaluate_synthetic_samples(records: list[dict], *, expected_frame_id: str) -> dict:
    """Evaluate valid synthetic references without granting publishing rights.

    A sample must explicitly name its reference kind, units, and both frames.
    Records failing these checks remain traceable in ``rejected_samples``.
    Repeated IDs, malformed containers and no valid samples are hard failures.
    Caller retains ownership of input structures; this function never mutates.
    """
    expected_frame_id = _nonempty_string(expected_frame_id, "expected_frame_id")
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a nonempty list")

    valid: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"sample[{i}] must be a dict")
        sample_id = _nonempty_string(item.get("sample_id"), f"sample[{i}].sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)

        reason = _rejection_reason(item, expected_frame_id)
        if reason is not None:
            rejected.append({
                "sample_id": sample_id,
                "status": "EXCLUDED_FROM_METRICS",
                "reason": reason,
                "gt_frame_id": item.get("gt_frame_id"),
                "estimate_frame_id": item.get("estimate_frame_id"),
                "gt_present": item.get("synthetic_gt_m") is not None,
                "estimate_present": item.get("estimate_m") is not None,
            })
            continue

        gt = _vec3_m(item["synthetic_gt_m"], "synthetic_gt_m")
        estimate = _vec3_m(item["estimate_m"], "estimate_m")
        signed_xyz_mm = (estimate - gt) * 1000.0
        error_mm = float(np.linalg.norm(signed_xyz_mm))
        if not np.isfinite(error_mm):
            rejected.append({
                "sample_id": sample_id,
                "status": "EXCLUDED_FROM_METRICS",
                "reason": "NONFINITE_ERROR",
                "gt_present": True,
                "estimate_present": True,
            })
            continue
        valid.append({
            "sample_id": sample_id,
            "status": "VALID_FOR_SYNTHETIC_EVALUATION",
            "frame_id": expected_frame_id,
            "reference_kind": SYNTHETIC_GT,
            "synthetic_gt_m": [float(x) for x in gt],
            "estimate_m": [float(x) for x in estimate],
            "signed_delta_xyz_mm": [float(x) for x in signed_xyz_mm],
            "signed_delta_x_mm": float(signed_xyz_mm[0]),
            "euclidean_error_mm": error_mm,
            "injected_outlier": item["injected_outlier"],
        })

    if not valid:
        raise ValueError("no valid synthetic ground truth samples for statistics")
    errors = np.array([row["euclidean_error_mm"] for row in valid], dtype=np.float64)
    signed_deltas_xyz = np.array([row["signed_delta_xyz_mm"] for row in valid], dtype=np.float64)
    stats = {
        "count": len(valid),
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "p95_mm": float(np.percentile(errors, 95, method="linear")),
        "max_mm": float(np.max(errors)),
        "percentile_method": "linear",
    }
    return {
        "reference_kind": SYNTHETIC_GT,
        "data_kind": "synthetic_deterministic",
        "frame_id": expected_frame_id,
        "input_sample_count": len(records),
        "valid_count": len(valid),
        "rejected_count": len(rejected),
        "valid_samples": valid,
        "rejected_samples": rejected,
        "statistics": stats,
        "mean_signed_delta_xyz_mm": [float(x) for x in signed_deltas_xyz.mean(axis=0)],
        "signed_delta_x_mean_mm": float(signed_deltas_xyz[:, 0].mean()),
        "shapes": {
            "gt_points": [len(valid), 3],
            "estimate_points": [len(valid), 3],
            "signed_delta_xyz_mm": [len(valid), 3],
            "signed_delta_x_mm": [len(valid)],
            "euclidean_errors_mm": [len(valid)],
        },
        "outlier_policy": "retain all valid samples in official statistics",
        "publishable": False,
        "published_positions": [],
    }
