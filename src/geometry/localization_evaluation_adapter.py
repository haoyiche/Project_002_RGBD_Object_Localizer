"""EXP036-E: diagnostic-only evaluation adapter for real EXP034/EXP035 APIs.

Crucial boundaries:
- Call existing EXP035 ``audit_chain_and_compute``. NEVER replace its gate.
- Calibration flag is only a CLAIM, not signed real-world provenance.
- Derived/self-referential synthetic GT is excluded from evaluation statistics.
- All results are synthetic-only and publication is permanently prohibited.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real

import numpy as np

from src.experiments.exp035_boss_student import audit_chain_and_compute
from src.geometry.rigid_transform import RigidTransform
from src.geometry.localization_error_metrics import evaluate_synthetic_samples


CAMERA = "camera_optical_frame"
INDEPENDENT = "independent_synthetic_design"
DERIVED = "derived_from_estimate_pipeline"


def _label(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty string")
    return value


def _vec3(value):
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError("point must be shape (3,)")
    for v in raw:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Real):
            raise ValueError("point must have numeric non-boolean elements")
    out = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(out).all():
        raise ValueError("point coordinates must be finite")
    return out


def _exclude(sample_id, cluster_label, reason, **extra):
    return {"sample_id": sample_id, "cluster_label": cluster_label,
            "evaluated": False, "error_mm": None, "reason": reason, **extra}


def evaluate_localization_pipeline(report, transforms, references_by_cluster_label):
    """Evaluate synthetic GT against *audited* EXP034/035 diagnostics.

    ``report`` is an EXP034 report with extra ``sample_id``; existing required
    report fields/objects/cache are checked by EXP035, not reimplemented here.
    ``transforms``: nonempty list of actual EXP035 RigidTransform objects.
    ``references_by_cluster_label``: mapping int label -> metadata:
      sample_id: '{report.sample_id}:cluster:{label}'
      source_sample_id: report.sample_id
      reference_kind: 'synthetic_gt'
      gt_source: 'independent_synthetic_design' OR 'derived_from_estimate_pipeline'
      gt_independent: bool, correspondingly True OR False
      frame_id: final transform frame; unit: 'm'
      synthetic_gt_m: XYZ (m)

    Returns numbers ONLY for independent synthetic comparisons that pass
    nominal transform-claim checks. Never makes a physical accuracy claim.
    """
    if not isinstance(report, dict):
        raise ValueError("report must be dict")
    report_id = _label(report.get("sample_id"), "report.sample_id")
    if not isinstance(references_by_cluster_label, Mapping):
        raise ValueError("references_by_cluster_label must be a mapping")

    # Existing contract guards, including invalid cluster contradiction,
    # top-level frame, cached positions, chain linkage and point shape.
    audited = audit_chain_and_compute(report, transforms)
    if audited["publishable"] is not False or audited["published_positions"] != []:
        raise ValueError("EXP035 diagnostic output unexpectedly permits publishing")

    if not isinstance(transforms, (tuple, list)) or not transforms or any(
        not isinstance(t, RigidTransform) for t in transforms
    ):
        raise ValueError("transforms must be nonempty sequence of RigidTransform")
    final_frame = transforms[-1].target_frame_id
    cross_frame = final_frame != report["frame_id"]
    nominal_claims_ok = all(t.calibration_verified is True for t in transforms)

    # Detect invalid external labels as configuration problems, rather than
    # silently omitting entries. References for rejected cluster IDs are allowed
    # but are NEVER used to fabricate a position.
    labels = {item["cluster_label"] for item in report["objects"]}
    for k in references_by_cluster_label:
        if type(k) is not int or k not in labels:
            raise ValueError("reference has unknown or invalid cluster_label")

    diag_by_label = {d["cluster_label"]: d for d in audited["diagnostics"]}
    accepted = []
    rows = []
    for obj in report["objects"]:
        cluster_label = obj["cluster_label"]
        sample_id = f"{report_id}:cluster:{cluster_label}"
        if obj["status"] == "INVALID_CLUSTER":
            rows.append(_exclude(sample_id, cluster_label, "INVALID_CLUSTER",
                                 cluster_reasons=obj.get("reasons", [])))
            continue

        # EXP035 has already audited and transformed each VALID object.
        pos = diag_by_label[cluster_label]["final_position_m"]
        ref = references_by_cluster_label.get(cluster_label)
        if ref is None:
            rows.append(_exclude(sample_id, cluster_label, "MISSING_GROUND_TRUTH"))
            continue
        if not isinstance(ref, Mapping):
            raise ValueError("reference metadata must be mapping")
        if ref.get("synthetic_gt_m") is None:
            rows.append(_exclude(sample_id, cluster_label, "MISSING_GROUND_TRUTH"))
            continue
        if ref.get("sample_id") != sample_id or ref.get("source_sample_id") != report_id:
            rows.append(_exclude(sample_id, cluster_label, "PROVENANCE_MISMATCH"))
            continue
        if ref.get("reference_kind") != "synthetic_gt":
            rows.append(_exclude(sample_id, cluster_label, "UNSUPPORTED_GT_KIND"))
            continue
        if ref.get("unit") != "m":
            rows.append(_exclude(sample_id, cluster_label, "UNCONFIRMED_UNITS"))
            continue
        if ref.get("frame_id") != final_frame:
            rows.append(_exclude(sample_id, cluster_label, "FRAME_MISMATCH",
                                 reference_frame_id=ref.get("frame_id"),
                                 estimate_frame_id=final_frame))
            continue
        if ref.get("gt_source") not in (INDEPENDENT, DERIVED) or type(ref.get("gt_independent")) is not bool:
            rows.append(_exclude(sample_id, cluster_label, "UNKNOWN_GT_PROVENANCE"))
            continue
        if ref["gt_independent"] != (ref["gt_source"] == INDEPENDENT):
            rows.append(_exclude(sample_id, cluster_label, "CONTRADICTORY_GT_PROVENANCE"))
            continue
        try:
            gt = _vec3(ref["synthetic_gt_m"])
        except (ValueError, TypeError, OverflowError):
            rows.append(_exclude(sample_id, cluster_label, "INVALID_GROUND_TRUTH"))
            continue
        delta_mm = (np.asarray(pos, dtype=np.float64) - gt) * 1000.0
        # 'demo_numerical_error_mm' is explicitly not an eligible statistic.
        numerical_error_mm = float(np.linalg.norm(delta_mm))
        if not np.isfinite(numerical_error_mm):
            rows.append(_exclude(sample_id, cluster_label, "NONFINITE_ERROR"))
            continue
        if ref["gt_source"] == DERIVED:
            rows.append(_exclude(sample_id, cluster_label, "SELF_DERIVED_REFERENCE",
                                 gt_source=DERIVED,
                                 demo_numerical_error_mm=numerical_error_mm))
            continue
        if cross_frame and not nominal_claims_ok:
            rows.append(_exclude(sample_id, cluster_label, "UNVERIFIED_TRANSFORM_FOR_CROSS_FRAME_GT",
                                 demo_only_position_m=list(pos)))
            continue

        accepted.append({
            "sample_id": sample_id,
            "reference_kind": "synthetic_gt",
            "frame_id": final_frame,
            "gt_frame_id": final_frame,
            "estimate_frame_id": final_frame,
            "gt_unit": "m",
            "estimate_unit": "m",
            "synthetic_gt_m": gt.tolist(),
            "estimate_m": list(pos),
            "injected_outlier": False,
        })
        rows.append({"sample_id": sample_id, "cluster_label": cluster_label,
                     "evaluated": True, "error_mm": numerical_error_mm,
                     "reason": None, "gt_source": INDEPENDENT})

    metrics = (evaluate_synthetic_samples(accepted, expected_frame_id=final_frame)
               if accepted else None)
    return {
        "experiment": "EXP036-E",
        "reference_kind": "synthetic_gt",
        "data_kind": "synthetic_diagnostic_only",
        "report_sample_id": report_id,
        "source_frame_id": report["frame_id"],
        "final_frame_id": final_frame,
        "transform_calibration_claims_all_true": nominal_claims_ok,
        "calibration_authenticated": False,
        "publishable": False,
        "published_positions": [],
        "comparison_rows": rows,
        "input_cluster_count": len(rows),
        "evaluated_count": len(accepted),
        "excluded_count": len(rows) - len(accepted),
        "synthetic_statistics": metrics["statistics"] if metrics else None,
        "synthetic_evaluation": metrics,
        "transform_diagnostics": audited["diagnostics"],
        "warning": "EXP034/035 audited synthetic code diagnostics ONLY; no authenticated calibration or real accuracy claim",
    }
