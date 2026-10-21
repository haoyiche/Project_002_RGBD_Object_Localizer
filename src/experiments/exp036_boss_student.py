import math
import numpy as np


_ALLOWED_GT_SOURCES = {
    "independent_synthetic_design",
    "independent_measurement",
}


def _is_finite_number(v):
    if isinstance(v, bool):
        return False
    if not isinstance(v, (int, float)):
        return False
    return math.isfinite(v)


def _is_finite_3d(v):
    if v is None or not isinstance(v, (list, tuple)) or len(v) != 3:
        return False
    return all(_is_finite_number(x) for x in v)


def _record_reason(excluded, sample_id, reason):
    excluded.append({
        "sample_id": sample_id,
        "reason": reason,
    })


def evaluate_localizations(records):
    """
    最小评估接口。

    契约：
    1. 只计算有效簇、GT 存在、单位/frame 一致且来源合格的样本。
    2. 非法数据保留诊断，不记为 0 mm。
    3. 计算 Mean、Median、P95(linear)、Max。
    4. 记录输入数量、评估数量、排除数量及原因。
    5. 空输入、全无效、重复 ID 有明确行为。
    6. 评估结果不自动赋予 publishable=True。
    """
    if not isinstance(records, list):
        raise ValueError("records must be a list")

    input_count = len(records)
    excluded = []
    evaluated = []

    # ---------- 重复 sample_id 检测 ----------
    id_counts = {}
    for rec in records:
        if isinstance(rec, dict):
            sid = rec.get("sample_id")
            if isinstance(sid, str) and sid.strip() != "":
                id_counts[sid] = id_counts.get(sid, 0) + 1

    duplicate_ids = {sid for sid, n in id_counts.items() if n > 1}

    seen_ids = set()

    for rec in records:
        if not isinstance(rec, dict):
            _record_reason(excluded, None, "INVALID_RECORD")
            continue

        sid = rec.get("sample_id")
        if not isinstance(sid, str) or sid.strip() == "":
            _record_reason(excluded, sid, "MISSING_SAMPLE_ID")
            continue

        if sid in duplicate_ids:
            _record_reason(excluded, sid, "DUPLICATE_SAMPLE_ID")
            continue

        if sid in seen_ids:
            _record_reason(excluded, sid, "DUPLICATE_SAMPLE_ID")
            continue
        seen_ids.add(sid)

        if rec.get("cluster_status") != "VALID":
            _record_reason(excluded, sid, "INVALID_CLUSTER")
            continue

        est = rec.get("estimate_m")
        gt = rec.get("synthetic_gt_m")

        if gt is None:
            _record_reason(excluded, sid, "MISSING_GROUND_TRUTH")
            continue

        if not _is_finite_3d(est):
            _record_reason(excluded, sid, "INVALID_ESTIMATE")
            continue

        if not _is_finite_3d(gt):
            _record_reason(excluded, sid, "INVALID_GROUND_TRUTH")
            continue

        if rec.get("estimate_unit") != "m":
            _record_reason(excluded, sid, "ESTIMATE_UNIT_MISMATCH")
            continue

        if rec.get("gt_unit") != "m":
            _record_reason(excluded, sid, "GT_UNIT_MISMATCH")
            continue

        est_frame = rec.get("estimate_frame_id")
        gt_frame = rec.get("gt_frame_id")

        if not isinstance(est_frame, str) or est_frame.strip() == "":
            _record_reason(excluded, sid, "MISSING_ESTIMATE_FRAME")
            continue

        if not isinstance(gt_frame, str) or gt_frame.strip() == "":
            _record_reason(excluded, sid, "MISSING_GT_FRAME")
            continue

        if est_frame != gt_frame:
            _record_reason(excluded, sid, "FRAME_MISMATCH")
            continue

        gt_source = rec.get("gt_source")
        if not isinstance(gt_source, str) or gt_source not in _ALLOWED_GT_SOURCES:
            _record_reason(excluded, sid, "UNQUALIFIED_GT_SOURCE")
            continue

        est_vec = np.asarray(est, dtype=np.float64)
        gt_vec = np.asarray(gt, dtype=np.float64)

        with np.errstate(over="ignore", invalid="ignore"):
            delta_m = est_vec - gt_vec
            delta_mm = delta_m * 1000.0
            error_mm = float(np.linalg.norm(delta_mm))

        if not np.isfinite(delta_mm).all() or not np.isfinite(error_mm):
            _record_reason(excluded, sid, "NONFINITE_COMPUTED_ERROR")
            continue

        evaluated.append({
            "sample_id": sid,
            "error_mm": error_mm,
            "signed_delta_mm": delta_mm.tolist(),
            "frame_id": est_frame,
            "gt_source": gt_source,
        })

    evaluated_count = len(evaluated)
    excluded_count = len(excluded)

    errors_mm = np.asarray(
        [e["error_mm"] for e in evaluated], dtype=np.float64
    )

    if evaluated_count == 0:
        stats = {
            "count": 0,
            "mean_mm": None,
            "median_mm": None,
            "p95_mm": None,
            "max_mm": None,
            "percentile_method": "linear",
        }
        status = "NO_VALID_SAMPLES"
    else:
        stats = {
            "count": evaluated_count,
            "mean_mm": float(np.mean(errors_mm)),
            "median_mm": float(np.median(errors_mm)),
            "p95_mm": float(np.percentile(errors_mm, 95, method="linear")),
            "max_mm": float(np.max(errors_mm)),
            "percentile_method": "linear",
        }
        status = "EVALUATED"

    return {
        "status": status,
        "input_count": input_count,
        "evaluated_count": evaluated_count,
        "excluded_count": excluded_count,
        "stats": stats,
        "excluded": excluded,
        "evaluated_samples": evaluated,
        "publishable": False,          # 永远不自动赋予
        "published_positions": [],
        "note": (
            "Evaluation PASS does not grant robot publishability. "
            "Independent calibration verification and safety policy are required."
        ),
    }