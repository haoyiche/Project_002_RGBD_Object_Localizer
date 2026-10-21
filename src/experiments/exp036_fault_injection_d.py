"""EXP036-D | controlled metric fault injection, RED -> PATCH -> GREEN.

All positions are synthetic. Produces diagnostics only, never robot commands.
The production EXP036-C evaluator is used unchanged as the GREEN reference.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.experiments.exp036_error_benchmark_c import (
    FRAME_ID,
    make_deterministic_samples,
)
from src.geometry.localization_error_metrics import evaluate_synthetic_samples


def oracle_stats(errors_mm):
    """Independent NumPy oracle; always uses linear percentiles."""
    a = np.asarray(errors_mm, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("errors_mm must be a nonempty 1D nonnegative finite array")
    return {
        "count": int(a.size),
        "mean_mm": float(np.mean(a)),
        "median_mm": float(np.median(a)),
        "p95_mm": float(np.percentile(a, 95, method="linear")),
        "max_mm": float(np.max(a)),
    }


def _eval(samples):
    return evaluate_synthetic_samples(samples, expected_frame_id=FRAME_ID)


def _as_errors_and_deltas(report):
    errors = np.array([v["euclidean_error_mm"] for v in report["valid_samples"]], dtype=float)
    delta = np.array([v["signed_delta_x_mm"] for v in report["valid_samples"]], dtype=float)
    return errors, delta


def broken_mean_from_signed(signed_delta_x_mm):
    return float(abs(np.mean(signed_delta_x_mm)))


def fixed_mean_from_distances(errors_mm):
    return float(np.mean(errors_mm))


def broken_p95_higher(errors_mm):
    return float(np.percentile(errors_mm, 95, method="higher"))


def fixed_p95_linear(errors_mm):
    return float(np.percentile(errors_mm, 95, method="linear"))


def broken_filter_large_errors(errors_mm):
    return np.asarray(errors_mm)[np.asarray(errors_mm) < 20.0]


def fixed_full_error_set(errors_mm):
    return np.asarray(errors_mm, dtype=np.float64).copy()


def broken_index_alignment(samples):
    """Fault: payload swapped while row sample IDs remain unchanged."""
    altered = copy.deepcopy(samples)
    altered[0]["estimate_m"], altered[1]["estimate_m"] = (
        altered[1]["estimate_m"], altered[0]["estimate_m"]
    )
    return altered


def require_source_provenance(sample_rows, provenance):
    """Independent source-row IDs must agree with sample IDs.

    In real systems source IDs must originate from a trusted acquisition layer,
    not from an unverified caller-supplied string.
    """
    if not isinstance(provenance, list) or len(provenance) != len(sample_rows):
        raise ValueError("provenance length mismatch")
    for row, meta in zip(sample_rows, provenance):
        if not isinstance(row, dict) or not isinstance(meta, dict):
            raise ValueError("invalid provenance record")
        if meta.get("record_sample_id") != row.get("sample_id"):
            raise ValueError("record_sample_id mismatch")
        if meta.get("estimate_source_sample_id") != row.get("sample_id"):
            raise ValueError("estimate_source_sample_id mismatch")


def normal_provenance(samples):
    return [{
        "record_sample_id": s["sample_id"],
        "estimate_source_sample_id": s["sample_id"],
    } for s in samples]


def swapped_provenance(provenance):
    altered = copy.deepcopy(provenance)
    altered[0]["estimate_source_sample_id"], altered[1]["estimate_source_sample_id"] = (
        altered[1]["estimate_source_sample_id"], altered[0]["estimate_source_sample_id"]
    )
    return altered


def with_missing_gt(samples):
    """Append an extra legal record without GT, retaining its diagnostic identity."""
    altered = copy.deepcopy(samples)
    item = copy.deepcopy(samples[0])
    item["sample_id"] = "MISSING_20"
    item["synthetic_gt_m"] = None
    item["injected_outlier"] = False
    altered.append(item)
    return altered


def broken_fill_missing_gt(samples):
    altered = copy.deepcopy(samples)
    for row in altered:
        if row.get("synthetic_gt_m") is None:
            row["synthetic_gt_m"] = copy.deepcopy(row["estimate_m"])
    return altered


def fixed_keep_missing_gt_as_rejected(samples):
    return _eval(samples)


def build_fault_report():
    samples = make_deterministic_samples()
    baseline = _eval(samples)
    errors, signed = _as_errors_and_deltas(baseline)
    baseline_stats = oracle_stats(errors)

    # C's public benchmark is the production GREEN comparator.
    for key in ("count", "mean_mm", "median_mm", "p95_mm", "max_mm"):
        assert np.isclose(baseline["statistics"][key], baseline_stats[key], atol=1e-9, rtol=0)
    assert len(samples) == 20 and baseline["valid_count"] == 20
    assert np.isclose(sum(errors), 163.0, atol=1e-8)
    assert np.isclose(sum(signed), -41.0, atol=1e-8)

    # D1: signed-bias cancellation.
    d1_red = broken_mean_from_signed(signed)
    d1_green = fixed_mean_from_distances(errors)
    assert np.isclose(d1_red, 2.05, atol=1e-9)
    assert np.isclose(d1_green, 8.15, atol=1e-9)
    assert not np.isclose(d1_red, d1_green)

    # D2: percentile definition silently changed.
    d2_red = broken_p95_higher(errors)
    d2_green = fixed_p95_linear(errors)
    assert np.isclose(d2_red, 50.0, atol=1e-8)
    assert np.isclose(d2_green, 12.0, atol=1e-8)

    # D3: favorable sample selection masquerading as improvement.
    d3_filtered = broken_filter_large_errors(errors)
    d3_red = oracle_stats(d3_filtered)
    d3_green = oracle_stats(fixed_full_error_set(errors))
    assert d3_red["count"] == 19 and d3_green["count"] == 20
    assert np.isclose(d3_red["mean_mm"], 113 / 19, atol=1e-8)
    assert np.isclose(d3_green["max_mm"], 50.0, atol=1e-8)

    # D4: index-based matching corruption. IDs still look unique.
    swapped = broken_index_alignment(samples)
    swapped_result = _eval(swapped)
    after_swapped = swapped_result["valid_samples"]
    err00 = float(after_swapped[0]["euclidean_error_mm"])
    err01 = float(after_swapped[1]["euclidean_error_mm"])
    assert np.isclose(err00, 8.0, atol=1e-8)
    assert np.isclose(err01, 9.0, atol=1e-8)
    assert np.isclose(swapped_result["statistics"]["mean_mm"], 8.85, atol=1e-8)
    assert np.isclose(swapped_result["signed_delta_x_mean_mm"], -2.05, atol=1e-8)
    orig_prov = normal_provenance(samples)
    require_source_provenance(samples, orig_prov)
    broken_prov = swapped_provenance(orig_prov)
    detected = False
    try:
        require_source_provenance(swapped, broken_prov)
    except ValueError as exc:
        detected = "estimate_source_sample_id" in str(exc)
    assert detected, "the independent provenance audit should catch wrong association"

    # D5: absent GT is NOT 0mm. Correctly rejected; wrong fill boosts denominator.
    missing = with_missing_gt(samples)
    d5_green = fixed_keep_missing_gt_as_rejected(missing)
    d5_bad_report = _eval(broken_fill_missing_gt(missing))
    assert (d5_green["input_sample_count"], d5_green["valid_count"], d5_green["rejected_count"]) == (21, 20, 1)
    assert d5_green["rejected_samples"][0]["reason"] == "MISSING_GROUND_TRUTH"
    assert d5_bad_report["statistics"]["count"] == 21
    assert np.isclose(d5_bad_report["statistics"]["mean_mm"], 163.0 / 21.0, atol=1e-8)
    assert np.isclose(d5_bad_report["statistics"]["max_mm"], 50.0, atol=1e-8)

    return {
        "experiment": "EXP036-D",
        "data_kind": "synthetic_deterministic",
        "baseline": baseline_stats,
        "D1_signed_mean_bug": {
            "RED_incorrect_mean_mm": d1_red, "GREEN_correct_mean_mm": d1_green,
            "underestimate_mm": d1_green - d1_red, "red_observed": True, "green_verified": True,
        },
        "D2_percentile_method_bug": {
            "RED_higher_p95_mm": d2_red, "GREEN_linear_p95_mm": d2_green,
            "method_difference_mm": d2_red - d2_green, "red_observed": True, "green_verified": True,
        },
        "D3_hidden_outlier_bug": {
            "RED_filtered": d3_red, "GREEN_full": d3_green,
            "removed_sample_ids": ["S19"], "red_observed": True, "green_verified": True,
        },
        "D4_misaligned_estimates_bug": {
            "RED_S00_euclidean_mm": err00, "RED_S01_euclidean_mm": err01,
            "RED_mean_mm": swapped_result["statistics"]["mean_mm"],
            "RED_signed_bias_mm": swapped_result["signed_delta_x_mean_mm"],
            "GREEN_provenance_mismatch_rejected": detected,
            "caution": "record IDs alone do not prove estimates came from the claimed samples",
            "red_observed": True, "green_verified": True,
        },
        "D5_missing_gt_zero_fill_bug": {
            "RED_count": d5_bad_report["statistics"]["count"],
            "RED_mean_mm": d5_bad_report["statistics"]["mean_mm"],
            "RED_max_mm": d5_bad_report["statistics"]["max_mm"],
            "GREEN_input_count": d5_green["input_sample_count"],
            "GREEN_evaluated_count": d5_green["valid_count"],
            "GREEN_missing_gt_count": d5_green["rejected_count"],
            "GREEN_missing_gt_reason": d5_green["rejected_samples"][0]["reason"],
            "red_observed": True, "green_verified": True,
        },
        "publishable": False,
        "published_positions": [],
        "caution": "Synthetic metric-debug experiment only; no physical sensor-accuracy claim",
    }


def run(project_root: str | Path = "."):
    report = build_fault_report()
    out_path = Path(project_root) / "results" / "exp036" / "exp036_D_fault_injection.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for idx, name in enumerate((
        "D1_signed_mean_bug", "D2_percentile_method_bug", "D3_hidden_outlier_bug",
        "D4_misaligned_estimates_bug", "D5_missing_gt_zero_fill_bug"
    ), start=1):
        item = report[name]
        assert item["red_observed"] and item["green_verified"]
        print(f"RED D{idx}: expected metric corruption observed | GREEN D{idx}: oracle/guard PASS")
    print("Saved:", out_path)
    print("EXP036-D RED/PATCH/GREEN VERIFIED; EXP036 UNIT COMPLETION: PENDING")
    return report


if __name__ == "__main__":
    run()
