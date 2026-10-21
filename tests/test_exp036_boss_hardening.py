"""EXP036 Boss code hardening: run against the learner's unmodified submission.

Install this file and exp036_boss_candidate_for_review.py side-by-side:
    python -m pytest -q test_exp036_boss_hardening.py

Expected original: 2 passed, 3 failed (RED). Patch learner code, re-run GREEN.
"""
import numpy as np
import pytest

from src.experiments.exp036_boss_student import evaluate_localizations


def base_record():
    return {
        "sample_id": "S001",
        "cluster_status": "VALID",
        "estimate_m": [0.100, 0.05, 1.0],
        "synthetic_gt_m": [0.101, 0.05, 1.0],
        "estimate_frame_id": "camera_optical_frame",
        "gt_frame_id": "camera_optical_frame",
        "estimate_unit": "m",
        "gt_unit": "m",
        "gt_source": "independent_synthetic_design",
    }


def test_valid_independent_synthetic_sample_still_works():
    result = evaluate_localizations([base_record()])
    assert result["status"] == "EVALUATED"
    assert result["evaluated_count"] == 1
    assert result["excluded_count"] == 0
    assert result["stats"]["mean_mm"] == pytest.approx(1.0)
    assert result["publishable"] is False
    assert result["published_positions"] == []


def test_missing_gt_is_not_zero_error():
    record = base_record()
    record["synthetic_gt_m"] = None
    result = evaluate_localizations([record])
    assert result["evaluated_count"] == 0
    assert result["excluded"][0]["reason"] == "MISSING_GROUND_TRUTH"


def test_unhashable_gt_source_is_excluded_with_reason_not_crash():
    record = base_record()
    record["gt_source"] = ["independent_synthetic_design"]
    result = evaluate_localizations([record])
    assert result["evaluated_count"] == 0
    assert result["excluded_count"] == 1
    assert result["excluded"][0]["reason"] == "UNQUALIFIED_GT_SOURCE"


def test_arithmetic_overflow_is_excluded_not_reported_as_infinite_accuracy():
    record = base_record()
    record["estimate_m"] = [1e308, 0.0, 0.0]
    record["synthetic_gt_m"] = [-1e308, 0.0, 0.0]
    with np.errstate(over="ignore", invalid="ignore"):
        result = evaluate_localizations([record])
    assert result["evaluated_count"] == 0
    assert result["excluded_count"] == 1
    assert result["excluded"][0]["reason"] == "NONFINITE_COMPUTED_ERROR"
    assert result["stats"]["count"] == 0


def test_p95_method_is_recorded_for_reproducibility():
    result = evaluate_localizations([base_record()])
    assert result["stats"]["percentile_method"] == "linear"
