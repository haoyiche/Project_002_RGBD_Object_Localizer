"""EXP036-D: each injected bug should be detected, each repair independently verified."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.experiments.exp036_error_benchmark_c import FRAME_ID, make_deterministic_samples
from src.geometry.localization_error_metrics import evaluate_synthetic_samples
from src.experiments.exp036_fault_injection_d import (
    broken_filter_large_errors, broken_fill_missing_gt, broken_index_alignment,
    broken_mean_from_signed, broken_p95_higher, build_fault_report,
    fixed_full_error_set, fixed_keep_missing_gt_as_rejected, fixed_mean_from_distances,
    fixed_p95_linear, normal_provenance, oracle_stats, require_source_provenance,
    run, swapped_provenance, with_missing_gt,
)


@pytest.fixture
def data():
    rows = make_deterministic_samples()
    report = evaluate_synthetic_samples(rows, expected_frame_id=FRAME_ID)
    errors = np.array([s["euclidean_error_mm"] for s in report["valid_samples"]])
    signed = np.array([s["signed_delta_x_mm"] for s in report["valid_samples"]])
    return rows, report, errors, signed


@pytest.mark.parametrize("key,expected", [
    ("count", 20), ("mean_mm", 8.15), ("median_mm", 5.5),
    ("p95_mm", 12.0), ("max_mm", 50.0)
])
def test_D0_baseline_is_unchanged(data, key, expected):
    _, report, errors, _ = data
    assert report["statistics"][key] == pytest.approx(expected)
    assert oracle_stats(errors)[key] == pytest.approx(expected)


def test_D1_expected_RED_signed_mean(data):
    _, _, _, signed = data
    assert broken_mean_from_signed(signed) == pytest.approx(2.05, abs=1e-9)


def test_D1_GREEN_mean_of_magnitudes(data):
    _, _, errors, _ = data
    assert fixed_mean_from_distances(errors) == pytest.approx(8.15, abs=1e-9)


def test_D1_signed_cancellation_minimal_counterexample():
    signed = np.array([50.0, -50.0, 50.0, -50.0])
    assert broken_mean_from_signed(signed) == 0.0
    assert fixed_mean_from_distances(np.abs(signed)) == 50.0


def test_D1_cancellation_should_not_replace_metric(data):
    _, _, errors, signed = data
    assert (fixed_mean_from_distances(errors) - broken_mean_from_signed(signed)) == pytest.approx(6.10, abs=1e-9)


def test_D2_expected_RED_higher(data):
    _, _, errors, _ = data
    assert broken_p95_higher(errors) == pytest.approx(50.0)


def test_D2_GREEN_linear(data):
    _, _, errors, _ = data
    assert fixed_p95_linear(errors) == pytest.approx(12.0)


def test_D2_method_is_part_of_contract(data):
    _, _, errors, _ = data
    assert broken_p95_higher(errors) - fixed_p95_linear(errors) == pytest.approx(38.0)


@pytest.mark.parametrize("key,expected", [
    ("count", 19), ("mean_mm", 113/19), ("median_mm", 5.0),
    ("p95_mm", 10.0), ("max_mm", 10.0)
])
def test_D3_expected_RED_silently_filter_S19(data, key, expected):
    _, _, errors, _ = data
    assert oracle_stats(broken_filter_large_errors(errors))[key] == pytest.approx(expected, abs=1e-9)


def test_D3_GREEN_keeps_valid_outlier(data):
    _, _, errors, _ = data
    full = fixed_full_error_set(errors)
    assert full.size == 20 and np.max(full) == pytest.approx(50.0)
    assert full is not errors and not np.shares_memory(full, errors)


def test_D3_no_mutation_of_input(data):
    _, _, errors, _ = data
    before = errors.copy()
    broken_filter_large_errors(errors)
    np.testing.assert_array_equal(errors, before)


@pytest.mark.parametrize("index,expected_delta,expected_error", [
    (0, 8.0, 8.0), (1, -9.0, 9.0)
])
def test_D4_expected_RED_swapped_payload(index, expected_delta, expected_error, data):
    rows, _, _, _ = data
    swapped = broken_index_alignment(rows)
    item = evaluate_synthetic_samples(swapped, expected_frame_id=FRAME_ID)["valid_samples"][index]
    assert item["signed_delta_x_mm"] == pytest.approx(expected_delta, abs=1e-9)
    assert item["euclidean_error_mm"] == pytest.approx(expected_error, abs=1e-9)


def test_D4_swap_increases_mean_without_changing_signed_bias(data):
    rows, original, _, _ = data
    bad = evaluate_synthetic_samples(broken_index_alignment(rows), expected_frame_id=FRAME_ID)
    assert bad["statistics"]["mean_mm"] == pytest.approx(8.85, abs=1e-9)
    assert bad["signed_delta_x_mean_mm"] == pytest.approx(original["signed_delta_x_mean_mm"], abs=1e-9)


def test_D4_record_ids_stay_unchanged_after_swap(data):
    rows, _, _, _ = data
    swapped = broken_index_alignment(rows)
    assert [s["sample_id"] for s in swapped] == [s["sample_id"] for s in rows]
    assert rows[0]["estimate_m"] != swapped[0]["estimate_m"]


def test_D4_GREEN_independent_provenance_checks_clean(data):
    rows, _, _, _ = data
    assert require_source_provenance(rows, normal_provenance(rows)) is None


def test_D4_GREEN_independent_provenance_detects_swap(data):
    rows, _, _, _ = data
    with pytest.raises(ValueError, match="estimate_source_sample_id"):
        require_source_provenance(broken_index_alignment(rows), swapped_provenance(normal_provenance(rows)))


def test_D4_provenance_cannot_be_inferred_from_id_alone(data):
    rows, _, _, _ = data
    # A matching, self-reported source ID would not prove that payload actually came from that source.
    assert require_source_provenance(broken_index_alignment(rows), normal_provenance(rows)) is None


@pytest.mark.parametrize("bad", [None, [], [None], [{"record_sample_id": "S00"}]])
def test_D4_provenance_malformed_inputs_fail(data, bad):
    rows, _, _, _ = data
    with pytest.raises(ValueError):
        require_source_provenance(rows, bad)


def test_D5_missing_GT_reported_and_excluded(data):
    rows, _, _, _ = data
    report = fixed_keep_missing_gt_as_rejected(with_missing_gt(rows))
    assert (report["input_sample_count"], report["valid_count"], report["rejected_count"]) == (21, 20, 1)
    assert report["rejected_samples"][0]["reason"] == "MISSING_GROUND_TRUTH"
    assert report["statistics"]["mean_mm"] == pytest.approx(8.15)
    assert report["statistics"]["count"] == 20


def test_D5_expected_RED_zero_fill_increases_count(data):
    rows, _, _, _ = data
    report = evaluate_synthetic_samples(broken_fill_missing_gt(with_missing_gt(rows)), expected_frame_id=FRAME_ID)
    assert report["valid_count"] == 21 and report["rejected_count"] == 0
    assert report["statistics"]["count"] == 21


def test_D5_expected_RED_makes_mean_optimistic(data):
    rows, _, _, _ = data
    report = evaluate_synthetic_samples(broken_fill_missing_gt(with_missing_gt(rows)), expected_frame_id=FRAME_ID)
    assert report["statistics"]["mean_mm"] == pytest.approx(163.0 / 21.0, abs=1e-9)
    assert report["statistics"]["mean_mm"] < 8.15


def test_D5_max_does_not_necessarily_decrease(data):
    rows, _, _, _ = data
    report = evaluate_synthetic_samples(broken_fill_missing_gt(with_missing_gt(rows)), expected_frame_id=FRAME_ID)
    assert report["statistics"]["max_mm"] == pytest.approx(50.0)


def test_D5_never_change_original_samples(data):
    rows, _, _, _ = data
    before = copy.deepcopy(rows)
    broken_fill_missing_gt(with_missing_gt(rows))
    assert rows == before


def test_D5_rejected_sample_not_as_zero_error(data):
    rows, _, _, _ = data
    item = fixed_keep_missing_gt_as_rejected(with_missing_gt(rows))["rejected_samples"][0]
    assert "euclidean_error_mm" not in item


def test_all_five_bug_scenarios_and_guards(data):
    result = build_fault_report()
    assert result["baseline"]["count"] == 20
    for name in (
        "D1_signed_mean_bug", "D2_percentile_method_bug", "D3_hidden_outlier_bug",
        "D4_misaligned_estimates_bug", "D5_missing_gt_zero_fill_bug",
    ):
        assert result[name]["red_observed"] is True
        assert result[name]["green_verified"] is True
    assert result["publishable"] is False and result["published_positions"] == []


def test_report_repeat_is_deterministic(tmp_path):
    first = run(tmp_path)
    path = tmp_path / "results/exp036/exp036_D_fault_injection.json"
    raw = path.read_bytes()
    second = run(tmp_path)
    assert raw == path.read_bytes() and first == second
    assert json.loads(raw)["experiment"] == "EXP036-D"
