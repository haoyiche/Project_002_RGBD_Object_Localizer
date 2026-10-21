"""EXP036-C unit tests: independent oracles and fail-closed evaluation contracts."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.experiments.exp036_error_benchmark_c import (
    FRAME_ID, SIGNED_X_MM, make_deterministic_samples, run,
)
from src.geometry.localization_error_metrics import evaluate_synthetic_samples


@pytest.fixture
def samples():
    return make_deterministic_samples()


def evaluate(samples):
    return evaluate_synthetic_samples(samples, expected_frame_id=FRAME_ID)


@pytest.mark.parametrize("i,delta", list(enumerate(SIGNED_X_MM)))
def test_each_sample_independent_gt_and_injection(i, delta, samples):
    row = samples[i]
    assert row["sample_id"] == f"S{i:02d}"
    independent_gt = [0.10 + 0.01 * (i % 5), 0.05 + 0.01 * (i // 5), 1.00]
    np.testing.assert_allclose(row["synthetic_gt_m"], independent_gt, rtol=0, atol=1e-12)
    np.testing.assert_allclose(row["estimate_m"],
                               [independent_gt[0] + delta / 1000, independent_gt[1], 1],
                               rtol=0, atol=1e-12)
    assert row["gt_unit"] == row["estimate_unit"] == "m"
    assert row["gt_frame_id"] == row["estimate_frame_id"] == FRAME_ID


@pytest.mark.parametrize("i,delta", list(enumerate(SIGNED_X_MM)))
def test_each_signed_and_euclidean_oracle(i, delta, samples):
    row = evaluate(samples)["valid_samples"][i]
    np.testing.assert_allclose(row["signed_delta_xyz_mm"], [delta, 0.0, 0.0],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(row["euclidean_error_mm"], abs(delta),
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("index,gt,est,delta,err", [
    (0, [0.10, 0.05, 1.0], [0.101, 0.05, 1.0], 1.0, 1.0),
    (1, [0.11, 0.05, 1.0], [0.108, 0.05, 1.0], -2.0, 2.0),
    (19, [0.14, 0.08, 1.0], [0.09, 0.08, 1.0], -50.0, 50.0),
])
def test_user_C1_three_hand_oracles(index, gt, est, delta, err, samples):
    row = evaluate(samples)["valid_samples"][index]
    np.testing.assert_allclose(row["synthetic_gt_m"], gt, rtol=0, atol=1e-12)
    np.testing.assert_allclose(row["estimate_m"], est, rtol=0, atol=1e-12)
    np.testing.assert_allclose(row["signed_delta_x_mm"], delta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(row["euclidean_error_mm"], err, rtol=0, atol=1e-10)


def test_shapes_and_counts(samples):
    out = evaluate(samples)
    assert out["shapes"] == {
        "gt_points": [20, 3],
        "estimate_points": [20, 3],
        "signed_delta_xyz_mm": [20, 3],
        "signed_delta_x_mm": [20],
        "euclidean_errors_mm": [20],
    }
    assert (out["input_sample_count"], out["valid_count"], out["rejected_count"]) == (20, 20, 0)


def test_C2_full_statistics(samples):
    stat = evaluate(samples)["statistics"]
    assert stat["count"] == 20 and stat["percentile_method"] == "linear"
    for key, expected in {"mean_mm": 8.15, "median_mm": 5.5,
                          "p95_mm": 12.0, "max_mm": 50.0}.items():
        assert stat[key] == pytest.approx(expected, abs=1e-10)


def test_C2_sorted_errors_and_sum(samples):
    errors = [r["euclidean_error_mm"] for r in evaluate(samples)["valid_samples"]]
    assert sum(errors) == pytest.approx(163.0, abs=1e-10)
    np.testing.assert_allclose(sorted(errors),
                               [1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 8,
                                10, 10, 10, 10, 10, 10, 50], rtol=0, atol=1e-10)


def test_C3_signed_bias_distinct_from_mean_error(samples):
    out = evaluate(samples)
    assert sum(SIGNED_X_MM) == -41
    assert out["signed_delta_x_mean_mm"] == pytest.approx(-2.05, abs=1e-10)
    np.testing.assert_allclose(out["mean_signed_delta_xyz_mm"], [-2.05, 0, 0],
                               rtol=0, atol=1e-10)
    assert abs(out["signed_delta_x_mean_mm"]) < out["statistics"]["mean_mm"]


def test_C4_sensitivity_is_separate(samples):
    full = evaluate(samples)
    reduced = evaluate([s for s in samples if s["sample_id"] != "S19"])
    assert full["statistics"]["mean_mm"] == pytest.approx(8.15, abs=1e-10)
    assert reduced["statistics"]["count"] == 19
    for key, expected in {"mean_mm": 113.0 / 19.0, "median_mm": 5.0,
                          "p95_mm": 10.0, "max_mm": 10.0}.items():
        assert reduced["statistics"][key] == pytest.approx(expected, abs=1e-10)
    assert full["valid_samples"][-1]["injected_outlier"] is True
    assert full["valid_count"] == 20


@pytest.mark.parametrize("change,reason", [
    ({"estimate_frame_id": "robot_base"}, "FRAME_MISMATCH"),
    ({"gt_frame_id": "world"}, "FRAME_MISMATCH"),
    ({"frame_id": "wrong_frame"}, "FRAME_MISMATCH"),
    ({"gt_frame_id": None}, "FRAME_MISMATCH"),
    ({"synthetic_gt_m": None}, "MISSING_GROUND_TRUTH"),
    ({"estimate_m": None}, "MISSING_ESTIMATE"),
    ({"gt_unit": "mm"}, "UNCONFIRMED_UNITS"),
    ({"estimate_unit": "mm"}, "UNCONFIRMED_UNITS"),
    ({"reference_kind": "measured_gt"}, "REFERENCE_KIND_MISMATCH"),
    ({"injected_outlier": "yes"}, "INVALID_OUTLIER_METADATA"),
    ({"synthetic_gt_m": [0.1, float("nan"), 1]}, "INVALID_GROUND_TRUTH"),
    ({"estimate_m": [0.1, float("inf"), 1]}, "INVALID_ESTIMATE"),
    ({"estimate_m": [0.1, True, 1]}, "INVALID_ESTIMATE"),
    ({"estimate_m": [0.1, 0.05]}, "INVALID_ESTIMATE"),
    ({"synthetic_gt_m": [1, "2", 3]}, "INVALID_GROUND_TRUTH"),
])
def test_C5_exclusions_never_count_as_zero(samples, change, reason):
    item = dict(samples[1])
    item.update(change)
    result = evaluate([samples[0], item])
    assert result["valid_count"] == 1 and result["rejected_count"] == 1
    assert result["rejected_samples"][0]["reason"] == reason
    assert result["statistics"]["count"] == 1
    assert result["statistics"]["mean_mm"] == pytest.approx(1.0, abs=1e-10)
    assert "euclidean_error_mm" not in result["rejected_samples"][0]


def test_rejected_sample_keeps_origin_metadata(samples):
    bad = dict(samples[1], sample_id="MISS", synthetic_gt_m=None)
    row = evaluate([samples[0], bad])["rejected_samples"][0]
    assert row["sample_id"] == "MISS"
    assert row["gt_present"] is False
    assert row["estimate_present"] is True


def test_no_valid_samples_fails_not_returns_zero(samples):
    missing = dict(samples[0], synthetic_gt_m=None)
    with pytest.raises(ValueError, match="no valid"):
        evaluate([missing])


def test_duplicate_ids_fail(samples):
    with pytest.raises(ValueError, match="duplicate sample_id"):
        evaluate([samples[0], dict(samples[1], sample_id="S00")])


@pytest.mark.parametrize("bad", [None, {}, (), [], ["x"], [42]])
def test_invalid_dataset_container_rejected(bad):
    with pytest.raises(ValueError):
        evaluate_synthetic_samples(bad, expected_frame_id=FRAME_ID)


@pytest.mark.parametrize("bad_frame", [None, "", "  ", 2])
def test_invalid_expected_frame_rejected(samples, bad_frame):
    with pytest.raises(ValueError, match="expected_frame_id"):
        evaluate_synthetic_samples(samples, expected_frame_id=bad_frame)


def test_does_not_mutate_inputs(samples):
    before = copy.deepcopy(samples)
    evaluate(samples)
    assert samples == before


def test_benchmark_reproducible_twice(tmp_path):
    first = run(tmp_path)
    filename = tmp_path / "results/exp036/exp036_C_error_benchmark.json"
    bytes_one = filename.read_bytes()
    second = run(tmp_path)
    bytes_two = filename.read_bytes()
    assert first == second and bytes_one == bytes_two
    assert bytes_one.endswith(b"\n")
    assert json.loads(bytes_one)["experiment"] == "EXP036-C"


def test_report_synthetic_not_physical_or_publishable(tmp_path):
    report = run(tmp_path)
    assert report["reference_kind"] == "synthetic_gt"
    assert report["data_kind"] == "synthetic_deterministic"
    assert report["random_seed"] is None and report["rng_used"] is False
    assert report["calibration_verified"] is False
    assert report["publishable"] is False and report["published_positions"] == []
    assert report["full_20_sample_benchmark"]["publishable"] is False
    assert report["sensitivity_without_S19"]["benchmark"]["published_positions"] == []


def test_report_sensitivity_does_not_overwrite_full_stats(tmp_path):
    report = run(tmp_path)
    full = report["full_20_sample_benchmark"]
    fewer = report["sensitivity_without_S19"]["benchmark"]
    assert full["statistics"]["count"] == 20
    assert fewer["statistics"]["count"] == 19
    assert any(r["sample_id"] == "S19" for r in full["valid_samples"])
    assert not any(r["sample_id"] == "S19" for r in fewer["valid_samples"])


def test_report_error_demo_diagnoses_both_failures(tmp_path):
    report = run(tmp_path)
    demo = report["invalid_sample_diagnostics_demo"]
    assert demo["valid_count"] == 1 and demo["rejected_count"] == 2
    assert {r["reason"] for r in demo["rejected_samples"]} == {
        "FRAME_MISMATCH", "MISSING_GROUND_TRUTH"
    }
