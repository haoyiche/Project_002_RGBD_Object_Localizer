"""EXP036-B independent prediction oracles and provenance / shape tests."""
from __future__ import annotations

import json
import numpy as np
import pytest

from src.experiments.exp036_error_ablation_b import (
    CAMERA_FRAME,
    BASE_FRAME,
    REFERENCE_KIND,
    depth_symmetry_b1,
    focal_spatial_b2,
    translation_bias_b3,
    run,
)
from src.experiments.exp036_error_minimal import deproject_pixel_m, position_error_mm


@pytest.fixture
def b1():
    return depth_symmetry_b1()


@pytest.fixture
def b2():
    return focal_spatial_b2()


@pytest.fixture
def b3():
    return translation_bias_b3()


def test_b1_reference_frame_and_ground_truth(b1):
    assert b1["frame_id"] == CAMERA_FRAME
    np.testing.assert_allclose(b1["synthetic_gt_m"], [0.10, 0.05, 1.0], rtol=0, atol=1e-12)


@pytest.mark.parametrize("index,expected", [
    (0, [0.101, 0.0505, 1.01]),
    (1, [0.099, 0.0495, 0.99]),
])
def test_b1_positions_match_prediction(b1, index, expected):
    np.testing.assert_allclose(b1["samples"][index]["estimate_m"], expected,
                               rtol=0, atol=1e-12)


@pytest.mark.parametrize("index,expected", [
    (0, [1.0, 0.5, 10.0]),
    (1, [-1.0, -0.5, -10.0]),
])
def test_b1_signed_deltas_match_prediction(b1, index, expected):
    np.testing.assert_allclose(b1["samples"][index]["signed_delta_mm"], expected,
                               rtol=0, atol=1e-10)


@pytest.mark.parametrize("index", [0, 1])
def test_b1_euclidean_error_nonzero(b1, index):
    np.testing.assert_allclose(b1["samples"][index]["euclidean_error_mm"],
                               np.sqrt(101.25), rtol=0, atol=1e-10)


def test_b1_mean_signed_zero(b1):
    np.testing.assert_allclose(b1["mean_signed_delta_mm"], [0, 0, 0], rtol=0, atol=1e-10)


def test_b1_mean_distance_not_zero(b1):
    np.testing.assert_allclose(b1["mean_euclidean_error_mm"], np.sqrt(101.25),
                               rtol=0, atol=1e-10)
    assert b1["mean_euclidean_error_mm"] > 10


def test_b1_fixed_positive_bias_not_zero(b1):
    fixed = b1["fixed_plus_bias_illustration"]
    assert fixed["repeats"] == 2
    np.testing.assert_allclose(fixed["signed_delta_mean_mm"], [1.0, 0.5, 10.0],
                               rtol=0, atol=1e-10)


def test_b1_independent_norm_oracle(b1):
    delta = np.array(b1["samples"][0]["signed_delta_mm"])
    np.testing.assert_allclose(delta @ delta, 101.25, rtol=0, atol=1e-10)


def test_b1_no_claim_of_real_noise_distribution(b1):
    assert "noise-distribution" in b1["fixed_plus_bias_illustration"]["note"]


@pytest.mark.parametrize("index,expected", [(0, 0.0), (1, 0.1), (2, 0.3)])
def test_b2_fx_correct_x(b2, index, expected):
    np.testing.assert_allclose(b2["samples"][index]["synthetic_gt_m"][0],
                               expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("index,expected", [(0, 0.0), (1, 2/21), (2, 2/7)])
def test_b2_fx_wrong_x(b2, index, expected):
    np.testing.assert_allclose(b2["samples"][index]["estimate_m"][0],
                               expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("index,expected", [(0, 0.0), (1, -100/21), (2, -100/7)])
def test_b2_signed_error_mm(b2, index, expected):
    s = b2["samples"][index]
    np.testing.assert_allclose(s["signed_delta_mm"], [expected, 0, 0],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(s["euclidean_error_mm"], abs(expected),
                               rtol=0, atol=1e-10)


def test_b2_spatial_error_increases(b2):
    errors = [s["euclidean_error_mm"] for s in b2["samples"]]
    assert errors[0] == 0
    assert errors[0] < errors[1] < errors[2]
    np.testing.assert_allclose(errors[2], 3*errors[1], rtol=0, atol=1e-10)


def test_b2_y_and_z_unchanged(b2):
    for s in b2["samples"]:
        np.testing.assert_allclose(s["estimate_m"][1:], [0.05, 1.0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(s["synthetic_gt_m"][1:], [0.05, 1.0], rtol=0, atol=1e-12)


def test_b2_principal_point_fx_only_b2(b2):
    p1 = b2["samples"][0]
    assert p1["u_px"] == 320.0
    assert p1["euclidean_error_mm"] == 0.0
    assert "NOT for every calibration error" in b2["note"]


def test_b2_fx_errors_reverse_on_left_side():
    kw = dict(u_px=140, v_px=270, z_m=1.0, fx_px=600,
              fy_px=600, cx_px=320, cy_px=240)
    truth = deproject_pixel_m(**kw)
    estimate = deproject_pixel_m(**dict(kw, fx_px=630))
    np.testing.assert_allclose((estimate-truth)*1000, [100/7, 0, 0],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(position_error_mm(estimate, truth), 100/7,
                               rtol=0, atol=1e-10)


def test_b2_sample_shape_b2(b2):
    a = np.array([s["estimate_m"] for s in b2["samples"]])
    assert a.shape == (3, 3) and a.dtype == np.float64
    assert b2["frame_id"] == CAMERA_FRAME


@pytest.mark.parametrize("index,correct,wrong", [
    (0, [0.40, -0.05, 1.20], [0.41, -0.05, 1.20]),
    (1, [0.10, 0.00, 1.00], [0.11, 0.00, 1.00]),
    (2, [0.60, -0.20, 1.40], [0.61, -0.20, 1.40]),
])
def test_b3_positions_oracles(b3, index, correct, wrong):
    s = b3["samples"][index]
    np.testing.assert_allclose(s["synthetic_gt_base_m"], correct, rtol=0, atol=1e-12)
    np.testing.assert_allclose(s["estimate_base_m"], wrong, rtol=0, atol=1e-12)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_b3_same_signed_bias(b3, index):
    np.testing.assert_allclose(b3["samples"][index]["signed_delta_mm"],
                               [10, 0, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(b3["samples"][index]["euclidean_error_mm"],
                               10, rtol=0, atol=1e-10)


def test_b3_stats_b3(b3):
    stats = b3["statistics"]
    assert stats["count"] == 3 and stats["percentile_method"] == "linear"
    for key in ("mean_mm", "median_mm", "p95_mm", "max_mm"):
        np.testing.assert_allclose(stats[key], 10.0, rtol=0, atol=1e-10)


def test_b3_not_a_repeatability_experiment(b3):
    assert b3["sample_kind"] == "different_spatial_positions_not_repeat_measurements"
    assert "does not measure repeatability" in b3["note"]


def test_b3_camera_and_base_frame(b3):
    assert b3["frame_id"] == BASE_FRAME
    assert b3["source_frame_id"] == CAMERA_FRAME
    assert len(b3["samples"]) == 3


def test_b3_bias_constant_for_arbitrary_point():
    camera = np.array([-0.08, 0.02, 2.0])
    truth = camera + np.array([0.30, -0.10, 0.20])
    wrong = camera + np.array([0.31, -0.10, 0.20])
    np.testing.assert_allclose((wrong - truth)*1000, [10, 0, 0], rtol=0, atol=1e-10)


def test_report_synthetic_safety_and_units(tmp_path):
    r = run(tmp_path)
    assert r["experiment"] == "EXP036-B"
    assert r["reference_kind"] == REFERENCE_KIND == "synthetic_gt"
    assert r["publishable"] is False
    assert r["calibration_verified"] is False
    assert r["published_positions"] == []
    assert r["units"]["signed_difference_and_error"] == "mm"
    saved = tmp_path / "results/exp036/exp036_B_error_ablation.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8")) == r


def test_report_reproducible_bytes(tmp_path):
    run(tmp_path)
    p = tmp_path / "results/exp036/exp036_B_error_ablation.json"
    old = p.read_bytes()
    run(tmp_path)
    assert p.read_bytes() == old


def test_a_unchanged_reference_and_no_side_effect(tmp_path):
    before = np.array([0.10, 0.05, 1.0])
    run(tmp_path)
    after = deproject_pixel_m(u_px=380, v_px=270, z_m=1.0,
                             fx_px=600, fy_px=600, cx_px=320, cy_px=240)
    np.testing.assert_array_equal(after, before)


def test_negative_depth_not_injected_into_valid_model():
    with pytest.raises(ValueError, match="positive"):
        deproject_pixel_m(u_px=380, v_px=270, z_m=-1.0,
                          fx_px=600, fy_px=600, cx_px=320, cy_px=240)


def test_all_errors_nonnegative_and_signed_deviations_preserved(b1, b2, b3):
    for experiment in (b1, b2, b3):
        for s in experiment["samples"]:
            assert s["euclidean_error_mm"] >= 0
            assert len(s["signed_delta_mm"]) == 3
