"""EXP036-A: independent hand-oracle tests, input contracts, JSON provenance."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.experiments.exp036_error_minimal import (
    CAMERA_FRAME, HAND_CAMERA_M, deproject_pixel_m,
    position_error_mm, summarize_errors_mm, run,
)


@pytest.fixture
def intrinsics():
    return dict(u_px=380.0, v_px=270.0, z_m=1.0,
                fx_px=600.0, fy_px=600.0, cx_px=320.0, cy_px=240.0)


def test_a1_hand_oracle_and_shape(intrinsics):
    p = deproject_pixel_m(**intrinsics)
    assert p.shape == (3,) and p.dtype == np.float64
    np.testing.assert_allclose(p, [0.1, 0.05, 1.0], rtol=0, atol=1e-12)


def test_a1_principal_point_is_optical_axis(intrinsics):
    intrinsics.update(u_px=320.0, v_px=240.0)
    np.testing.assert_allclose(deproject_pixel_m(**intrinsics), [0, 0, 1], rtol=0, atol=1e-12)


def test_a2_depth_changed_hand_oracle(intrinsics):
    intrinsics["z_m"] = 1.02
    p = deproject_pixel_m(**intrinsics)
    np.testing.assert_allclose(p, [0.102, 0.051, 1.02], rtol=0, atol=1e-12)
    np.testing.assert_allclose((p-HAND_CAMERA_M)*1000, [2,1,20], rtol=0, atol=1e-10)
    np.testing.assert_allclose(position_error_mm(p, HAND_CAMERA_M), np.sqrt(405),
                               rtol=0, atol=1e-10)


def test_a2_depth_changes_all_coordinates(intrinsics):
    p0 = deproject_pixel_m(**intrinsics)
    intrinsics["z_m"] = 2.0
    p1 = deproject_pixel_m(**intrinsics)
    np.testing.assert_allclose(p1, 2*p0, rtol=0, atol=1e-12)


def test_a3_fx_increase_hand_oracle(intrinsics):
    intrinsics["fx_px"] = 630.0
    p = deproject_pixel_m(**intrinsics)
    np.testing.assert_allclose(p, [2/21, 0.05, 1], rtol=0, atol=1e-12)
    np.testing.assert_allclose((p-HAND_CAMERA_M)*1000, [-100/21, 0, 0],
                               rtol=0, atol=1e-10)


def test_a3_only_x_changes(intrinsics):
    p0 = deproject_pixel_m(**intrinsics)
    intrinsics["fx_px"] = 630.0
    p1 = deproject_pixel_m(**intrinsics)
    assert p1[0] < p0[0]
    np.testing.assert_array_equal(p1[1:], p0[1:])


def test_a4_translation_bias_oracle(intrinsics):
    p = deproject_pixel_m(**intrinsics)
    correct = p + [0.30, -0.10, 0.20]
    wrong = p + [0.31, -0.10, 0.20]
    np.testing.assert_allclose(correct, [0.40, -0.05, 1.20], rtol=0, atol=1e-12)
    np.testing.assert_allclose(wrong, [0.41, -0.05, 1.20], rtol=0, atol=1e-12)
    np.testing.assert_allclose((wrong-correct)*1000, [10, 0, 0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(position_error_mm(wrong, correct), 10, rtol=0, atol=1e-10)


def test_a5_hand_statistics():
    stats = summarize_errors_mm([0, 2, 4, 6, 18])
    assert stats["count"] == 5
    assert stats["percentile_method"] == "linear"
    for k, expected in {"mean_mm":6,"median_mm":4,"p95_mm":15.6,"max_mm":18}.items():
        np.testing.assert_allclose(stats[k], expected, rtol=0, atol=1e-12)


def test_a5_permutation_invariance():
    x = [18, 4, 0, 6, 2]
    assert summarize_errors_mm(x) == summarize_errors_mm(list(reversed(x)))


def test_a5_tail_not_equal_mean():
    s = summarize_errors_mm([0, 2, 4, 6, 18])
    assert s["p95_mm"] > s["mean_mm"] > s["median_mm"]


@pytest.mark.parametrize("bad", [0, -1, -1e-9, float("nan"), float("inf"), True, "1.0"])
def test_reject_invalid_depth(intrinsics, bad):
    intrinsics["z_m"] = bad
    with pytest.raises(ValueError):
        deproject_pixel_m(**intrinsics)


@pytest.mark.parametrize("key", ["fx_px", "fy_px"])
@pytest.mark.parametrize("bad", [0, -600.0, float("nan")])
def test_reject_invalid_focal(intrinsics, key, bad):
    intrinsics[key] = bad
    with pytest.raises(ValueError):
        deproject_pixel_m(**intrinsics)


@pytest.mark.parametrize("key", ["u_px", "v_px", "cx_px", "cy_px"])
def test_reject_nonfinite_pixel_or_principal(intrinsics, key):
    intrinsics[key] = float("inf")
    with pytest.raises(ValueError):
        deproject_pixel_m(**intrinsics)


@pytest.mark.parametrize("estimate,reference", [
    ([1, 2], [1, 2, 3]),
    ([[1, 2, 3]], [1, 2, 3]),
    ([1, 2, 3], [1, 2, "3"]),
    ([1, True, 3], [1, 2, 3]),
    ([1, 2, float("nan")], [1, 2, 3]),
])
def test_reject_invalid_xyz(estimate, reference):
    with pytest.raises(ValueError):
        position_error_mm(estimate, reference)


@pytest.mark.parametrize("bad", [
    [], [[1, 2]], [-1, 2], [0, float("nan")], [1, float("inf")], [1, True],
])
def test_reject_invalid_error_samples(bad):
    with pytest.raises(ValueError):
        summarize_errors_mm(bad)


def test_position_error_norm_not_signed():
    np.testing.assert_allclose(position_error_mm([0,0.01,0], [0,0,0]), 10,
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(position_error_mm([0,-0.01,0], [0,0,0]), 10,
                               rtol=0, atol=1e-12)


def test_json_output_is_synthetic_and_blocked(tmp_path):
    before = HAND_CAMERA_M.copy()
    result = run(tmp_path)
    path = tmp_path / "results/exp036/exp036_A_error_minimal.json"
    assert path.is_file()
    read = json.loads(path.read_text(encoding="utf-8"))
    assert read == result
    assert read["reference_kind"] == "synthetic_gt"
    assert read["publishable"] is False
    assert read["calibration_verified"] is False
    assert read["A1"]["frame_id"] == CAMERA_FRAME
    assert read["A5"]["statistics"]["percentile_method"] == "linear"
    np.testing.assert_array_equal(HAND_CAMERA_M, before)


def test_repeatable_output(tmp_path):
    run(tmp_path)
    path = tmp_path / "results/exp036/exp036_A_error_minimal.json"
    first = path.read_bytes()
    run(tmp_path)
    assert first == path.read_bytes()
