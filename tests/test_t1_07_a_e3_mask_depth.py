"""T1-07-A E3 V2: geometry, negative controls, and -O regression."""

import subprocess
import sys

import numpy as np
import pytest

import src.experiments.t1_07_a_e3_mask_depth as e3


def sample():
    rgb = np.zeros((9, 9, 3), dtype=np.uint8)
    depth = np.ones((9, 9), dtype=np.float64)
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 2:7] = True
    K = np.array([
        [600.0, 0.0, 4.0],
        [0.0, 600.0, 4.0],
        [0.0, 0.0, 1.0],
    ])
    return rgb, depth, mask, K


def test_exact_point_counts():
    result = e3.evaluate_frame(*sample())
    variants = result["variants"]
    assert result["baseline_points"] == 15
    assert variants["erode"]["point_count"] == 3
    assert variants["dilate"]["point_count"] == 35
    assert variants["mask_strip"]["point_count"] == 12
    assert variants["depth_strip"]["point_count"] == 12
    assert result["strip_target_pixels_removed"] == 3


def test_same_cloud_different_upstream_sources():
    result = e3.evaluate_frame(*sample())
    a = result["variants"]["mask_strip"]
    b = result["variants"]["depth_strip"]
    assert result["strip_outputs_identical"] is True
    assert a["point_count"] == b["point_count"]
    assert a["centroid_shift_mm"] == b["centroid_shift_mm"]
    assert a["mask_pixels"] == 12
    assert b["mask_pixels"] == 15
    assert result["mask_strip_source"] == "MASK_REMOVAL"
    assert result["depth_strip_source"] == "DEPTH_ZEROED"


def test_input_arrays_are_not_modified():
    inputs = sample()
    saved = tuple(x.copy() for x in inputs)
    e3.evaluate_frame(*inputs)
    for actual, expected in zip(inputs, saved):
        np.testing.assert_array_equal(actual, expected)


def test_empty_baseline_rejected():
    rgb, depth, mask, K = sample()
    depth[:] = 0.0
    with pytest.raises(ValueError, match="EMPTY_BASELINE"):
        e3.evaluate_frame(rgb, depth, mask, K)


def test_empty_eroded_cloud_rejected():
    rgb, depth, mask, K = sample()
    mask[:] = False
    mask[4, 4] = True
    with pytest.raises(ValueError, match="EMPTY_PERTURBED_CLOUD"):
        e3.evaluate_frame(rgb, depth, mask, K)


def test_invalid_intrinsics_rejected():
    rgb, depth, mask, K = sample()
    K[0, 0] = 0.0
    with pytest.raises(ValueError, match="INVALID_FOCAL_LENGTH"):
        e3.evaluate_frame(rgb, depth, mask, K)


def test_fault_injection_detects_point_mismatch(monkeypatch):
    real_cloud = e3.mask_to_pointcloud

    def faulty_cloud(depth, rgb, mask, *args, **kwargs):
        points, colors, selected = real_cloud(depth, rgb, mask, *args, **kwargs)
        if np.any(depth == 0) and len(points):
            points = points.copy()
            points[0, 0] += 0.001
        return points, colors, selected

    monkeypatch.setattr(e3, "mask_to_pointcloud", faulty_cloud)
    with pytest.raises(ValueError, match="STRIP_OUTPUT_MISMATCH"):
        e3.evaluate_frame(*sample())


def test_optimized_python_still_rejects_fault():
    """A real subprocess verifies -O; pytest's own asserts are not the oracle."""
    program = r'''
import numpy as np
import src.experiments.t1_07_a_e3_mask_depth as e3

rgb = np.zeros((9, 9, 3), dtype=np.uint8)
depth = np.ones((9, 9), dtype=np.float64)
mask = np.zeros((9, 9), dtype=bool)
mask[3:6, 2:7] = True
K = np.array([[600., 0., 4.], [0., 600., 4.], [0., 0., 1.]])
original = e3.mask_to_pointcloud

def broken(d, r, m, *args, **kwargs):
    points, colors, selected = original(d, r, m, *args, **kwargs)
    if np.any(d == 0) and len(points):
        points = points.copy()
        points[0, 0] += 0.001
    return points, colors, selected

e3.mask_to_pointcloud = broken
try:
    e3.evaluate_frame(rgb, depth, mask, K)
except ValueError as exc:
    if "STRIP_OUTPUT_MISMATCH" not in str(exc):
        raise
    print("T107_A_E3_OPTIMIZED_FAIL_CLOSED_PASS")
else:
    raise SystemExit("OPTIMIZED_FAIL_CLOSED_NOT_TRIGGERED")
'''
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "T107_A_E3_OPTIMIZED_FAIL_CLOSED_PASS" in result.stdout
