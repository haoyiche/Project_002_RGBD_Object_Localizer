"""EXP036-A: prediction-first controlled RGB-D 3D localization error demo.

Units: image coordinates/intrinsics in pixels, optical Z and XYZ in meters,
error magnitudes in millimeters. All references are synthetic, NOT measured GT.
Run from P02 root: python -m src.experiments.exp036_error_minimal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CAMERA_FRAME = "camera_optical_frame"
BASE_FRAME = "robot_base"
REFERENCE_KIND = "synthetic_gt"

# Independent hand-computed oracles from EXP036-A predictions.
HAND_CAMERA_M = np.array([0.10, 0.05, 1.00], dtype=np.float64)
HAND_DEPTH_NEW_M = np.array([0.102, 0.051, 1.020], dtype=np.float64)
HAND_DEPTH_DELTA_MM = np.array([2.0, 1.0, 20.0], dtype=np.float64)
HAND_FX_NEW_M = np.array([60.0 / 630.0, 0.05, 1.0], dtype=np.float64)
HAND_BASE_CORRECT_M = np.array([0.40, -0.05, 1.20], dtype=np.float64)
HAND_BASE_WRONG_M = np.array([0.41, -0.05, 1.20], dtype=np.float64)
HAND_STATISTICS_MM = {"count": 5, "mean_mm": 6.0, "median_mm": 4.0,
                      "p95_mm": 15.6, "max_mm": 18.0}


def _finite_scalar(value, name: str) -> float:
    """Reject bool, strings, arrays, NaN, infinity, and non-scalar values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vec3_m(value, name: str) -> np.ndarray:
    """Return an independent float64 (3,) position expressed in meters."""
    raw = np.asarray(value, dtype=object)
    if raw.shape != (3,):
        raise ValueError(f"{name} must have shape (3,) in meters")
    # Inspect elements before NumPy coerces mixed [1, True, 3] into integer [1, 1, 3].
    return np.array([_finite_scalar(v, name) for v in raw], dtype=np.float64)


def deproject_pixel_m(*, u_px, v_px, z_m, fx_px, fy_px, cx_px, cy_px) -> np.ndarray:
    """Pixel + optical-axis depth -> camera optical XYZ in m, shape (3,).

    Pinhole model, rectified/undistorted pixels, z_m is optical Z (not range).
    Callers must already have verified and normalized all physical units.
    """
    u = _finite_scalar(u_px, "u_px")
    v = _finite_scalar(v_px, "v_px")
    z = _finite_scalar(z_m, "z_m")
    fx = _finite_scalar(fx_px, "fx_px")
    fy = _finite_scalar(fy_px, "fy_px")
    cx = _finite_scalar(cx_px, "cx_px")
    cy = _finite_scalar(cy_px, "cy_px")
    if z <= 0.0 or fx <= 0.0 or fy <= 0.0:
        raise ValueError("optical depth z_m and focal lengths must be positive")
    p = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)
    if not np.isfinite(p).all():
        raise ValueError("deprojection output is non-finite")
    return p


def position_error_mm(estimate_m, synthetic_gt_m) -> float:
    """3D Euclidean error between positions in SAME frame, both in meters."""
    estimate = _vec3_m(estimate_m, "estimate_m")
    reference = _vec3_m(synthetic_gt_m, "synthetic_gt_m")
    # The caller must ensure the two frames and time instants match.
    error_mm = float(np.linalg.norm((estimate - reference) * 1000.0))
    if not np.isfinite(error_mm):
        raise ValueError("3D position error is non-finite")
    return error_mm


def summarize_errors_mm(errors_mm) -> dict:
    """Mean/median/P95/max using NumPy linear percentile; sample count explicit."""
    raw = np.asarray(errors_mm, dtype=object)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("errors_mm must be nonempty 1D numeric errors in mm")
    a = np.array([_finite_scalar(v, "errors_mm") for v in raw], dtype=np.float64)
    if (a < 0).any():
        raise ValueError("errors_mm must be non-negative")
    return {
        "count": int(a.size),
        "mean_mm": float(np.mean(a)),
        "median_mm": float(np.median(a)),
        "p95_mm": float(np.percentile(a, 95, method="linear")),
        "max_mm": float(np.max(a)),
        "percentile_method": "linear",
    }


def run(project_root: Path) -> dict:
    """Verify predictions independently, then save a synthetic-only JSON report."""
    kw = dict(u_px=380.0, v_px=270.0, fx_px=600.0,
              fy_px=600.0, cx_px=320.0, cy_px=240.0)
    p_ref = deproject_pixel_m(**kw, z_m=1.0)
    np.testing.assert_allclose(p_ref, HAND_CAMERA_M, rtol=0, atol=1e-12)
    assert p_ref.shape == (3,) and p_ref.dtype == np.float64
    print("A1 hand oracle / camera-frame shape: PASS |", p_ref.tolist())

    p_depth = deproject_pixel_m(**kw, z_m=1.02)
    delta_depth_mm = (p_depth - p_ref) * 1000.0
    depth_err_mm = position_error_mm(p_depth, p_ref)
    np.testing.assert_allclose(p_depth, HAND_DEPTH_NEW_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta_depth_mm, HAND_DEPTH_DELTA_MM, rtol=0, atol=1e-10)
    np.testing.assert_allclose(depth_err_mm, np.sqrt(405.0), rtol=0, atol=1e-10)
    print("A2 depth +20mm: PASS | delta(mm) =", delta_depth_mm.tolist(),
          "| error(mm) =", round(depth_err_mm, 6))

    kw_fx = {**kw, "fx_px": 630.0}
    p_fx = deproject_pixel_m(**kw_fx, z_m=1.0)
    delta_fx_mm = (p_fx - p_ref) * 1000.0
    np.testing.assert_allclose(p_fx, HAND_FX_NEW_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta_fx_mm, [-100.0 / 21.0, 0.0, 0.0],
                               rtol=0, atol=1e-10)
    print("A3 fx +5%: PASS | delta(mm) =", delta_fx_mm.tolist())

    t_correct_m = np.array([0.30, -0.10, 0.20], dtype=np.float64)
    t_wrong_m = np.array([0.31, -0.10, 0.20], dtype=np.float64)
    p_base_correct = p_ref + t_correct_m   # R=I, synthetic camera->base
    p_base_wrong = p_ref + t_wrong_m
    delta_base_mm = (p_base_wrong - p_base_correct) * 1000.0
    translation_error_mm = position_error_mm(p_base_wrong, p_base_correct)
    np.testing.assert_allclose(p_base_correct, HAND_BASE_CORRECT_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_base_wrong, HAND_BASE_WRONG_M, rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta_base_mm, [10.0, 0.0, 0.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(translation_error_mm, 10.0, rtol=0, atol=1e-10)
    print("A4 synthetic extrinsic +10mm X: PASS | error(mm) =",
          round(translation_error_mm, 6))

    errors_mm = np.array([0.0, 2.0, 4.0, 6.0, 18.0], dtype=np.float64)
    stats = summarize_errors_mm(errors_mm)
    for key, expected in HAND_STATISTICS_MM.items():
        np.testing.assert_allclose(stats[key], expected, rtol=0, atol=1e-12)
    print("A5 stats: PASS | mean=6, median=4, P95=15.6, max=18 (mm; n=5)")

    report = {
        "experiment": "EXP036-A",
        "data_kind": "synthetic_controlled_experiment",
        "reference_kind": REFERENCE_KIND,
        "calibration_verified": False,
        "publishable": False,
        "units": {"pixel": "px", "depth_xyz_translation": "m", "error": "mm"},
        "assumptions": ["rectified pinhole camera", "Z means optical-axis depth, not ray distance",
                        "reference and estimate are compared in the same frame"],
        "A1": {"frame_id": CAMERA_FRAME, "pixel_uv": [380.0, 270.0],
               "intrinsics_px": {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0},
               "synthetic_gt_m": p_ref.tolist(), "shape": list(p_ref.shape)},
        "A2": {"frame_id": CAMERA_FRAME, "depth_new_m": 1.02,
               "estimate_m": p_depth.tolist(), "delta_mm": delta_depth_mm.tolist(),
               "error_mm": depth_err_mm},
        "A3": {"frame_id": CAMERA_FRAME, "fx_new_px": 630.0,
               "estimate_m": p_fx.tolist(), "delta_mm": delta_fx_mm.tolist(),
               "error_mm": position_error_mm(p_fx, p_ref)},
        "A4": {"frame_id": BASE_FRAME, "rotation": "identity (teaching assumption)",
               "t_correct_m": t_correct_m.tolist(), "t_wrong_m": t_wrong_m.tolist(),
               "synthetic_gt_m": p_base_correct.tolist(),
               "estimate_m": p_base_wrong.tolist(), "delta_mm": delta_base_mm.tolist(),
               "error_mm": translation_error_mm},
        "A5": {"sample_count": 5, "errors_mm": errors_mm.tolist(),
               "statistics": stats, "warning": "n=5 is NOT sufficient to characterize real P95 risk"},
        "warning": "SYNTHETIC_GT ONLY; no real-camera precision or robot publication claim",
    }
    path = Path(project_root) / "results/exp036/exp036_A_error_minimal.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print("Saved:", path)
    print("EXP036-A NUMERICAL EXPERIMENT VERIFIED; EXP036 UNIT COMPLETION: PENDING")
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
