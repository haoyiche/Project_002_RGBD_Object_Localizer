"""EXP036-B: prediction-first controlled localization-error ablations.

All positions are synthetic, not independently measured ground truth. Camera
coordinates are meters; errors and signed differences are millimeters. The
camera model assumes rectified pixels and optical-axis depth. Run as:
    python -m src.experiments.exp036_error_ablation_b

Do not interpret diagnostic results as real-device accuracy or robot commands.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.exp036_error_minimal import (
    BASE_FRAME,
    CAMERA_FRAME,
    REFERENCE_KIND,
    deproject_pixel_m,
    position_error_mm,
    summarize_errors_mm,
)

INTRINSICS_PX = {
    "fx_px": 600.0,
    "fy_px": 600.0,
    "cx_px": 320.0,
    "cy_px": 240.0,
}
PIXEL_UV = (380.0, 270.0)
REFERENCE_CAMERA_M = np.array([0.10, 0.05, 1.00], dtype=np.float64)


def _xyz(x: np.ndarray) -> list[float]:
    """Convert finite (3,) meter-coordinate vector for strict JSON."""
    a = np.asarray(x, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("diagnostic XYZ must be finite with shape (3,)")
    return [float(v) for v in a]


def depth_symmetry_b1() -> dict:
    """Same pixel, only optical Z varies: +/-10 mm, plus a fixed +10 mm case."""
    kw = dict(INTRINSICS_PX, u_px=PIXEL_UV[0], v_px=PIXEL_UV[1])
    p_ref = deproject_pixel_m(**kw, z_m=1.0)
    np.testing.assert_allclose(p_ref, REFERENCE_CAMERA_M, rtol=0, atol=1e-12)

    records = []
    delta_vectors_mm = []
    distance_errors_mm = []
    for label, z_m in (("plus_10mm", 1.01), ("minus_10mm", 0.99)):
        p_est = deproject_pixel_m(**kw, z_m=z_m)
        delta_mm = 1000.0 * (p_est - p_ref)
        err = position_error_mm(p_est, p_ref)
        records.append({
            "perturbation": label,
            "optical_z_m": z_m,
            "estimate_m": _xyz(p_est),
            "signed_delta_mm": _xyz(delta_mm),
            "euclidean_error_mm": err,
        })
        delta_vectors_mm.append(delta_mm)
        distance_errors_mm.append(err)

    mean_signed = np.mean(np.stack(delta_vectors_mm, axis=0), axis=0)
    mean_abs = float(np.mean(distance_errors_mm))
    # Independent hand-computed oracle, not inferred from the implementation.
    np.testing.assert_allclose(delta_vectors_mm[0], [1.0, 0.5, 10.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(delta_vectors_mm[1], [-1.0, -0.5, -10.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(mean_signed, [0.0, 0.0, 0.0], rtol=0, atol=1e-10)
    np.testing.assert_allclose(distance_errors_mm, [np.sqrt(101.25)] * 2,
                               rtol=0, atol=1e-10)
    print("B1 symmetrical +/-10mm depth: PASS | mean_signed(mm) =",
          _xyz(mean_signed), "| mean_euclidean(mm) =", round(mean_abs, 6))

    return {
        "frame_id": CAMERA_FRAME,
        "pixel_uv": list(PIXEL_UV),
        "synthetic_gt_m": _xyz(p_ref),
        "samples": records,
        "mean_signed_delta_mm": _xyz(mean_signed),
        "mean_euclidean_error_mm": mean_abs,
        "fixed_plus_bias_illustration": {
            "repeats": 2,
            "signed_delta_mean_mm": _xyz(delta_vectors_mm[0]),
            "mean_euclidean_error_mm": distance_errors_mm[0],
            "note": "Repeated deterministic offset is not a noise-distribution estimate",
        },
        "note": "Zero mean signed error is not zero mean Euclidean error",
    }


def focal_spatial_b2() -> dict:
    """Constant fx+5% only, test three image positions at identical depth."""
    samples = []
    for label, u_px in (("P1", 320.0), ("P2", 380.0), ("P3", 500.0)):
        kw = dict(INTRINSICS_PX, u_px=u_px, v_px=270.0, z_m=1.0)
        truth = deproject_pixel_m(**kw)
        wrong = deproject_pixel_m(**dict(kw, fx_px=630.0))
        delta_mm = (wrong - truth) * 1000.0
        err_mm = position_error_mm(wrong, truth)
        samples.append({
            "sample_id": label,
            "u_px": u_px,
            "synthetic_gt_m": _xyz(truth),
            "estimate_m": _xyz(wrong),
            "signed_delta_mm": _xyz(delta_mm),
            "euclidean_error_mm": err_mm,
        })
    # Independent analytic values: 60*(1/630-1/600) m = -1/210 m.
    np.testing.assert_allclose([s["signed_delta_mm"][0] for s in samples],
                               [0.0, -100.0 / 21.0, -100.0 / 7.0],
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose([s["signed_delta_mm"][1:] for s in samples],
                               np.zeros((3, 2)), rtol=0, atol=1e-12)
    np.testing.assert_allclose([s["euclidean_error_mm"] for s in samples],
                               [0.0, 100.0 / 21.0, 100.0 / 7.0],
                               rtol=0, atol=1e-10)
    print("B2 fx+5% spatial dependence: PASS | errors(mm) =",
          [round(s["euclidean_error_mm"], 6) for s in samples])
    return {
        "frame_id": CAMERA_FRAME,
        "fx_correct_px": 600.0,
        "fx_wrong_px": 630.0,
        "fy_px": 600.0,
        "optical_z_m": 1.0,
        "samples": samples,
        "note": "Center gives zero error for this fx-only perturbation, NOT for every calibration error",
    }


def translation_bias_b3() -> dict:
    """Three different spatial points, fixed translation bias, NOT repeats."""
    camera_points_m = np.array([
        [0.10, 0.05, 1.00],
        [-0.20, 0.10, 0.80],
        [0.30, -0.10, 1.20],
    ], dtype=np.float64)
    assert camera_points_m.shape == (3, 3)
    t_correct = np.array([0.30, -0.10, 0.20], dtype=np.float64)
    t_wrong = np.array([0.31, -0.10, 0.20], dtype=np.float64)
    R = np.eye(3, dtype=np.float64)
    base_truth = camera_points_m @ R.T + t_correct
    base_wrong = camera_points_m @ R.T + t_wrong
    delta_mm = (base_wrong - base_truth) * 1000.0
    errors = [position_error_mm(wrong, truth) for wrong, truth in zip(base_wrong, base_truth)]
    stats = summarize_errors_mm(errors)

    np.testing.assert_allclose(base_truth,
                               [[0.40, -0.05, 1.20], [0.10, 0.0, 1.00], [0.60, -0.20, 1.40]],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(base_wrong,
                               [[0.41, -0.05, 1.20], [0.11, 0.0, 1.00], [0.61, -0.20, 1.40]],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(delta_mm, [[10.0, 0.0, 0.0]] * 3,
                               rtol=0, atol=1e-10)
    for key in ("mean_mm", "median_mm", "p95_mm", "max_mm"):
        np.testing.assert_allclose(stats[key], 10.0, rtol=0, atol=1e-10)
    print("B3 fixed +10mm translation bias: PASS | errors(mm) =",
          [round(e, 6) for e in errors], "| mean/median/max=10mm")
    records = []
    for i in range(3):
        records.append({
            "sample_id": f"P{i+1}",
            "camera_m": _xyz(camera_points_m[i]),
            "synthetic_gt_base_m": _xyz(base_truth[i]),
            "estimate_base_m": _xyz(base_wrong[i]),
            "signed_delta_mm": _xyz(delta_mm[i]),
            "euclidean_error_mm": errors[i],
        })
    return {
        "frame_id": BASE_FRAME,
        "source_frame_id": CAMERA_FRAME,
        "rotation": "identity synthetic example",
        "t_correct_m": _xyz(t_correct),
        "t_wrong_m": _xyz(t_wrong),
        "sample_kind": "different_spatial_positions_not_repeat_measurements",
        "samples": records,
        "statistics": stats,
        "note": "Identical fixed bias across targets does not measure repeatability",
    }


def run(project_root: Path) -> dict:
    """Run B1/B2/B3 without modifying EXP036-A or downstream pipeline."""
    report = {
        "experiment": "EXP036-B",
        "data_kind": "synthetic_controlled_experiment",
        "reference_kind": REFERENCE_KIND,
        "calibration_verified": False,
        "publishable": False,
        "published_positions": [],
        "units": {"pixel_intrinsics": "px", "xyz_translation_depth": "m",
                  "signed_difference_and_error": "mm"},
        "assumptions": [
            "Rectified pinhole camera, optical-axis depth Z",
            "Every estimate and reference compared in matching frames",
            "Independent deterministic controlled perturbations, no random sampling",
        ],
        "B1": depth_symmetry_b1(),
        "B2": focal_spatial_b2(),
        "B3": translation_bias_b3(),
        "warning": "SYNTHETIC_GT ONLY: not a claim about real-device accuracy or repeatability",
    }
    path = Path(project_root) / "results" / "exp036" / "exp036_B_error_ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    print("Saved:", path)
    print("EXP036-B CONTROLLED ABLATIONS VERIFIED; EXP036 UNIT COMPLETION: PENDING")
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
