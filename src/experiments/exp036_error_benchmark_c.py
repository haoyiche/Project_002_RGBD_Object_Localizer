"""EXP036-C: deterministic 20-sample synthetic localization error benchmark.

Run from P02 project root:
    python -m src.experiments.exp036_error_benchmark_c

Synthetic references and injected errors are NOT real RGB-D accuracy measurements.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.experiments.exp036_error_minimal import position_error_mm, summarize_errors_mm
from src.geometry.localization_error_metrics import evaluate_synthetic_samples

FRAME_ID = "camera_optical_frame"
SIGNED_X_MM = (1, -2, 3, -4, 5, -2, 4, -6, 8, -10,
               10, -10, 10, -10, 10, -3, 4, -5, 6, -50)

# The following are independent arithmetic oracles from the learner's C1-C4.
ORACLE_SAMPLES = {
    "S00": ([0.10, 0.05, 1.00], [0.101, 0.05, 1.00], 1, 1),
    "S01": ([0.11, 0.05, 1.00], [0.108, 0.05, 1.00], -2, 2),
    "S19": ([0.14, 0.08, 1.00], [0.09, 0.08, 1.00], -50, 50),
}
ORACLE_FULL = {"count": 20, "mean_mm": 8.15, "median_mm": 5.5,
               "p95_mm": 12.0, "max_mm": 50.0}
ORACLE_WITHOUT_S19 = {"count": 19, "mean_mm": 113.0 / 19.0,
                      "median_mm": 5.0, "p95_mm": 10.0, "max_mm": 10.0}


def make_deterministic_samples() -> list[dict]:
    """Make 20 unique spatial samples, all within the same camera frame.

    No random number generator is used. S19 is deliberately injected at -50 mm.
    """
    records = []
    for i, signed_x_mm in enumerate(SIGNED_X_MM):
        gt = [0.10 + 0.01 * (i % 5), 0.05 + 0.01 * (i // 5), 1.00]
        estimate = [gt[0] + signed_x_mm / 1000.0, gt[1], gt[2]]
        records.append({
            "sample_id": f"S{i:02d}",
            "frame_id": FRAME_ID,
            "gt_frame_id": FRAME_ID,
            "estimate_frame_id": FRAME_ID,
            "reference_kind": "synthetic_gt",
            "gt_unit": "m",
            "estimate_unit": "m",
            "synthetic_gt_m": gt,
            "estimate_m": estimate,
            "injected_outlier": i == 19,
        })
    return records


def _check_oracle(actual: dict, expected: dict, label: str) -> None:
    for key, oracle in expected.items():
        np.testing.assert_allclose(actual[key], oracle, rtol=0, atol=1e-10,
                                   err_msg=f"{label}: {key}")


def run(project_root: Path) -> dict:
    samples = make_deterministic_samples()
    assert len(samples) == 20
    full = evaluate_synthetic_samples(samples, expected_frame_id=FRAME_ID)
    assert full["valid_count"] == 20 and full["rejected_count"] == 0

    # C1: independently predicted sample positions and errors.
    by_id = {sample["sample_id"]: sample for sample in full["valid_samples"]}
    for sample_id, (gt, estimate, signed_x, error) in ORACLE_SAMPLES.items():
        row = by_id[sample_id]
        np.testing.assert_allclose(row["synthetic_gt_m"], gt, rtol=0, atol=1e-12)
        np.testing.assert_allclose(row["estimate_m"], estimate, rtol=0, atol=1e-12)
        np.testing.assert_allclose(row["signed_delta_x_mm"], signed_x, rtol=0, atol=1e-10)
        np.testing.assert_allclose(row["euclidean_error_mm"], error, rtol=0, atol=1e-10)
    assert full["shapes"]["gt_points"] == [20, 3]
    assert full["shapes"]["signed_delta_xyz_mm"] == [20, 3]
    assert full["shapes"]["signed_delta_x_mm"] == [20]
    print("C1 per-sample hand oracles and shape contracts: PASS | S00/S01/S19")

    # C2: cross-check with the A-stage statistical implementation and hand truth.
    errors = [row["euclidean_error_mm"] for row in full["valid_samples"]]
    original_stats = summarize_errors_mm(errors)
    _check_oracle(full["statistics"], ORACLE_FULL, "C2 FULL hand oracle")
    _check_oracle(original_stats, ORACLE_FULL, "C2 A-stage independent statistics")
    _check_oracle(full["statistics"], {k: original_stats[k] for k in ORACLE_FULL},
                  "C2 cross-implementation stats")
    for row in full["valid_samples"]:
        external_error = position_error_mm(row["estimate_m"], row["synthetic_gt_m"])
        np.testing.assert_allclose(row["euclidean_error_mm"], external_error,
                                   rtol=0, atol=1e-10)
    print("C2 20-sample statistics: PASS | mean=8.15, median=5.5, P95=12, max=50 mm")

    # C3: signed x error (-41/20) differs from mean nonnegative distance.
    np.testing.assert_allclose(full["signed_delta_x_mean_mm"], -41.0 / 20.0,
                               rtol=0, atol=1e-10)
    np.testing.assert_allclose(full["mean_signed_delta_xyz_mm"], [-2.05, 0.0, 0.0],
                               rtol=0, atol=1e-10)
    assert abs(full["signed_delta_x_mean_mm"]) < full["statistics"]["mean_mm"]
    print("C3 signed x bias: PASS | -2.05 mm vs mean Euclidean 8.15 mm")

    # C4: the exclusion is a separate sensitivity study, never official stats.
    sensitivity_records = [row for row in samples if row["sample_id"] != "S19"]
    sensitivity = evaluate_synthetic_samples(sensitivity_records,
                                              expected_frame_id=FRAME_ID)
    _check_oracle(sensitivity["statistics"], ORACLE_WITHOUT_S19,
                  "C4 EXCLUDE S19 hand oracle")
    assert by_id["S19"]["injected_outlier"] is True
    assert "S19" in by_id
    print("C4 S19 sensitivity: PASS | n=19 mean=113/19, median=5, P95=10, max=10 mm")

    # C5: invalid samples are diagnosed and excluded, NEVER assigned 0 mm.
    wrong_frame = dict(samples[0], sample_id="FRAME_BAD", estimate_frame_id="robot_base")
    no_gt = dict(samples[0], sample_id="GT_MISSING", synthetic_gt_m=None)
    diagnosis = evaluate_synthetic_samples([samples[0], wrong_frame, no_gt],
                                            expected_frame_id=FRAME_ID)
    assert diagnosis["valid_count"] == 1 and diagnosis["rejected_count"] == 2
    assert {row["reason"] for row in diagnosis["rejected_samples"]} == {
        "FRAME_MISMATCH", "MISSING_GROUND_TRUTH"
    }
    assert diagnosis["statistics"]["count"] == 1
    print("C5 wrong-frame / missing-GT excluded with diagnostics: PASS")

    report = {
        "experiment": "EXP036-C",
        "data_kind": "synthetic_deterministic",
        "reference_kind": "synthetic_gt",
        "rng_used": False,
        "random_seed": None,
        "frame_id": FRAME_ID,
        "units": {"xyz": "m", "signed_delta_and_error": "mm"},
        "injected_model": "X-direction signed offsets (mm) from FIXED 20-entry list",
        "signed_x_injection_mm": list(SIGNED_X_MM),
        "full_20_sample_benchmark": full,
        "sensitivity_without_S19": {
            "excluded_sample_id": "S19",
            "purpose": "outlier sensitivity only; not a replacement for full statistics",
            "benchmark": sensitivity,
        },
        "invalid_sample_diagnostics_demo": {
            "note": "Frame mismatch and missing GT are excluded; neither gets zero error",
            "valid_count": diagnosis["valid_count"],
            "rejected_count": diagnosis["rejected_count"],
            "rejected_samples": diagnosis["rejected_samples"],
        },
        "calibration_verified": False,
        "publishable": False,
        "published_positions": [],
        "warning": "SYNTHETIC_GT ONLY. n=20 and synthetic P95 do not establish real-device accuracy or tail reliability.",
    }
    assert report["publishable"] is False and report["published_positions"] == []
    output = Path(project_root) / "results" / "exp036" / "exp036_C_error_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("Saved:", output)
    print("EXP036-C BENCHMARK VERIFIED; EXP036 UNIT COMPLETION: PENDING")
    return report


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
