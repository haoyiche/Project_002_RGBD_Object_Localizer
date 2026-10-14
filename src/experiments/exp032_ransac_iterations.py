"""EXP032-C: RANSAC iteration-budget experiment.

Only num_iterations changes: 1, 10, 100, 1000.
The 20 mm tabletop noise is identical to EXP032-B (noise seed 20261013).
Run from P02 root:
    python src/experiments/exp032_ransac_iterations.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import open3d as o3d


NOISE_SEED = 20261013
SIGMA_M = 0.020
DISTANCE_THRESHOLD_M = 0.010
RANSAC_N = 3
PROBABILITY = 1.0
ITERATION_BUDGETS = (1, 10, 100, 1000)
RANSAC_SEEDS = (101, 202, 303, 404, 505)


def build_b2_scene() -> tuple[np.ndarray, np.ndarray]:
    """Recreate EXP032-B's *same* 9 noisy table + 3 fixed object points."""
    table = np.array(
        [[x, y, 1.0] for y in (0.0, 0.1, 0.2)
         for x in (0.0, 0.1, 0.2)],
        dtype=np.float64,
    )
    objects = np.array(
        [[0.05, 0.04, 1.15],
         [0.14, 0.17, 1.20],
         [0.22, 0.08, 1.25]],
        dtype=np.float64,
    )
    standard_noise = np.random.default_rng(NOISE_SEED).standard_normal(9)
    table[:, 2] += SIGMA_M * standard_noise
    return np.vstack((table, objects)), standard_noise


def evaluate(points: np.ndarray, budget: int, seed: int) -> dict:
    """Fit RANSAC and score against ground-truth table membership / plane."""
    row = {
        "iterations": budget,
        "ransac_seed": seed,
        "status": "OK",
        "inliers": "",
        "table_tp": "",
        "object_leak": "",
        "table_precision": "",
        "table_recall": "",
        "normal_angle_deg": "",
        "reference_error_mm": "",
        "table_rmse_mm": "",
        "plane_a": "",
        "plane_b": "",
        "plane_c": "",
        "plane_d": "",
        "inlier_indices": "",
        "error": "",
    }
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.utility.random.seed(seed)
        model, indices_raw = pcd.segment_plane(
            distance_threshold=DISTANCE_THRESHOLD_M,
            ransac_n=RANSAC_N,
            num_iterations=budget,
            probability=PROBABILITY,
        )
        plane = np.asarray(model, dtype=np.float64).copy()
        norm = float(np.linalg.norm(plane[:3]))
        if plane.shape != (4,) or not np.isfinite(plane).all() or norm <= 1e-12:
            raise ValueError(f"Invalid fitted plane: {plane}")

        plane /= norm
        if plane[2] < 0:
            plane *= -1  # canonical sign for output only

        indices = np.unique(np.asarray(indices_raw, dtype=np.int64))
        if (indices.size and
                (np.any(indices < 0) or np.any(indices >= points.shape[0]))):
            raise ValueError("Invalid returned inlier indices")
        table_tp = int(np.count_nonzero(indices < 9))
        object_leak = int(np.count_nonzero(indices >= 9))
        residual = np.abs(points @ plane[:3] + plane[3])
        row.update({
            "inliers": len(indices),
            "table_tp": table_tp,
            "object_leak": object_leak,
            "table_precision": table_tp / len(indices) if len(indices) else 0.0,
            "table_recall": table_tp / 9.0,
            "normal_angle_deg": float(np.degrees(np.arccos(
                np.clip(abs(plane[2]), 0.0, 1.0)))),
            "reference_error_mm": float(abs(
                np.dot(plane[:3], [0.0, 0.0, 1.0]) + plane[3]) * 1000),
            "table_rmse_mm": float(np.sqrt(
                np.mean(residual[:9] ** 2)) * 1000),
            "plane_a": float(plane[0]),
            "plane_b": float(plane[1]),
            "plane_c": float(plane[2]),
            "plane_d": float(plane[3]),
            "inlier_indices": ",".join(map(str, indices.tolist())),
        })
    except (RuntimeError, ValueError) as exc:
        # One iteration may yield a degenerate candidate, depending on sampling.
        # Record failures instead of pretending every trial returned a valid plane.
        row["status"] = "FAILED"
        row["error"] = str(exc)
    return row


def main() -> None:
    np.set_printoptions(precision=5, suppress=True)
    points, noise = build_b2_scene()
    oracle_count = int(np.count_nonzero(
        np.abs(points[:9, 2] - 1.0) <= DISTANCE_THRESHOLD_M
    ))
    print(f"Open3D={o3d.__version__}, NumPy={np.__version__}")
    print(f"B2 fixed scene: noise_seed={NOISE_SEED}, sigma={SIGMA_M*1000:g} mm")
    print("Standardized noise:", noise)
    print("Tabletop Z noise (mm):", noise * SIGMA_M * 1000)
    print(f"Threshold={DISTANCE_THRESHOLD_M*1000:g} mm; "
          f"ransac_n={RANSAC_N}; probability={PROBABILITY}")
    print(f"Fixed true-plane table inliers: {oracle_count}/9")
    print("RANSAC seeds:", RANSAC_SEEDS)

    rows = []
    for budget in ITERATION_BUDGETS:
        print(f"\n======== iteration budget = {budget} ========")
        print(" seed | status | inliers | table TP | object leak | "
              "angle(deg) | ref err(mm)")
        for seed in RANSAC_SEEDS:
            result = evaluate(points, budget, seed)
            rows.append(result)
            if result["status"] == "OK":
                print(f" {seed:4d} |   OK   | {result['inliers']:7d} | "
                      f"{result['table_tp']:8d} | {result['object_leak']:11d} | "
                      f"{result['normal_angle_deg']:10.4f} | "
                      f"{result['reference_error_mm']:11.4f}")
                if seed == RANSAC_SEEDS[0]:
                    print("  First plane:",
                          [round(result[f"plane_{k}"], 5) for k in "abcd"])
                    print("  First inliers:", result["inlier_indices"])
            else:
                print(f" {seed:4d} | FAILED | {result['error']}")
        valid = [r for r in rows if r["iterations"] == budget
                 and r["status"] == "OK"]
        if valid:
            print("  Valid trials:", len(valid), "/", len(RANSAC_SEEDS))
            print("  Inliers range:", min(r["inliers"] for r in valid),
                  "..", max(r["inliers"] for r in valid))
            print("  Angle range(deg):",
                  f"{min(r['normal_angle_deg'] for r in valid):.4f}.."
                  f"{max(r['normal_angle_deg'] for r in valid):.4f}")

    root = Path(__file__).resolve().parents[2]
    result_path = root / "experiments" / "exp032_results" / "exp032_C_iterations.csv"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved: {result_path}")
    print(f"Trials recorded: {len(rows)} (including failed trials)")
    print("Interpretation: ONE fixed B2 noise sample; this is not a general "
          "noise-robustness or failure-probability estimate.")


if __name__ == "__main__":
    main()
