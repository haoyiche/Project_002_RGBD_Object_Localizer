"""EXP032-B: controlled tabletop-depth-noise experiment for Open3D 0.20.0.

Run from P02 root: python src/experiments/exp032_ransac_noise.py
Only tabletop Z is perturbed; all other inputs are held fixed.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import open3d as o3d


# Units: meters, unless explicitly labeled mm or degrees.
DISTANCE_THRESHOLD_M = 0.01
RANSAC_N = 3
NUM_ITERATIONS = 1000
PROBABILITY = 1.0  # full iteration budget for Open3D segment_plane
NOISE_SEED = 20261013
RANSAC_SEEDS = (101, 202, 303, 404, 505)
SIGMAS_M = (0.0, 0.005, 0.020)


def build_scene() -> tuple[np.ndarray, np.ndarray]:
    """Return 9 known tabletop points and 3 fixed non-table points."""
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
    assert table.shape == (9, 3) and objects.shape == (3, 3)
    return table, objects


def fit_and_measure(points: np.ndarray, seed: int) -> dict:
    """Fit a plane and compare it against the *known* tabletop labels."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Seeds are controlled, but parallel RANSAC is not guaranteed bit-for-bit
    # deterministic on every platform/build.
    o3d.utility.random.seed(seed)
    model, raw_inliers = pcd.segment_plane(
        distance_threshold=DISTANCE_THRESHOLD_M,
        ransac_n=RANSAC_N,
        num_iterations=NUM_ITERATIONS,
        probability=PROBABILITY,
    )

    plane = np.asarray(model, dtype=np.float64)
    normal_norm = np.linalg.norm(plane[:3])
    if not np.isfinite(normal_norm) or normal_norm <= 0:
        raise RuntimeError(f"Invalid plane: {plane}")
    plane /= normal_norm
    if plane[2] < 0:  # canonical orientation for easier display
        plane *= -1

    indices = np.unique(np.asarray(raw_inliers, dtype=np.int64))
    if np.any(indices < 0) or np.any(indices >= len(points)):
        raise RuntimeError("Open3D returned invalid inlier indices")

    table_tp = int(np.count_nonzero(indices < 9))
    object_leak = int(np.count_nonzero(indices >= 9))
    precision = table_tp / len(indices) if len(indices) else 0.0
    recall = table_tp / 9.0
    angle_deg = float(np.degrees(np.arccos(
        np.clip(abs(plane[2]), 0.0, 1.0))))
    reference_error_mm = float(abs(np.dot(plane[:3], [0, 0, 1])
                                   + plane[3]) * 1000.0)
    residual_mm = np.abs(points @ plane[:3] + plane[3]) * 1000.0

    return {
        "ransac_seed": seed,
        "plane_a": float(plane[0]),
        "plane_b": float(plane[1]),
        "plane_c": float(plane[2]),
        "plane_d": float(plane[3]),
        "inliers": int(len(indices)),
        "table_tp": table_tp,
        "table_fn": 9 - table_tp,
        "object_leak": object_leak,
        "precision": precision,
        "recall": recall,
        "normal_angle_deg": angle_deg,
        "reference_error_mm": reference_error_mm,
        "max_residual_mm": float(np.max(residual_mm)),
        "table_residual_rmse_mm": float(np.sqrt(
            np.mean(np.square(residual_mm[:9])))),
        "inlier_indices": ",".join(map(str, indices.tolist())),
    }


def main() -> None:
    np.set_printoptions(precision=4, suppress=True)
    table, objects = build_scene()
    rng = np.random.default_rng(NOISE_SEED)
    standard_noise = rng.standard_normal(len(table))
    rows: list[dict] = []

    print(f"Open3D={o3d.__version__} NumPy={np.__version__}")
    print(f"threshold={DISTANCE_THRESHOLD_M * 1000:g} mm, "
          f"ransac_n={RANSAC_N}, iterations={NUM_ITERATIONS}, "
          f"probability={PROBABILITY}")
    print(f"noise_seed={NOISE_SEED}, ransac_seeds={RANSAC_SEEDS}")
    print("Standardized noise samples:", standard_noise)

    for sigma_m in SIGMAS_M:
        noisy_table = table.copy()
        noisy_table[:, 2] += sigma_m * standard_noise
        points = np.vstack((noisy_table, objects))
        sigma_mm = sigma_m * 1000

        # Independent oracle: classification against the true plane z=1,
        # WITHOUT fitting any model.
        true_dist_mm = np.abs(points[:, 2] - 1.0) * 1000.0
        oracle_table_count = int(np.count_nonzero(
            true_dist_mm[:9] <= DISTANCE_THRESHOLD_M * 1000))

        print(f"\n========== B: sigma={sigma_mm:g} mm ==========")
        print("Noisy tabletop Z (m):", noisy_table[:, 2])
        print("Table Z perturbations (mm):",
              (noisy_table[:, 2] - 1.0) * 1000)
        print("Against FIXED true plane: tabletop inliers=",
              f"{oracle_table_count}/9")
        print("seed | inliers | table TP/9 | object leak/3 | "
              "angle(deg) | ref error(mm) | table RMSE(mm)")

        for seed in RANSAC_SEEDS:
            result = fit_and_measure(points, seed)
            result.update({
                "sigma_mm": sigma_mm,
                "noise_seed": NOISE_SEED,
                "oracle_true_plane_table_inliers": oracle_table_count,
            })
            rows.append(result)
            print(f"{seed:4d} | {result['inliers']:7d} | "
                  f"{result['table_tp']:10d} | "
                  f"{result['object_leak']:13d} | "
                  f"{result['normal_angle_deg']:10.4f} | "
                  f"{result['reference_error_mm']:13.4f} | "
                  f"{result['table_residual_rmse_mm']:14.4f}")
            if seed == RANSAC_SEEDS[0]:
                print("  First fitted plane [a,b,c,d] =",
                      np.array([result[k] for k in
                                ("plane_a", "plane_b", "plane_c", "plane_d")]))
                print("  First fitted inlier indices =", result["inlier_indices"])

        group = [r for r in rows if r["sigma_mm"] == sigma_mm]
        print("Across five RANSAC seeds: table TP range =",
              f"{min(x['table_tp'] for x in group)}.."
              f"{max(x['table_tp'] for x in group)}; "
              "angle range(deg) =",
              f"{min(x['normal_angle_deg'] for x in group):.4f}.."
              f"{max(x['normal_angle_deg'] for x in group):.4f}")

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "experiments" / "exp032_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "exp032_B_noise_metrics.csv"

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nCSV saved: {output_csv}")
    print(f"Trials recorded: {len(rows)}")


if __name__ == "__main__":
    main()
