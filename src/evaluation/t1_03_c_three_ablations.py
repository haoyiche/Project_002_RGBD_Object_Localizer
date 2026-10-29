"""T1-03-C: controlled geometry sensitivity on a verified real BOP RGB-D frame.

Three *in-memory* ablations: positive depth bias, fx calibration shift,
and deterministic valid-depth dropout. NEVER writes to input PNG or claims
that a visible-cloud centroid is a 6D pose estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.t1_02_bop_rgbd import read_instance, verify_manifest
from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics


def three_ablations(manifest: dict, scene_dir: Path, *, depth_bias_m=0.01,
                    fx_multiplier=1.10, dropout_stride=10) -> dict:
    """Describe three controlled perturbations to the FIRST manifest frame.

    Units: depth/XYZ in meters, fx in pixels. The BOP GT stays untouched.
    Depth dropout: zero the 10th, 20th, ... valid target pixel in row-major order.
    """
    if not np.isfinite(depth_bias_m) or depth_bias_m <= 0:
        raise ValueError("INVALID_DEPTH_BIAS")
    if not np.isfinite(fx_multiplier) or fx_multiplier <= 0:
        raise ValueError("INVALID_FX_MULTIPLIER")
    if type(dropout_stride) is not int or dropout_stride < 2:
        raise ValueError("INVALID_DROPOUT_STRIDE")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list) or not manifest["samples"]:
        raise ValueError("INVALID_MANIFEST")
    root = Path(scene_dir)
    if verify_manifest(manifest, root) is not True:
        raise ValueError("MANIFEST_VERIFICATION_FAILED")
    sample = manifest["samples"][0]
    c, rgb, depth, visible, usable = read_instance(
        root, im_id=sample["im_id"], gt_index=sample["gt_index"]
    )
    fx, fy, cx, cy = float(c.K[0, 0]), float(c.K[1, 1]), float(c.K[0, 2]), float(c.K[1, 2])

    def measure(d: np.ndarray, f: float) -> dict:
        points, _, selection = mask_to_pointcloud(d, rgb, visible, fx=f, fy=fy, cx=cx, cy=cy)
        if not np.isfinite(points).all() or points.shape[0] == 0:
            raise ValueError("INVALID_PERTURBED_POINTCLOUD")
        stats = compute_pointcloud_statistics(points)
        return {
            "point_count": int(len(points)),
            "centroid_camera_m": stats["centroid"].tolist(),
            "aabb_extent_camera_m": stats["bbox_extent"].tolist(),
            "depth_min_m": float(points[:, 2].min()),
            "depth_max_m": float(points[:, 2].max()),
            "selected_pixels": int(selection.sum()),
        }

    baseline = measure(depth, fx)
    if baseline["point_count"] != sample["valid_visible_depth_pixels"]:
        raise ValueError("MANIFEST_GEOMETRY_COUNT_MISMATCH")
    if baseline["point_count"] < dropout_stride:
        raise ValueError("INSUFFICIENT_VALID_POINTS_FOR_DROPOUT")

    # A: Add 10 mm to every *valid visible* depth in a writable in-memory copy.
    biased_depth = depth.copy()
    biased_depth[usable] += depth_bias_m
    a = measure(biased_depth, fx)

    # B: Intentionally perturb only the focal length fx; do not recalibrate GT.
    b = measure(depth, fx * fx_multiplier)

    # C: Set every dropout_stride-th valid visible pixel depth to zero, in memory.
    dropout_depth = depth.copy()
    valid_flat_indices = np.flatnonzero(usable.ravel())
    removed = valid_flat_indices[dropout_stride - 1::dropout_stride]
    dropout_depth.ravel()[removed] = 0.0
    d = measure(dropout_depth, fx)

    return {
        "schema": "t1-03-c-geometry-perturbations-v1",
        "sample_id": sample["sample_id"],
        "reference_kind": "VISIBLE_CLOUD_PERTURBATION_SENSITIVITY_NOT_POSE_ERROR",
        "unit": "m",
        "depth_bias_m": float(depth_bias_m),
        "fx_multiplier": float(fx_multiplier),
        "dropout_stride": dropout_stride,
        "dropout_removed_count": int(len(removed)),
        "baseline": baseline,
        "A_depth_bias": a,
        "B_fx_change": b,
        "C_depth_dropout": d,
        "visible_mask_pixels": int(visible.sum()),
        "gt_model_origin_camera_m": c.t_cam_from_model_m.tolist(),
        "source_input_unchanged": True,
        "physical_accuracy_claimed": False,
        "pose_accuracy_claimed": False,
        "publishable": False,
        "position_m": None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result = three_ablations(manifest, args.scene_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("baseline", "A_depth_bias", "B_fx_change", "C_depth_dropout"):
        x = result[key]
        print(f"{key}: n={x['point_count']} centroid_m={x['centroid_camera_m']} extent_m={x['aabb_extent_camera_m']}")
    print("dropout_removed_count:", result["dropout_removed_count"])
    print("T1-03-C controlled geometry sensitivity: PASS")


if __name__ == "__main__":
    main()
