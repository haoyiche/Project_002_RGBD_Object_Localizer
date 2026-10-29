"""T1-03-B: 50-frame real BOP visible-cloud geometry baseline (CPU only).

Consumes a T1-02 manifest and the associated RGB/Depth/visible-Mask PNGs.
This is a *diagnostic*, not a 3D-position accuracy or 6D-pose estimator.
All XYZ values are in camera coordinates (meters). No robot publication.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.evaluation.t1_02_bop_rgbd import read_instance, sha256_file, verify_manifest
from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics


def analyze_manifest_geometry(manifest: dict, scene_dir: Path) -> dict:
    """Verify the saved manifest, then compute geometry for *every* listed frame.

    Fail-closed: an exception prevents a partially successful report being
    treated as a completed 50-frame audit. No GT-origin/centroid error metric.
    """
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError("INVALID_MANIFEST")
    samples = manifest["samples"]
    if not samples or manifest.get("distinct_rgbd_frame_count") != len(samples):
        raise ValueError("MANIFEST_COUNT_MISMATCH")
    root = Path(scene_dir)
    if verify_manifest(manifest, root) is not True:
        raise ValueError("MANIFEST_VERIFICATION_FAILED")

    rows = []
    for sample in samples:
        c, rgb, depth_m, visible, usable = read_instance(
            root, im_id=sample["im_id"], gt_index=sample["gt_index"]
        )
        K = c.K
        points, colors, selected = mask_to_pointcloud(
            depth_m, rgb, visible,
            fx=float(K[0, 0]), fy=float(K[1, 1]),
            cx=float(K[0, 2]), cy=float(K[1, 2]),
        )
        if (not np.array_equal(selected, usable)
                or len(points) != sample["valid_visible_depth_pixels"]
                or int(visible.sum()) != sample["visible_mask_pixels"]
                or colors.shape != points.shape):
            raise ValueError("VISIBLE_GEOMETRY_INPUT_MISMATCH")
        stats = compute_pointcloud_statistics(points)
        if not np.isclose(float(points[:, 2].min()), sample["visible_depth_min_m"], atol=1e-12):
            raise ValueError("DEPTH_RANGE_MISMATCH")
        if not np.isclose(float(points[:, 2].max()), sample["visible_depth_max_m"], atol=1e-12):
            raise ValueError("DEPTH_RANGE_MISMATCH")
        visible_count = int(visible.sum())
        valid_count = len(points)
        rows.append({
            "sample_id": sample["sample_id"],
            "scene_id": sample["scene_id"],
            "im_id": sample["im_id"],
            "gt_index": sample["gt_index"],
            "obj_id": c.obj_id,
            "visible_mask_pixels": visible_count,
            "valid_visible_depth_pixels": valid_count,
            "invalid_visible_depth_pixels": visible_count - valid_count,
            "valid_visible_depth_ratio": valid_count / visible_count,
            "centroid_camera_m": stats["centroid"].tolist(),
            "bbox_center_camera_m": stats["bbox_center"].tolist(),
            "bbox_min_camera_m": stats["min_bound"].tolist(),
            "bbox_max_camera_m": stats["max_bound"].tolist(),
            "bbox_extent_camera_m": stats["bbox_extent"].tolist(),
            "depth_min_m": float(points[:, 2].min()),
            "depth_max_m": float(points[:, 2].max()),
            # Annotation is for provenance, *not* a centroid-error reference.
            "gt_model_origin_camera_m": c.t_cam_from_model_m.tolist(),
        })

    ratios = np.asarray([s["valid_visible_depth_ratio"] for s in rows], dtype=np.float64)
    z_extents = np.asarray([s["bbox_extent_camera_m"][2] for s in rows], dtype=np.float64)
    return {
        "schema": "t1-03-a-visible-cloud-geometry-v1",
        "dataset_id": manifest["dataset_id"],
        "split": manifest["split"],
        "scene_id": manifest["scene_id"],
        "reference_kind": "BOP_VISIBLE_CLOUD_GEOMETRY_DIAGNOSTIC",
        "gt_translation_semantics": "MODEL_ORIGIN_NOT_VISIBLE_CLOUD_CENTROID",
        "units": "m",
        "coordinate_frame": "camera_optical_frame",
        "frame_count": len(rows),
        "valid_visible_depth_ratio_mean": float(ratios.mean()),
        "valid_visible_depth_ratio_min": float(ratios.min()),
        "bbox_extent_z_m_median": float(np.median(z_extents)),
        "bbox_extent_z_m_max": float(z_extents.max()),
        "physical_accuracy_claimed": False,
        "pose_accuracy_claimed": False,
        "publishable": False,
        "position_m": None,
        "samples": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = analyze_manifest_geometry(manifest, args.scene_dir)
    report["source_manifest_sha256"] = sha256_file(args.manifest)
    # Write only after every image/GT/hash/geometry check has passed.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print("Real BOP frame count:", report["frame_count"])
    print("First centroid camera (m):", report["samples"][0]["centroid_camera_m"])
    print("First AABB extent camera (m):", report["samples"][0]["bbox_extent_camera_m"])
    print("Mean valid visible depth ratio:", report["valid_visible_depth_ratio_mean"])
    print("Physical/Pose accuracy claimed: False/False")
    print("publishable: False; position_m: None")
    print("Report:", args.output)
    print("T1-03-B real geometry batch: PASS")


if __name__ == "__main__":
    main()
