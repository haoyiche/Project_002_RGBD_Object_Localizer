"""T1-07-A E3: paired Mask/Depth sensitivity on fixed BOP RGB-D frames.

V2: critical checks use explicit exceptions and survive ``python -O``.
The V1 report is immutable; V2 writes a new report with a new source hash.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.t1_02_bop_rgbd import read_instance, verify_manifest
from src.geometry.mask_to_pointcloud import mask_to_pointcloud


SCENE = Path("data/raw/bop/lm/test/000001")
MANIFEST_PATH = Path("data/manifests/t1_02_lm_50.json")
OUTPUT_PATH = Path("outputs/t1_07_a/e3_mask_depth_real50_v2.json")
VARIANT_NAMES = ("erode", "dilate", "mask_strip", "depth_strip")


def require(condition, code):
    """Explicit fail-closed check; never compiled away by ``python -O``."""
    if not condition:
        raise ValueError(code)


def evaluate_frame(rgb, depth, mask, K):
    """Compare paired geometry changes; no GT or 6D pose is used here."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    mask = np.asarray(mask, dtype=bool)
    K = np.asarray(K, dtype=np.float64)

    require(depth.ndim == 2, "INVALID_DEPTH_SHAPE")
    require(mask.shape == depth.shape, "INVALID_MASK_SHAPE")
    require(rgb.shape == (*depth.shape, 3), "INVALID_RGB_SHAPE")
    require(K.shape == (3, 3) and np.isfinite(K).all(), "INVALID_INTRINSICS")

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    require(fx > 0 and fy > 0, "INVALID_FOCAL_LENGTH")

    valid_depth = np.isfinite(depth) & (depth > 0)
    usable = mask & valid_depth
    require(bool(usable.any()), "EMPTY_BASELINE")

    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)

    columns = np.nonzero(usable)[1]
    span = int(columns.max() - columns.min() + 1)
    width = max(1, int(round(span * 0.10)))
    center = int(np.median(columns))
    left = max(0, min(depth.shape[1] - width, center - width // 2))

    band = np.zeros(mask.shape, dtype=bool)
    band[:, left:left + width] = True

    cut_mask = mask.copy()
    cut_mask[band] = False
    cut_depth = depth.copy()
    cut_depth[band] = 0.0

    def cloud(selected_mask, selected_depth):
        points, colors, selected = mask_to_pointcloud(
            selected_depth, rgb, selected_mask,
            fx=fx, fy=fy, cx=cx, cy=cy,
        )
        points = np.asarray(points)
        colors = np.asarray(colors)
        selected = np.asarray(selected)
        require(points.ndim == 2 and points.shape[1] == 3, "INVALID_CLOUD_SHAPE")
        require(points.shape[0] > 0, "EMPTY_PERTURBED_CLOUD")
        require(colors.shape == points.shape, "INVALID_CLOUD_COLORS")
        require(selected.shape == depth.shape, "INVALID_SELECTED_SHAPE")
        require(np.isfinite(points).all(), "NONFINITE_CLOUD_POINTS")
        require(
            int(np.count_nonzero(selected)) == len(points),
            "POINT_SELECTION_COUNT_MISMATCH",
        )
        return points, colors, selected

    baseline_points, _, baseline_selected = cloud(mask, depth)
    require(np.array_equal(baseline_selected, usable), "BASELINE_SELECTION_MISMATCH")
    require(len(baseline_points) == int(np.count_nonzero(usable)), "BASELINE_POINT_MISMATCH")

    variants = {
        "erode": (eroded, depth),
        "dilate": (dilated, depth),
        "mask_strip": (cut_mask, depth),
        "depth_strip": (mask, cut_depth),
    }
    result = {
        "baseline_points": int(len(baseline_points)),
        "baseline_mask_pixels": int(mask.sum()),
        "baseline_valid_depth_pixels": int(valid_depth.sum()),
        "strip_target_pixels_removed": int(np.count_nonzero(usable & band)),
        "strip_left": int(left),
        "strip_width": int(width),
        "variants": {},
    }
    clouds = {}

    for name, (variant_mask, variant_depth) in variants.items():
        points, colors, selected = cloud(variant_mask, variant_depth)
        expected_selected = variant_mask & np.isfinite(variant_depth) & (variant_depth > 0)
        require(np.array_equal(selected, expected_selected), "VARIANT_SELECTION_MISMATCH")
        clouds[name] = (points, colors, selected)
        shift_mm = float(
            np.linalg.norm(points.mean(axis=0) - baseline_points.mean(axis=0))
            * 1000.0
        )
        result["variants"][name] = {
            "mask_pixels": int(variant_mask.sum()),
            "point_count": int(len(points)),
            "retention_fraction": float(len(points) / len(baseline_points)),
            "centroid_shift_mm": shift_mm,
        }

    require(len(clouds["erode"][0]) <= len(baseline_points), "ERODE_COUNT_INCREASED")
    require(len(clouds["dilate"][0]) >= len(baseline_points), "DILATE_COUNT_DECREASED")
    require(result["strip_target_pixels_removed"] > 0, "EMPTY_STRIP_REMOVAL")

    for a, b in zip(clouds["mask_strip"], clouds["depth_strip"]):
        require(np.array_equal(a, b), "STRIP_OUTPUT_MISMATCH")

    result["strip_outputs_identical"] = True
    result["mask_strip_source"] = "MASK_REMOVAL"
    result["depth_strip_source"] = "DEPTH_ZEROED"
    return result


def main():
    raw_manifest = MANIFEST_PATH.read_bytes()
    manifest = json.loads(raw_manifest)
    require(isinstance(manifest.get("samples"), list), "INVALID_MANIFEST_SAMPLES")
    require(len(manifest["samples"]) == 50, "MANIFEST_NOT_50_FRAMES")
    require(verify_manifest(manifest, SCENE) is True, "MANIFEST_VERIFICATION_FAILED")

    frames = []
    for sample in manifest["samples"]:
        c, rgb, depth, mask, _ = read_instance(
            SCENE, im_id=sample["im_id"], gt_index=sample["gt_index"]
        )
        result = evaluate_frame(rgb, depth, mask, c.K)
        require(
            result["baseline_points"] == sample["valid_visible_depth_pixels"],
            "MANIFEST_POINT_COUNT_MISMATCH",
        )
        frames.append({
            "sample_id": sample["sample_id"],
            "im_id": int(sample["im_id"]),
            **result,
        })

    require(len(frames) == 50, "FRAME_COUNT_MISMATCH")
    require(len({f["sample_id"] for f in frames}) == 50, "DUPLICATE_SAMPLE_ID")
    require(len({f["im_id"] for f in frames}) == 50, "DUPLICATE_IMAGE_ID")
    require(all(f["strip_outputs_identical"] for f in frames), "STRIP_FRAME_MISMATCH")

    summary = {}
    for name in VARIANT_NAMES:
        shifts = np.asarray([
            frame["variants"][name]["centroid_shift_mm"] for frame in frames
        ])
        summary[name] = {
            "mean_centroid_shift_mm": float(shifts.mean()),
            "median_centroid_shift_mm": float(np.median(shifts)),
            "p95_centroid_shift_mm": float(np.percentile(shifts, 95, method="linear")),
            "max_centroid_shift_mm": float(shifts.max()),
            "total_points": int(sum(
                frame["variants"][name]["point_count"] for frame in frames
            )),
        }

    report = {
        "schema": "t1-07-a-e3-mask-depth-real50-v2",
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "frame_count": len(frames),
        "strip_equivalent_frames": sum(f["strip_outputs_identical"] for f in frames),
        "summary": summary,
        "frames": frames,
        "reference_kind": "CONTROLLED_GEOMETRY_SENSITIVITY_NOT_POSE_ACCURACY",
        "simulated_occlusion_not_physical_capture": True,
        "physical_accuracy_claimed": False,
        "pose_accuracy_claimed": False,
        "publishable": False,
        "supersedes": "e3_mask_depth_real50.json",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")

    print(json.dumps(summary, indent=2))
    print("STRIP_EQUIVALENT_FRAMES:", report["strip_equivalent_frames"])
    print("SOURCE_SHA256:", report["source_sha256"])
    print("OUTPUT:", OUTPUT_PATH)
    print("T107_A_E3_V2_REAL50_PASS")


if __name__ == "__main__":
    main()
