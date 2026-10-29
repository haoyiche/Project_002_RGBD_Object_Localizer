"""T1-02: audit ACTUAL BOP-scenewise RGB, uint16 depth and visible mask PNGs.

This is a data-ingestion/traceability tool, NOT a pose estimator. Uses the
already-verified T1-01 metadata contract. All XYZ/depth statistics are meters.
Run on openly downloaded data only; generated PNGs in tests are format fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from src.evaluation.t1_01_bop_contract import load_bop_frame_contract, depth_raw_to_m

SOURCE_URL = "https://bop.felk.cvut.cz/datasets/"
REFERENCE_KIND = "BOP_FORMAT_POSE_ANNOTATION_NOT_INDEPENDENT_PHYSICAL_GT"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _png(path: Path, *, role: str) -> np.ndarray:
    # IMREAD_UNCHANGED preserves 16-bit depth. OpenCV decodes RGB as BGR.
    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None:
        raise ValueError(f"PNG_DECODE_FAILED:{role}")
    return value


def read_instance(scene_dir, *, im_id: int, gt_index: int = 0):
    """Read and validate ONE ground-truth instance of ONE acquired frame."""
    c = load_bop_frame_contract(scene_dir, im_id=im_id, gt_index=gt_index)
    bgr = _png(c.rgb_path, role="rgb")
    raw = _png(c.depth_path, role="depth")
    mask_raw = _png(c.mask_visib_path, role="mask_visib")

    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError("RGB_MUST_BE_UINT8_HWC3")
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError("DEPTH_MUST_BE_UINT16_HW")
    if mask_raw.ndim != 2 or mask_raw.dtype != np.uint8:
        raise ValueError("VISIBLE_MASK_MUST_BE_UINT8_HW")
    if not np.all((mask_raw == 0) | (mask_raw == 255)):
        raise ValueError("VISIBLE_MASK_NOT_BINARY_0_255")
    if bgr.shape[:2] != raw.shape or mask_raw.shape != raw.shape:
        raise ValueError("RGB_DEPTH_MASK_SHAPE_MISMATCH")
    visible = mask_raw == 255
    if not visible.any():
        raise ValueError("EMPTY_VISIBLE_MASK")
    depth_m, valid_depth = depth_raw_to_m(raw, c)
    usable = visible & valid_depth
    if not usable.any():
        raise ValueError("NO_VALID_DEPTH_IN_VISIBLE_MASK")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return c, rgb, depth_m, visible, usable


def _frame_ids(scene_dir: Path) -> list[int]:
    """Require camera and GT image ID sets to agree; do not skip missing IDs."""
    camera = json.loads((scene_dir / "scene_camera.json").read_text(encoding="utf-8"))
    gt = json.loads((scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    if not isinstance(camera, dict) or not isinstance(gt, dict):
        raise ValueError("INVALID_SCENE_JSON_OBJECT")
    if set(camera) != set(gt):
        raise ValueError("CAMERA_GT_IMAGE_IDS_MISMATCH")
    try:
        annotated_ids = {int(k) for k in camera}
        rgb_ids = {int(p.stem) for p in (scene_dir / "rgb").glob("*.png")}
        depth_ids = {int(p.stem) for p in (scene_dir / "depth").glob("*.png")}
    except (ValueError, TypeError) as exc:
        raise ValueError("INVALID_IMAGE_ID") from exc
    if any(i < 0 for i in annotated_ids | rgb_ids | depth_ids) or len(annotated_ids) != len(camera):
        raise ValueError("INVALID_IMAGE_ID")
    # In BOP'19 subsets the scene JSON can index more images than the
    # downloaded image subset. Select actual files, never missing data.
    if rgb_ids != depth_ids:
        raise ValueError("RGB_DEPTH_IMAGE_IDS_MISMATCH")
    if not rgb_ids.issubset(annotated_ids):
        raise ValueError("PNG_IMAGE_ID_NOT_IN_ANNOTATIONS")
    return sorted(rgb_ids)


def make_manifest(scene_dir, *, dataset_id="lm", split="test", scene_id=1,
                  count=1, gt_index=0):
    """Index exactly 'count' *distinct frame IDs* from one real BOP scene.

    Missing/corrupt images are a hard failure, not a silently smaller manifest.
    No files are downloaded and no test split is used for tuning by this tool.
    """
    root = Path(scene_dir)
    if type(count) is not int or count < 1:
        raise ValueError("COUNT_MUST_BE_POSITIVE")
    if type(scene_id) is not int or scene_id < 0:
        raise ValueError("INVALID_SCENE_ID")
    if not dataset_id or not split:
        raise ValueError("DATASET_AND_SPLIT_REQUIRED")
    ids = _frame_ids(root)
    if len(ids) < count:
        raise ValueError(f"INSUFFICIENT_DISTINCT_FRAMES:available={len(ids)},required={count}")

    items = []
    seen_rgbd_content = set()
    for im_id in ids[:count]:
        c, rgb, depth_m, visible, usable = read_instance(root, im_id=im_id,
                                                          gt_index=gt_index)
        rgb_hash = sha256_file(c.rgb_path)
        depth_hash = sha256_file(c.depth_path)
        if (rgb_hash, depth_hash) in seen_rgbd_content:
            raise ValueError("DUPLICATE_RGBD_FRAME_CONTENT")
        seen_rgbd_content.add((rgb_hash, depth_hash))
        items.append({
            "sample_id": f"{dataset_id}/{split}/{scene_id:06d}/{im_id:06d}/{gt_index:06d}",
            "scene_id": scene_id,
            "im_id": im_id,
            "gt_index": gt_index,
            "obj_id": c.obj_id,
            "image_hw": list(depth_m.shape),
            "rgb_dtype": str(rgb.dtype),
            "depth_dtype_raw": "uint16",
            "depth_scale_mm_per_unit": c.depth_scale_mm_per_unit,
            "visible_mask_pixels": int(visible.sum()),
            "valid_visible_depth_pixels": int(usable.sum()),
            "visible_depth_min_m": float(depth_m[usable].min()),
            "visible_depth_max_m": float(depth_m[usable].max()),
            "K_row_major": c.K.ravel().tolist(),
            "R_cam_from_model_row_major": c.R_cam_from_model.ravel().tolist(),
            "t_cam_from_model_m": c.t_cam_from_model_m.tolist(),
            "files": {
                "rgb": {"relative": f"rgb/{im_id:06d}.png", "sha256": rgb_hash},
                "depth": {"relative": f"depth/{im_id:06d}.png", "sha256": depth_hash},
                "mask_visib": {"relative": f"mask_visib/{im_id:06d}_{gt_index:06d}.png",
                               "sha256": sha256_file(c.mask_visib_path)},
            },
        })

    return {
        "schema": "t1-02-bop-rgbd-manifest-v1",
        "dataset_id": dataset_id,
        "split": split,
        "scene_id": scene_id,
        "source_url": SOURCE_URL,
        "claimed_dataset_repository": f"bop-benchmark/{dataset_id}",
        "data_origin_authenticated": False,
        "dataset_license": "CC BY 4.0" if dataset_id == "lm" else "LICENSE_REQUIRES_VERIFICATION",
        "reference_kind": REFERENCE_KIND,
        "physical_accuracy_claimed": False,
        "pose_accuracy_claimed": False,
        "distinct_rgbd_frame_count": len(items),
        "annotation_sha256": {
            "scene_camera.json": sha256_file(root / "scene_camera.json"),
            "scene_gt.json": sha256_file(root / "scene_gt.json"),
        },
        "samples": items,
    }


def verify_manifest(manifest, scene_dir):
    """Recompute from image/annotation bytes; fail if content or order drifts."""
    root = Path(scene_dir)
    expected = make_manifest(root, dataset_id=manifest["dataset_id"],
                             split=manifest["split"], scene_id=manifest["scene_id"],
                             count=manifest["distinct_rgbd_frame_count"],
                             gt_index=manifest["samples"][0]["gt_index"])
    if expected != manifest:
        raise ValueError("MANIFEST_CONTENT_MISMATCH")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", required=True, type=Path,
                    help="e.g. data/raw/bop/lm/test/000001")
    ap.add_argument("--dataset-id", default="lm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scene-id", type=int, default=1)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--gt-index", type=int, default=0)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    manifest = make_manifest(args.scene_dir, dataset_id=args.dataset_id,
                             split=args.split, scene_id=args.scene_id,
                             count=args.count, gt_index=args.gt_index)
    verify_manifest(manifest, args.scene_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("reference_kind:", REFERENCE_KIND)
    print("distinct_rgbd_frames:", manifest["distinct_rgbd_frame_count"])
    print("first_sample:", manifest["samples"][0]["sample_id"])
    print("valid_visible_depth_pixels:", manifest["samples"][0]["valid_visible_depth_pixels"])
    print("physical_accuracy_claimed: False; pose_accuracy_claimed: False")
    print("Manifest:", args.output)
    print("T1-02 PNG / GT / manifest audit: PASS")


if __name__ == "__main__":
    main()
