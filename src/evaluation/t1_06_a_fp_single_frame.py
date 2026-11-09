"""T1-06-A: GT-isolated FoundationPose model-based SINGLE-FRAME smoke test.

Run inside the pinned FoundationPose Docker image from its repository root.
Input root must contain exactly six whitelisted files (one BOP frame).
No scene_gt.json, GT-containing manifests, temporal tracking, or GT metrics.
Python 3.8 compatible. The official register() pose is ALREADY in the
ORIGINAL CAD model frame; never compensate for mesh centering again.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import trimesh


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gt_isolated_input(root, im_id, obj_id, gt_index):
    """Validate BOP RGB/uint16 depth/mask/CAD. Never access scene GT."""
    root = Path(root)
    _require(root.is_dir(), "INPUT_DIRECTORY_MISSING")
    _require(im_id >= 0 and obj_id > 0 and gt_index >= 0, "INVALID_IDS")
    paths = {
        "rgb": root / "rgb" / ("%06d.png" % im_id),
        "depth": root / "depth" / ("%06d.png" % im_id),
        "mask_visib": root / "mask_visib" / ("%06d_%06d.png" % (im_id, gt_index)),
        "scene_camera": root / "scene_camera.json",
        "model_mesh": root / "models" / ("obj_%06d.ply" % obj_id),
        "models_info": root / "models" / "models_info.json",
    }
    required = set(p.relative_to(root).as_posix() for p in paths.values())
    present = set(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
    _require(present == required,
             "GT_ISOLATION_OR_INPUT_SET_FAILED: missing=%s extra=%s" %
             (sorted(required - present), sorted(present - required)))

    with paths["scene_camera"].open("r", encoding="utf-8") as f:
        camera = json.load(f)
    frame = camera.get(str(im_id))
    _require(isinstance(frame, dict), "CAMERA_IMAGE_ID_NOT_FOUND")
    K = np.asarray(frame["cam_K"], dtype=np.float64).reshape(3, 3)
    depth_scale = float(frame["depth_scale"])
    _require(np.isfinite(K).all() and K[0, 0] > 0 and K[1, 1] > 0
             and np.allclose(K[2], [0, 0, 1], atol=1e-8, rtol=0)
             and np.isfinite(depth_scale) and depth_scale > 0,
             "INVALID_CAMERA_INTRINSICS_OR_DEPTH_SCALE")

    bgr = cv2.imread(str(paths["rgb"]), cv2.IMREAD_UNCHANGED)
    depth_raw = cv2.imread(str(paths["depth"]), cv2.IMREAD_UNCHANGED)
    mask_raw = cv2.imread(str(paths["mask_visib"]), cv2.IMREAD_UNCHANGED)
    _require(bgr is not None and depth_raw is not None and mask_raw is not None,
             "PNG_DECODE_FAILED")
    _require(bgr.ndim == 3 and bgr.shape[2] == 3 and bgr.dtype == np.uint8,
             "INVALID_RGB_SHAPE_OR_DTYPE")
    _require(depth_raw.ndim == 2 and depth_raw.dtype == np.uint16,
             "DEPTH_MUST_BE_UINT16")
    _require(mask_raw.ndim == 2 and mask_raw.dtype == np.uint8
             and np.isin(mask_raw, (0, 255)).all(), "INVALID_BINARY_MASK")
    _require(bgr.shape[:2] == depth_raw.shape == mask_raw.shape,
             "RGB_DEPTH_MASK_SHAPE_MISMATCH")
    visible = mask_raw > 0
    valid_depth = visible & (depth_raw > 0)
    _require(int(valid_depth.sum()) >= 30, "INSUFFICIENT_VISIBLE_DEPTH_POINTS")

    # BOP depth_scale is millimeters per raw uint16 unit; FP expects meters.
    depth_m = np.ascontiguousarray(depth_raw.astype(np.float32) * (depth_scale / 1000.0))
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mask = np.ascontiguousarray(visible.astype(np.uint8))

    with paths["models_info"].open("r", encoding="utf-8") as f:
        models_info = json.load(f)
    info = models_info.get(str(obj_id))
    _require(isinstance(info, dict), "OBJECT_NOT_IN_MODELS_INFO")
    expected_mm = np.asarray([info["size_" + k] for k in "xyz"], dtype=np.float64)
    _require(expected_mm.shape == (3,) and np.isfinite(expected_mm).all()
             and np.all(expected_mm > 0), "INVALID_MODEL_SIZE_METADATA")
    mesh = trimesh.load_mesh(str(paths["model_mesh"]), process=False)
    _require(isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) >= 30
             and len(mesh.faces) > 0, "INVALID_TRIANGULAR_CAD_MODEL")
    vertices_mm = np.asarray(mesh.vertices, dtype=np.float64)
    _require(np.isfinite(vertices_mm).all(), "INVALID_CAD_VERTICES")
    actual_mm = np.ptp(vertices_mm, axis=0)
    _require(np.allclose(actual_mm, expected_mm, rtol=0, atol=1.0),
             "MODEL_SIZE_MM_MISMATCH")
    mesh.apply_scale(0.001)
    _require(np.isfinite(np.asarray(mesh.vertex_normals)).all(),
             "INVALID_CAD_VERTEX_NORMALS")

    audit = {
        "frame_id": im_id,
        "obj_id": obj_id,
        "oracle_mask_index": gt_index,
        "image_shape_hw": list(depth_raw.shape),
        "visible_mask_pixels": int(visible.sum()),
        "visible_valid_depth_pixels": int(valid_depth.sum()),
        "median_visible_depth_m": float(np.median(depth_m[valid_depth])),
        "depth_scale_mm_per_raw_unit": depth_scale,
        "model_extent_mm": actual_mm.tolist(),
        "model_center_m": ((np.asarray(mesh.bounds[0]) + np.asarray(mesh.bounds[1])) / 2).tolist(),
        "K": K.tolist(),
        "sha256": {name: sha256_file(p) for name, p in paths.items()},
        "input_files": sorted(required),
        "oracle_mask_used": True,
        "pose_gt_available_to_predictor": False,
        "track_one_called": False,
        "prediction_uses_scene_gt": False,
    }
    return K, rgb, depth_m, mask, mesh, audit


def run(args):
    out_path = Path(args.output_json)
    _require(not out_path.exists(), "OUTPUT_ALREADY_EXISTS_NO_OVERWRITE")
    _require(out_path.parent.is_dir(), "OUTPUT_PARENT_DIRECTORY_MISSING")
    _require(1 <= args.iteration <= 20, "INVALID_REFINE_ITERATION")

    K, rgb, depth_m, mask, mesh, audit = read_gt_isolated_input(
        args.input_root, args.im_id, args.obj_id, args.gt_index
    )
    print("INPUT_PREFLIGHT_PASS", json.dumps({
        "frame_id": audit["frame_id"], "image_shape_hw": audit["image_shape_hw"],
        "valid_depth_pixels": audit["visible_valid_depth_pixels"],
        "model_extent_mm": audit["model_extent_mm"],
    }), flush=True)

    if args.preflight_only:
        print("NO_INFERENCE_PERFORMED", flush=True)
        return

    # CUDA/FP imports intentionally deferred until the complete GT-free input check.
    # The adapter is mounted outside the FoundationPose tree, so explicitly
    # add the pinned upstream repository to Python import search paths.
    import sys
    sys.path.insert(0, "/workspace/FoundationPose")
    import torch
    import nvdiffrast.torch as dr
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    refiner = PoseRefinePredictor()
    scorer = ScorePredictor()
    glctx = dr.RasterizeCudaContext()
    estimator = FoundationPose(
        model_pts=np.asarray(mesh.vertices).copy(),
        model_normals=np.asarray(mesh.vertex_normals).copy(),
        mesh=mesh,
        symmetry_tfs=None,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx,
        debug=0,
        debug_dir="/tmp/fp_t106_debug",
    )
    torch.cuda.synchronize()
    init_s = time.perf_counter() - t0
    free_pre_register, total = torch.cuda.mem_get_info()
    print("ESTIMATOR_INITIALIZED", "init_s=%.3f" % init_s,
          "free_before_register_mib=%.1f" % (free_pre_register / 1024**2),
          flush=True)

    t1 = time.perf_counter()
    # Each frame uses register(), never track_one(). Official output already
    # includes mesh-to-original-origin transformation; DO NOT apply it twice.
    pose = estimator.register(
        K=K, rgb=rgb, depth=depth_m, ob_mask=mask,
        ob_id=args.obj_id, iteration=args.iteration,
    )
    torch.cuda.synchronize()
    register_s = time.perf_counter() - t1
    _require(estimator.pose_last is not None, "REGISTER_EARLY_FALLBACK_NO_NEURAL_INFERENCE")

    pose = np.asarray(pose, dtype=np.float64)
    _require(pose.shape == (4, 4) and np.isfinite(pose).all(), "INVALID_POSE")
    _require(np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4, rtol=0),
             "POSE_HOMOGENEOUS_ROW_INVALID")
    R = pose[:3, :3]
    _require(np.allclose(R.T @ R, np.eye(3), atol=2e-3, rtol=0)
             and abs(np.linalg.det(R) - 1.0) < 2e-3, "POSE_ROTATION_NOT_SO3")
    free_post, _ = torch.cuda.mem_get_info()
    result = {
        "schema": "t1_06_a_foundationpose_single_frame_smoke_v1",
        "source_commit": args.source_commit,
        "image_digest": args.image_digest,
        "algorithm": "FoundationPose_model_based",
        "mode": "single_frame_register_not_track",
        "refine_iterations": args.iteration,
        "official_register_output_original_model_frame": True,
        "input_audit": audit,
        "pose_model_to_camera_4x4": pose.tolist(),
        "translation_m": pose[:3, 3].tolist(),
        "det_R": float(np.linalg.det(R)),
        "timing_seconds": {
            "estimator_init": init_s,
            "register_cold": register_s,
            "register_includes_possible_jit": True,
        },
        "gpu_mib": {
            "total": total / 1024**2,
            "free_before_register": free_pre_register / 1024**2,
            "free_after_register": free_post / 1024**2,
            "torch_peak_reserved": torch.cuda.max_memory_reserved() / 1024**2,
            "torch_peak_allocated": torch.cuda.max_memory_allocated() / 1024**2,
        },
        "neural_register_performed": True,
        "evaluated_against_gt": False,
        "publishable": False,
    }
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("FOUNDATIONPOSE_SINGLE_FRAME_REGISTER_PASS", flush=True)
    print("OUTPUT_JSON:", str(out_path), flush=True)
    print("REGISTER_SECONDS:", round(register_s, 3), flush=True)
    print("TORCH_PEAK_RESERVED_MIB:",
          round(result["gpu_mib"]["torch_peak_reserved"], 1), flush=True)
    print("POSE_TRANSLATION_M:", result["translation_m"], flush=True)
    print("GT_EVALUATION_PERFORMED: False", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/fp_inputs"))
    parser.add_argument("--output-json", type=Path, default=Path("/fp_outputs/pose_000055.json"))
    parser.add_argument("--im-id", type=int, default=55)
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--gt-index", type=int, default=0)
    parser.add_argument("--iteration", type=int, default=5)
    parser.add_argument("--source-commit", default="a1b694b83e633c2cb6115b9063d940a687759392")
    parser.add_argument("--image-digest", default="sha256:22837f81c0cfc1906b96daa344f8e80919c7619b816fcab487a024f1f566c215")
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
