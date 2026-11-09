"""T1-06-A: deterministic GT-blind fixed-N FoundationPose batch on Windows.

Usage (from project root):
  python -m src.evaluation.t1_06_a_fp_fixed_batch plan --count 50
  python -m src.evaluation.t1_06_a_fp_fixed_batch predict --count 3 --run-dir outputs/t1_06_a/fp_pilot3_v1
  python -m src.evaluation.t1_06_a_fp_fixed_batch predict --count 50 --run-dir outputs/t1_06_a/fp_fixed50_v1
  python -m src.evaluation.t1_06_a_fp_fixed_batch resume --count 50 --run-dir outputs/t1_06_a/fp_fixed50_v1
  python -m src.evaluation.t1_06_a_fp_fixed_batch audit --count 50 --run-dir outputs/t1_06_a/fp_fixed50_v1

Prediction never reads scene_gt.json or a GT-bearing manifest. Every Docker
process receives ONLY six GT-free input files, plus pinned upstream code,
weights, an adapter and a JIT cache. Each frame uses independent register(),
never tracking. Failures stop the batch; partial batches cannot be reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from src.evaluation import t1_04_b4_fixed_50_batch as b4

select_rgbd_frames = b4.select_rgbd_frames
from src.evaluation import t1_06_a_fp_posthoc_eval as audit

SOURCE_COMMIT = "a1b694b83e633c2cb6115b9063d940a687759392"
IMAGE_DIGEST = "sha256:22837f81c0cfc1906b96daa344f8e80919c7619b816fcab487a024f1f566c215"
PLAN_SCHEMA = "t1-06-a-fp-gt-blind-fixed-selection-v1"
BATCH_SCHEMA = "t1-06-a-fp-all-predictions-frozen-v1"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return audit.sha256_file(path)


def new_json(path, payload):
    path = Path(path)
    require(path.parent.is_dir(), "OUTPUT_PARENT_MISSING:" + str(path.parent))
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def get_paths(args, im_id):
    stem = f"{im_id:06d}"
    expected = audit.expected_inputs(im_id, args.obj_id, args.gt_index)
    return expected, {
        "rgb": args.scene_dir / expected["rgb"],
        "depth": args.scene_dir / expected["depth"],
        "mask_visib": args.scene_dir / expected["mask_visib"],
        "scene_camera": args.scene_dir / "scene_camera.json",
        "model_mesh": args.models_dir / expected["model_mesh"].split("/")[-1],
        "models_info": args.models_dir / "models_info.json",
    }


def make_plan(args):
    # This is the same GT-free selection function used by T1-04-B4.
    require(args.count in (3, 50), "ONLY_PILOT3_OR_FIXED50_ALLOWED")
    ids = select_rgbd_frames(args.scene_dir, count=args.count, gt_index=args.gt_index)
    require(ids == sorted(set(ids)) and len(ids) == args.count, "FRAME_SELECTION_INVALID")
    require(args.adapter.is_file(), "ADAPTER_MISSING")
    require(args.posthoc.is_file(), "POSTHOC_FREEZE_SCRIPT_MISSING")
    required_weights = [
        "weights/2023-10-28-18-33-37/config.yml",
        "weights/2023-10-28-18-33-37/model_best.pth",
        "weights/2024-01-11-20-02-45/config.yml",
        "weights/2024-01-11-20-02-45/model_best.pth",
    ]
    weights_hashes = {}
    for rel in required_weights:
        file = args.fp_root / rel
        require(file.is_file() and not file.is_symlink(), "OFFICIAL_WEIGHT_MISSING:" + rel)
        weights_hashes[rel] = sha(file)
    frame_hashes = {}
    for im_id in ids:
        _, sources = get_paths(args, im_id)
        for name, path in sources.items():
            require(path.is_file() and not path.is_symlink(), f"SOURCE_INPUT_MISSING_OR_LINK:{im_id}:{name}")
        frame_hashes[str(im_id)] = {key: sha(path) for key, path in sources.items()}
    return {
        "schema": PLAN_SCHEMA,
        "count": args.count,
        "frame_ids": ids,
        "obj_id": args.obj_id,
        "gt_index": args.gt_index,
        "refine_iterations": args.iteration,
        "source_commit": SOURCE_COMMIT,
        "image_digest": IMAGE_DIGEST,
        "adapter_sha256": sha(args.adapter),
        "posthoc_sha256": sha(args.posthoc),
        "b4_frame_selection_source_sha256": sha(Path(b4.__file__)),
        "model_weight_files_sha256": weights_hashes,
        "frame_input_sha256": frame_hashes,
        "selection_policy": "T1-04-B4_select_rgbd_frames_first_N_sorted_PNG_ids",
        "oracle_mask_used": True,
        "gt_read_during_prediction": False,
        "complete": False,
        "publishable": False,
    }


def verify_runtime(args):
    require(args.fp_root.is_dir(), "FOUNDATIONPOSE_ROOT_MISSING")
    require(args.cache_root.is_dir(), "JIT_CACHE_MISSING")
    try:
        status = subprocess.check_output(
            ["git", "-C", str(args.fp_root), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        ).strip()
        require(not status, "PINNED_UPSTREAM_SOURCE_HAS_TRACKED_MODIFICATIONS")
        commit = subprocess.check_output(
            ["git", "-C", str(args.fp_root), "rev-parse", "HEAD"], text=True
        ).strip()
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError("RUNTIME_IDENTITY_CHECK_FAILED") from e
    require(commit == SOURCE_COMMIT, "SOURCE_COMMIT_UNPINNED:" + commit)
    require(image_id == IMAGE_DIGEST, "IMAGE_ID_CHANGED:" + image_id)
    require((args.fp_root / "mycpp" / "build").is_dir(), "MYCPP_BUILD_DIR_MISSING")


def verify_selection(plan, expected):
    require(plan == expected, "FROZEN_SELECTION_OR_SOURCE_CHANGED")


def prepare_bundle(args, im_id, root, expected_hashes):
    name = f"frame_{im_id:06d}"
    target = root / "input_bundles" / name
    expected, sources = get_paths(args, im_id)
    if not target.exists():
        temp = Path(tempfile.mkdtemp(prefix=".bundle_", dir=target.parent))
        try:
            for key, src in sources.items():
                dst = temp / expected[key]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
            audit.check_bundle(temp, expected, expected_hashes)
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    audit.check_bundle(target, expected, expected_hashes)
    return target


def verify_frame(args, im_id, bundle, frame_dir, expected_hashes):
    stem = f"{im_id:06d}"
    candidate_path = frame_dir / f"pose_{stem}.json"
    freeze_path = frame_dir / "frozen_prediction.json"
    require(candidate_path.is_file() and freeze_path.is_file(), f"FRAME_NOT_FROZEN:{im_id}")
    candidate = audit.read_json(candidate_path)
    inputs, hashes, _est, ids = audit.validate_candidate(candidate)
    require(ids == (im_id, args.obj_id, args.gt_index), f"FRAME_ID_MISMATCH:{im_id}")
    require(hashes == expected_hashes, f"FRAME_INPUT_FINGERPRINT_MISMATCH:{im_id}")
    require(candidate.get("source_commit") == SOURCE_COMMIT, "SOURCE_COMMIT_LABEL_MISMATCH")
    require(candidate.get("image_digest") == IMAGE_DIGEST, "IMAGE_DIGEST_LABEL_MISMATCH")
    require(candidate.get("refine_iterations") == args.iteration, "ITERATION_MISMATCH")
    audit.check_bundle(bundle, inputs, hashes)
    frozen = audit.read_json(freeze_path)
    require(frozen.get("candidate_sha256") == sha(candidate_path), f"CANDIDATE_HASH_MISMATCH:{im_id}")
    require(frozen.get("input_sha256") == hashes, f"FREEZE_INPUT_HASH_MISMATCH:{im_id}")
    require(frozen.get("gt_read_during_freeze") is False, "GT_FREEZE_FLAG_INVALID")
    return {
        "frame_id": im_id,
        "candidate_file": f"frames/{stem}/pose_{stem}.json",
        "candidate_sha256": sha(candidate_path),
        "freeze_file": f"frames/{stem}/frozen_prediction.json",
        "freeze_sha256": sha(freeze_path),
        "register_seconds": candidate["timing_seconds"]["register_cold"],
        "torch_peak_reserved_mib": candidate["gpu_mib"]["torch_peak_reserved"],
    }


def run_frame(args, im_id, bundle, frame_dir, log_path):
    stem = f"{im_id:06d}"
    output = f"/fp_outputs/pose_{stem}.json"
    cmd = [
        "docker", "run", "--rm", "--gpus", "all",
        "--mount", f"type=bind,source={args.fp_root},target=/workspace/FoundationPose,readonly",
        "--mount", f"type=bind,source={bundle},target=/fp_inputs,readonly",
        "--mount", f"type=bind,source={args.adapter},target=/fp_adapter/t1_06_a_fp_single_frame.py,readonly",
        "--mount", f"type=bind,source={frame_dir},target=/fp_outputs",
        "--mount", f"type=bind,source={args.cache_root},target=/fp_build/torch_extensions",
        "--workdir", "/workspace/FoundationPose",
        "-e", "MAX_JOBS=1", "-e", "TORCH_CUDA_ARCH_LIST=7.5",
        "-e", "TORCH_EXTENSIONS_DIR=/fp_build/torch_extensions",
        "--entrypoint", "python", args.image,
        "/fp_adapter/t1_06_a_fp_single_frame.py",
        "--input-root", "/fp_inputs", "--output-json", output,
        "--im-id", str(im_id), "--obj-id", str(args.obj_id),
        "--gt-index", str(args.gt_index), "--iteration", str(args.iteration),
        "--source-commit", SOURCE_COMMIT, "--image-digest", IMAGE_DIGEST,
    ]
    print(f"FRAME_START id={im_id} log={log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            raise RuntimeError(f"DOCKER_INVOCATION_FAILED:{im_id}:{exc}") from exc
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print("\n".join(tail), flush=True)
        raise RuntimeError(f"FOUNDATIONPOSE_REGISTER_FAILED:{im_id}:exit={result.returncode}:log={log_path}")
    require((frame_dir / f"pose_{stem}.json").is_file(), f"POSE_OUTPUT_NOT_CREATED:{im_id}")


def ensure_run(args, expected):
    root = args.run_dir
    plan_path = root / "selection_manifest.json"
    if args.phase == "predict":
        require(not root.exists(), "RUN_DIR_EXISTS_USE_RESUME_OR_NEW_NAME")
        root.mkdir(parents=True)
        (root / "input_bundles").mkdir()
        (root / "frames").mkdir()
        new_json(plan_path, expected)
    else:
        require(plan_path.is_file(), "RUN_NOT_INITIALIZED")
        verify_selection(audit.read_json(plan_path), expected)
    return root


def collect(args, plan, *, allow_partial=False):
    root = args.run_dir
    rows = []
    for im_id in plan["frame_ids"]:
        bundle = root / "input_bundles" / f"frame_{im_id:06d}"
        frame_dir = root / "frames" / f"{im_id:06d}"
        if not (frame_dir / f"pose_{im_id:06d}.json").exists() and allow_partial:
            continue
        if not (frame_dir / "frozen_prediction.json").exists() and allow_partial:
            continue
        rows.append(verify_frame(args, im_id, bundle, frame_dir, plan["frame_input_sha256"][str(im_id)]))
    if not allow_partial:
        require(len(rows) == plan["count"], "BATCH_INCOMPLETE")
    return rows


def predict(args, plan):
    verify_runtime(args)
    root = ensure_run(args, plan)
    final_path = root / "frozen_predictions.json"
    require(not final_path.exists(), "BATCH_ALREADY_FROZEN_USE_AUDIT")
    for idx, im_id in enumerate(plan["frame_ids"]):
        frame_dir = root / "frames" / f"{im_id:06d}"
        hashes = plan["frame_input_sha256"][str(im_id)]
        try:
            bundle = prepare_bundle(args, im_id, root, hashes)
            frame_dir.mkdir(exist_ok=True)
            pose = frame_dir / f"pose_{im_id:06d}.json"
            frozen = frame_dir / "frozen_prediction.json"
            if frozen.is_file():
                verify_frame(args, im_id, bundle, frame_dir, hashes)
                print(f"FRAME_VERIFIED_ALREADY_FROZEN {idx+1}/{args.count} id={im_id}", flush=True)
                continue
            if not pose.exists():
                log_path = frame_dir / "register.log"
                if log_path.exists():
                    existing_logs = list(frame_dir.glob("register_retry_*.log"))
                    log_path = frame_dir / f"register_retry_{len(existing_logs)+1:02d}.log"
                run_frame(args, im_id, bundle, frame_dir, log_path)
            # Freeze must precede ANY GT use. Uses only candidate and GT-free bundle.
            audit.freeze(pose, bundle, frozen)
            verify_frame(args, im_id, bundle, frame_dir, hashes)
            print(f"FRAME_FROZEN {idx+1}/{args.count} id={im_id}", flush=True)
        except Exception as exc:
            event = {
                "status": "PARTIAL_BATCH_NOT_REPORTABLE",
                "frame_id": im_id,
                "completed_before_failure": idx,
                "error": repr(exc),
                "gt_evaluation_performed": False,
            }
            new_json(root / f"failure_{im_id:06d}_{len(list(root.glob('failure_*.json')))+1:02d}.json", event)
            print(f"BATCH_STOPPED_AT_FRAME:{im_id} reason={exc}", flush=True)
            raise
    rows = collect(args, plan)
    payload = {
        "schema": BATCH_SCHEMA,
        "status": "ALL_FRAMES_PREDICTED_AND_FROZEN_GT_BLIND",
        "frame_count": len(rows),
        "selection_manifest_sha256": sha(root / "selection_manifest.json"),
        "source_commit": SOURCE_COMMIT,
        "image_digest": IMAGE_DIGEST,
        "candidates": rows,
        "gt_evaluated": False,
        "publishable": False,
        "prior_failure_event_count": len(list(root.glob("failure_*.json"))),
    }
    new_json(final_path, payload)
    print("T106_FP_FIXED_BATCH_ALL_FROZEN_PASS", "count=", len(rows), "manifest=", final_path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["plan", "predict", "resume", "audit"])
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--obj-id", type=int, default=1)
    p.add_argument("--gt-index", type=int, default=0)
    p.add_argument("--iteration", type=int, default=5)
    p.add_argument("--scene-dir", type=Path, default=Path("data/raw/bop/lm/test/000001"))
    p.add_argument("--models-dir", type=Path, default=Path("data/raw/bop/lm/models"))
    p.add_argument("--adapter", type=Path, default=Path("src/evaluation/t1_06_a_fp_single_frame.py"))
    p.add_argument("--posthoc", type=Path, default=Path("src/evaluation/t1_06_a_fp_posthoc_eval.py"))
    p.add_argument("--fp-root", type=Path, default=Path(r"D:\AI_Lab\_external\FoundationPose"))
    p.add_argument("--cache-root", type=Path, default=Path(r"D:\AI_Lab\_external\fp_torch_extensions"))
    p.add_argument("--image", default="foundationpose:t106")
    p.add_argument("--run-dir", type=Path, default=Path("outputs/t1_06_a/fp_fixed50_v1"))
    args = p.parse_args()
    require(args.obj_id == 1 and args.gt_index == 0 and args.iteration == 5,
            "FIXED_PROTOCOL_ONLY_OBJ1_MASK0_ITER5")
    for label in ("scene_dir", "models_dir", "adapter", "posthoc", "fp_root", "cache_root", "run_dir"):
        setattr(args, label, getattr(args, label).resolve())
    plan = make_plan(args)
    if args.phase == "plan":
        print("T106_FP_FIXED_BATCH_PLAN_PASS")
        print("COUNT:", plan["count"])
        print("FRAME_IDS:", plan["frame_ids"])
        print("SCENE_GT_READ: False")
        print("SELECTION_POLICY:", plan["selection_policy"])
        return
    if args.phase in ("predict", "resume"):
        predict(args, plan)
        return
    root = args.run_dir
    manifest = audit.read_json(root / "selection_manifest.json")
    verify_selection(manifest, plan)
    rows = collect(args, plan)
    frozen = audit.read_json(root / "frozen_predictions.json")
    require(frozen.get("schema") == BATCH_SCHEMA, "INVALID_BATCH_FREEZE_SCHEMA")
    require(frozen.get("selection_manifest_sha256") == sha(root / "selection_manifest.json"),
            "SELECTION_MANIFEST_HASH_MISMATCH")
    require(frozen.get("candidates") == rows and frozen.get("frame_count") == args.count,
            "BATCH_FROZEN_HASHES_MISMATCH")
    print("T106_FP_FIXED_BATCH_AUDIT_PASS", "count=", len(rows))
    print("GT_EVALUATED: False")


if __name__ == "__main__":
    main()
