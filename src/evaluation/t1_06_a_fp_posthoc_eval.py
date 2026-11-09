"""T1-06-A: freeze GT-blind FoundationPose output, then evaluate it posthoc.

Two explicit phases:
  freeze     -- candidate + six-file GT-isolated input bundle; NEVER touches pose GT.
  evaluate   -- verifies frozen candidate + all original input hashes BEFORE scene_gt.

CPU-only on the user's original ai_3d environment. Uses the existing
T1-04-A raw, non-symmetry-aware pose metric contract. This is NOT a BOP
benchmark or independently established physical accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation.t1_04_a_pose_metrics import PoseM2C, compare_pose_m2c

CANDIDATE_SCHEMA = "t1_06_a_foundationpose_single_frame_smoke_v1"
FREEZE_SCHEMA = "t1_06_a_fp_frozen_single_frame_v1"
EVAL_SCHEMA = "t1_06_a_fp_posthoc_annotation_eval_v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    require(path.is_file(), "FILE_NOT_FOUND:" + str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, dict), "JSON_ROOT_MUST_BE_OBJECT")
    return data


def expected_inputs(im_id, obj_id, gt_index):
    name = f"{im_id:06d}"
    return {
        "rgb": f"rgb/{name}.png",
        "depth": f"depth/{name}.png",
        "mask_visib": f"mask_visib/{name}_{gt_index:06d}.png",
        "scene_camera": "scene_camera.json",
        "model_mesh": f"models/obj_{obj_id:06d}.ply",
        "models_info": "models/models_info.json",
    }


def validate_candidate(candidate):
    require(candidate.get("schema") == CANDIDATE_SCHEMA, "CANDIDATE_SCHEMA_MISMATCH")
    require(candidate.get("algorithm") == "FoundationPose_model_based", "ALGORITHM_MISMATCH")
    require(candidate.get("mode") == "single_frame_register_not_track", "TEMPORAL_MODE_INVALID")
    require(candidate.get("neural_register_performed") is True, "NEURAL_REGISTER_NOT_DONE")
    require(candidate.get("evaluated_against_gt") is False, "CANDIDATE_PREVIOUSLY_EVALUATED")
    require(candidate.get("publishable") is False, "UNSAFE_PUBLICATION_FLAG")
    require(candidate.get("official_register_output_original_model_frame") is True,
            "POSE_FRAME_UNVERIFIED")
    require(type(candidate.get("refine_iterations")) is int
            and 1 <= candidate["refine_iterations"] <= 20, "INVALID_ITERATIONS")
    audit = candidate.get("input_audit")
    require(isinstance(audit, dict), "INPUT_AUDIT_MISSING")
    for flag, expected in (
        ("oracle_mask_used", True),
        ("pose_gt_available_to_predictor", False),
        ("track_one_called", False),
        ("prediction_uses_scene_gt", False),
    ):
        require(audit.get(flag) is expected, "GT_OR_TEMPORAL_CONTRACT_MISMATCH:" + flag)
    im_id = audit.get("frame_id")
    obj_id = audit.get("obj_id")
    gt_index = audit.get("oracle_mask_index")
    require(type(im_id) is int and im_id >= 0
            and type(obj_id) is int and obj_id > 0
            and type(gt_index) is int and gt_index >= 0, "INVALID_SAMPLE_IDENTITY")
    inputs = expected_inputs(im_id, obj_id, gt_index)
    require(audit.get("input_files") == sorted(inputs.values()), "INPUT_WHITELIST_MISMATCH")
    hashes = audit.get("sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(inputs), "INPUT_HASH_SET_MISMATCH")
    require(all(isinstance(v, str) and len(v) == 64 for v in hashes.values()),
            "INPUT_HASH_FORMAT_INVALID")
    T = np.asarray(candidate.get("pose_model_to_camera_4x4"), dtype=np.float64)
    require(T.shape == (4, 4) and np.isfinite(T).all(), "INVALID_POSE_MATRIX")
    require(np.allclose(T[3], [0, 0, 0, 1], atol=1e-5, rtol=0),
            "INVALID_POSE_BOTTOM_ROW")
    t = np.asarray(candidate.get("translation_m"), dtype=np.float64)
    require(t.shape == (3,) and np.allclose(t, T[:3, 3], atol=1e-7, rtol=0),
            "TRANSLATION_FIELD_MISMATCH")
    estimate = PoseM2C(T[:3, :3], T[:3, 3])
    return inputs, hashes, estimate, (im_id, obj_id, gt_index)


def check_bundle(bundle, inputs, hashes):
    bundle = Path(bundle)
    require(bundle.is_dir(), "GT_ISOLATED_INPUT_BUNDLE_MISSING")
    found = {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()}
    require(found == set(inputs.values()), "GT_ISOLATION_OR_BUNDLE_FILE_SET_FAILED")
    for name, rel in inputs.items():
        p = bundle / rel
        require(not p.is_symlink(), "SYMLINK_IN_BUNDLE:" + rel)
        require(sha256_file(p) == hashes[name], "BUNDLE_HASH_MISMATCH:" + name)


def _save_new_json(output, value):
    output = Path(output)
    require(output.parent.is_dir(), "OUTPUT_PARENT_MISSING")
    # 'x' prevents silently overwriting previously frozen evidence.
    with output.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def freeze(candidate_path, bundle, output):
    candidate_path = Path(candidate_path)
    require(candidate_path.is_file(), "CANDIDATE_NOT_FOUND")
    candidate = read_json(candidate_path)
    inputs, hashes, _, ids = validate_candidate(candidate)
    check_bundle(bundle, inputs, hashes)
    im_id, obj_id, gt_index = ids
    record = {
        "schema": FREEZE_SCHEMA,
        "candidate_file": candidate_path.name,
        "candidate_sha256": sha256_file(candidate_path),
        "frame_id": im_id,
        "obj_id": obj_id,
        "oracle_mask_index": gt_index,
        "input_sha256": hashes,
        "source_commit_label": candidate.get("source_commit"),
        "image_digest_label": candidate.get("image_digest"),
        "refine_iterations": candidate["refine_iterations"],
        "neural_register_performed": True,
        "oracle_mask_used": True,
        "gt_read_during_freeze": False,
        "gt_available_to_inference_container": False,
        "prediction_uses_scene_gt": False,
        "candidate_posthoc_reselected": False,
        "publishable": False,
        "note": "Provenance labels are adapter metadata; they are not runtime cryptographic verification of source or weights.",
    }
    _save_new_json(output, record)
    print("T106_FP_PREDICTION_FROZEN_PASS")
    print("CANDIDATE_SHA256:", record["candidate_sha256"])
    print("FROZEN_MANIFEST:", output)
    print("GT_READ_DURING_FREEZE: False")
    return record


def evaluate(candidate_path, frozen_path, scene_dir, models_dir, output):
    # 1) Validate frozen output BEFORE looking at scene GT.
    candidate_path = Path(candidate_path)
    frozen = read_json(frozen_path)
    require(frozen.get("schema") == FREEZE_SCHEMA, "FREEZE_SCHEMA_MISMATCH")
    require(frozen.get("candidate_file") == candidate_path.name, "FROZEN_CANDIDATE_NAME_MISMATCH")
    require(sha256_file(candidate_path) == frozen.get("candidate_sha256"),
            "FROZEN_CANDIDATE_HASH_MISMATCH")
    candidate = read_json(candidate_path)
    inputs, hashes, estimate, ids = validate_candidate(candidate)
    im_id, obj_id, gt_index = ids
    require(frozen.get("frame_id") == im_id and frozen.get("obj_id") == obj_id
            and frozen.get("oracle_mask_index") == gt_index, "FROZEN_SAMPLE_IDENTITY_MISMATCH")
    require(frozen.get("input_sha256") == hashes, "FROZEN_INPUT_HASH_MISMATCH")
    require(frozen.get("gt_read_during_freeze") is False
            and frozen.get("candidate_posthoc_reselected") is False, "FREEZE_CONTRACT_INVALID")
    scene_dir, models_dir = Path(scene_dir), Path(models_dir)
    original = {
        "rgb": scene_dir / inputs["rgb"],
        "depth": scene_dir / inputs["depth"],
        "mask_visib": scene_dir / inputs["mask_visib"],
        "scene_camera": scene_dir / inputs["scene_camera"],
        "model_mesh": models_dir / f"obj_{obj_id:06d}.ply",
        "models_info": models_dir / "models_info.json",
    }
    for name, p in original.items():
        require(p.is_file(), "ORIGINAL_INPUT_MISSING:" + name)
        require(sha256_file(p) == hashes[name], "ORIGINAL_INPUT_HASH_MISMATCH:" + name)

    # 2) Only the posthoc evaluator is permitted to read scene_gt.json.
    gt_path = scene_dir / "scene_gt.json"
    gt_obj = read_json(gt_path)
    entries = gt_obj.get(str(im_id))
    require(isinstance(entries, list) and gt_index < len(entries), "GT_INSTANCE_NOT_FOUND")
    row = entries[gt_index]
    require(type(row.get("obj_id")) is int and row["obj_id"] == obj_id,
            "GT_OBJECT_ID_MISMATCH")
    gt = PoseM2C.from_bop(
        np.asarray(row["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
        row["cam_t_m2c"],
    )
    measures = compare_pose_m2c(estimate, gt, symmetry_policy="raw")
    record = {
        "schema": EVAL_SCHEMA,
        "status": "ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY",
        "sample": {"im_id": im_id, "obj_id": obj_id, "gt_index": gt_index},
        "frozen_manifest_sha256": sha256_file(frozen_path),
        "candidate_sha256": frozen["candidate_sha256"],
        "gt_annotation_sha256": sha256_file(gt_path),
        "translation_error_mm": measures["translation_error_mm"],
        "rotation_error_deg_raw": measures["rotation_error_deg_raw"],
        "symmetry_aware": False,
        "oracle_mask_used": True,
        "gt_pose_read_by_evaluator": True,
        "gt_pose_read_by_predictor": False,
        "prediction_rerun_or_reselected": False,
        "bop_benchmark_score_claimed": False,
        "physical_accuracy_claimed": False,
        "publishable": False,
        "note": "This frame was already used for earlier diagnostics. Raw rotation ignores symmetry; not an unbiased held-out score.",
    }
    _save_new_json(output, record)
    print("T106_FP_POSTHOC_EVAL_PASS")
    print("TRANSLATION_ERROR_MM:", record["translation_error_mm"])
    print("RAW_ROTATION_ERROR_DEG:", record["rotation_error_deg_raw"])
    print("SYMMETRY_AWARE: False; PHYSICAL_ACCURACY_CLAIMED: False")
    print("EVALUATION_FILE:", output)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="phase", required=True)
    fr = sub.add_parser("freeze", help="GT-free candidate snapshot, does not read annotations")
    fr.add_argument("--candidate", type=Path, required=True)
    fr.add_argument("--bundle", type=Path, required=True)
    fr.add_argument("--output", type=Path, required=True)
    ev = sub.add_parser("evaluate", help="posthoc read scene_gt only after SHA verification")
    ev.add_argument("--candidate", type=Path, required=True)
    ev.add_argument("--frozen", type=Path, required=True)
    ev.add_argument("--scene-dir", type=Path, required=True)
    ev.add_argument("--models-dir", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.phase == "freeze":
        freeze(args.candidate, args.bundle, args.output)
    elif args.phase == "evaluate":
        evaluate(args.candidate, args.frozen, args.scene_dir, args.models_dir, args.output)


if __name__ == "__main__":
    main()
