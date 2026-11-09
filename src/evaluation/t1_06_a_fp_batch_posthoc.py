"""T1-06-A: GT-separated posthoc evaluation of ALL 50 frozen FoundationPose poses.

From Project_002 repo root:
  python -m src.evaluation.t1_06_a_fp_batch_posthoc \
    --run-dir outputs/t1_06_a/fp_fixed50_v1 \
    --output-dir outputs/t1_06_a/fp_fixed50_eval_v1

Optional paired PCA+ICP comparison:
  --icp-summary path/to/t1_04_b4/batch_summary.json

Preflight validates ALL 50 candidates, frozen manifests, original RGB-D/mask/CAD
SHA-256 fingerprints and pinned local source/weight metadata BEFORE reading
scene_gt.json. Existing predictions are NEVER modified or rerun. Writes outputs
atomically to a new directory. Dataset-annotation-relative raw diagnostics ONLY.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import numpy as np

from src.evaluation import t1_06_a_fp_fixed_batch as batch
from src.evaluation import t1_06_a_fp_posthoc_eval as posthoc
from src.evaluation import t1_04_b4_fixed_50_batch as b4

SCHEMA = "t1-06-a-fp-fixed50-annotation-relative-evaluation-v1"
COMPARE_SCHEMA = "t1-06-a-fp-vs-icp-fixed50-paired-comparison-v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_number(value, label):
    require(type(value) in (float, int) and math.isfinite(value) and value >= 0,
            "INVALID_NUMERIC_METRIC:" + label)
    return float(value)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.ndim == 1 and a.size == 50 and np.isfinite(a).all() and np.all(a >= 0),
            "STATS_SCOPE_OR_VALUE_INVALID")
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p95_linear": float(np.percentile(a, 95, method="linear")),
        "max": float(a.max()),
    }


def make_batch_args(user):
    """Pass exact, explicitly fixed protocol into the existing batch auditor."""
    return SimpleNamespace(
        count=50, obj_id=1, gt_index=0, iteration=5,
        scene_dir=Path(user.scene_dir).resolve(),
        models_dir=Path(user.models_dir).resolve(),
        run_dir=Path(user.run_dir).resolve(),
        fp_root=Path(user.fp_root).resolve(),
        cache_root=Path(user.cache_root).resolve(),
        adapter=Path(user.adapter).resolve(),
        posthoc=Path(user.posthoc).resolve(),
    )


def preflight(user):
    """No reads of GT here, even if the user passes an ICP summary."""
    args = make_batch_args(user)
    require(args.run_dir.is_dir(), "FROZEN_BATCH_DIR_MISSING")
    plan_file = args.run_dir / "selection_manifest.json"
    frozen_file = args.run_dir / "frozen_predictions.json"
    plan = posthoc.read_json(plan_file)
    frozen = posthoc.read_json(frozen_file)

    require(plan.get("schema") == batch.PLAN_SCHEMA, "PLAN_SCHEMA_MISMATCH")
    require(plan.get("count") == 50, "NOT_FIXED_50")
    require(plan.get("obj_id") == 1 and plan.get("gt_index") == 0
            and plan.get("refine_iterations") == 5, "FIXED_PROTOCOL_MISMATCH")
    require(plan.get("gt_read_during_prediction") is False
            and plan.get("oracle_mask_used") is True, "GT_BLIND_PLAN_INVALID")
    require(plan.get("source_commit") == batch.SOURCE_COMMIT
            and plan.get("image_digest") == batch.IMAGE_DIGEST,
            "PINNED_RUNTIME_LABEL_MISMATCH")
    expected = batch.make_plan(args)  # NO scene_gt read; validates hashes/code/weights
    batch.verify_selection(plan, expected)
    ids = plan["frame_ids"]
    require(len(ids) == 50 and ids == sorted(set(ids)), "INVALID_FIXED_50_FRAME_IDS")
    require(ids == b4.select_rgbd_frames(args.scene_dir, count=50, gt_index=0),
            "FRAME_SELECTION_DIFFERS_FROM_T104_B4")

    require(frozen.get("schema") == batch.BATCH_SCHEMA
            and frozen.get("status") == "ALL_FRAMES_PREDICTED_AND_FROZEN_GT_BLIND"
            and frozen.get("frame_count") == 50
            and frozen.get("gt_evaluated") is False
            and frozen.get("publishable") is False,
            "FROZEN_BATCH_STATUS_INVALID")
    require(frozen.get("selection_manifest_sha256") == batch.sha(plan_file),
            "FROZEN_PLAN_SHA_MISMATCH")
    require(frozen.get("source_commit") == batch.SOURCE_COMMIT
            and frozen.get("image_digest") == batch.IMAGE_DIGEST,
            "FROZEN_RUNTIME_LABEL_MISMATCH")

    # Fail closed on partial or changed files BEFORE any GT is opened.
    verified_rows = batch.collect(args, plan)
    require(frozen.get("candidates") == verified_rows,
            "FROZEN_CANDIDATE_LIST_MISMATCH")
    require(len(verified_rows) == 50, "FROZEN_BATCH_INCOMPLETE")
    for im_id in ids:
        paths = batch.get_paths(args, im_id)[1]
        source_hashes = plan["frame_input_sha256"][str(im_id)]
        for key, p in paths.items():
            require(p.is_file() and batch.sha(p) == source_hashes[key],
                    f"ORIGINAL_INPUT_CHANGED:{im_id}:{key}")

    return args, plan, frozen, verified_rows


def load_icp_summary(path, expected_ids, expected_gt_sha):
    """Optional comparison of same frames, same GT annotation and raw metrics."""
    path = Path(path).resolve()
    icp = posthoc.read_json(path)
    require(icp.get("schema") == b4.SUMMARY_SCHEMA
            and icp.get("frame_count_expected") == 50
            and icp.get("frame_count_evaluated") == 50
            and icp.get("failed_count") == 0
            and icp.get("is_50_frame_scope") is True,
            "ICP_REFERENCE_NOT_COMPLETE_B4_50")
    require(icp.get("obj_id") == 1 and icp.get("gt_index") == 0
            and icp.get("raw_rotation_not_symmetry_aware") is True
            and icp.get("dataset_gt_used_only_in_evaluation") is True
            and icp.get("bop_benchmark_score_claimed") is False,
            "ICP_REFERENCE_PROTOCOL_MISMATCH")
    require(icp.get("gt_annotation_sha256") == expected_gt_sha,
            "ICP_AND_FP_GT_ANNOTATION_DIFFER")
    freeze = path.parent / "frozen_predictions.json"
    require(freeze.is_file()
            and batch.sha(freeze) == icp.get("frozen_predictions_sha256"),
            "ICP_FROZEN_PREDICTIONS_HASH_MISMATCH")
    samples = icp.get("samples")
    require(isinstance(samples, list) and len(samples) == 50,
            "ICP_SAMPLES_MISSING")
    ids = [r.get("im_id") for r in samples]
    require(ids == expected_ids, "ICP_FRAME_SET_OR_ORDER_DIFFERENT")
    for r in samples:
        finite_number(r.get("translation_error_mm"), "icp_translation")
        finite_number(r.get("rotation_error_deg_raw"), "icp_rotation")
    return icp, batch.sha(path)


def paired_comparison(fp_samples, icp, icp_sha, gt_sha, frozen_sha):
    rows = []
    for f, b in zip(fp_samples, icp["samples"]):
        require(f["im_id"] == b["im_id"], "PAIRED_ID_MISMATCH")
        rows.append({
            "im_id": f["im_id"],
            "fp_translation_error_mm": f["translation_error_mm"],
            "icp_translation_error_mm": b["translation_error_mm"],
            "fp_raw_rotation_error_deg": f["rotation_error_deg_raw"],
            "icp_raw_rotation_error_deg": b["rotation_error_deg_raw"],
            "fp_minus_icp_translation_mm": f["translation_error_mm"] - b["translation_error_mm"],
            "fp_minus_icp_raw_rotation_deg": f["rotation_error_deg_raw"] - b["rotation_error_deg_raw"],
        })
    t_fp = [r["fp_translation_error_mm"] for r in rows]
    t_icp = [r["icp_translation_error_mm"] for r in rows]
    r_fp = [r["fp_raw_rotation_error_deg"] for r in rows]
    r_icp = [r["icp_raw_rotation_error_deg"] for r in rows]
    return {
        "schema": COMPARE_SCHEMA,
        "scope": "same_50_previously_diagnosed_lm_frames_oracle_mask",
        "frame_count": 50,
        "fp_frozen_predictions_sha256": frozen_sha,
        "icp_summary_sha256": icp_sha,
        "shared_gt_annotation_sha256": gt_sha,
        "translation_error_mm": {"foundationpose": stats(t_fp), "pca_icp": stats(t_icp)},
        "rotation_error_deg_raw": {"foundationpose": stats(r_fp), "pca_icp": stats(r_icp)},
        "paired_frame_counts": {
            "fp_lower_translation_error": sum(x < y for x, y in zip(t_fp, t_icp)),
            "fp_lower_raw_rotation_error": sum(x < y for x, y in zip(r_fp, r_icp)),
            "ties_translation": sum(x == y for x, y in zip(t_fp, t_icp)),
            "ties_raw_rotation": sum(x == y for x, y in zip(r_fp, r_icp)),
        },
        "samples": rows,
        "symmetry_aware": False, "bop_benchmark_score_claimed": False,
        "physical_accuracy_claimed": False, "publishable": False,
        "note": "Diagnostic comparison on frames previously used for ICP experiments. Not held-out or symmetry-aware.",
    }


def save_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def evaluate_fixed50(user):
    dest = Path(user.output_dir).resolve()
    require(not dest.exists(), "EVALUATION_OUTPUT_ALREADY_EXISTS")
    args, plan, frozen, records = preflight(user)
    # All candidate / original input provenance has now been checked, before GT.
    print("FROZEN_BATCH_PREFLIGHT_PASS count=50; GT_READ=False", flush=True)

    stage = None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage = Path(tempfile.mkdtemp(prefix=".fp_eval_stage_", dir=dest.parent))
        per_frame = stage / "frames"
        per_frame.mkdir()
        samples = []
        gt_sha = None
        for i, row in enumerate(records, start=1):
            im_id = row["frame_id"]
            stem = f"{im_id:06d}"
            candidate = args.run_dir / row["candidate_file"]
            snap = args.run_dir / row["freeze_file"]
            target = per_frame / f"evaluation_{stem}.json"
            with redirect_stdout(io.StringIO()):
                result = posthoc.evaluate(candidate, snap, args.scene_dir, args.models_dir, target)
            require(result.get("candidate_sha256") == row["candidate_sha256"],
                    f"EVALUATION_CANDIDATE_SHA_MISMATCH:{im_id}")
            require(result.get("sample") == {"im_id": im_id, "obj_id": 1, "gt_index": 0},
                    f"EVALUATION_ID_MISMATCH:{im_id}")
            require(result.get("gt_pose_read_by_evaluator") is True
                    and result.get("gt_pose_read_by_predictor") is False
                    and result.get("symmetry_aware") is False
                    and result.get("physical_accuracy_claimed") is False
                    and result.get("bop_benchmark_score_claimed") is False
                    and result.get("publishable") is False,
                    f"EVALUATION_SAFETY_FLAG_MISMATCH:{im_id}")
            if gt_sha is None:
                gt_sha = result["gt_annotation_sha256"]
            require(result["gt_annotation_sha256"] == gt_sha,
                    "GT_CHANGED_DURING_EVALUATION")
            t = finite_number(result.get("translation_error_mm"), "translation")
            a = finite_number(result.get("rotation_error_deg_raw"), "rotation")
            require(a <= 180.0001, "ROTATION_ERROR_OUT_OF_RANGE")
            samples.append({
                "im_id": im_id,
                "candidate_sha256": row["candidate_sha256"],
                "evaluation_sha256": batch.sha(target),
                "translation_error_mm": t,
                "rotation_error_deg_raw": a,
                "register_seconds": finite_number(row.get("register_seconds"), "register_seconds"),
                "torch_peak_reserved_mib": finite_number(row.get("torch_peak_reserved_mib"), "peak_vram"),
            })
            print(f"EVALUATED {i}/50 id={im_id}", flush=True)
        require(len(samples) == 50, "BATCH_EVALUATION_INCOMPLETE")
        require(batch.sha(args.scene_dir / "scene_gt.json") == gt_sha,
                "GT_CHANGED_BEFORE_REPORT")
        summary = {
            "schema": SCHEMA,
            "status": "FULL_50_ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY",
            "count_expected": 50,
            "count_evaluated": len(samples),
            "failed_count": 0,
            "frame_ids": plan["frame_ids"],
            "frozen_predictions_sha256": batch.sha(args.run_dir / "frozen_predictions.json"),
            "selection_manifest_sha256": batch.sha(args.run_dir / "selection_manifest.json"),
            "gt_annotation_sha256": gt_sha,
            "source_commit": frozen["source_commit"],
            "image_digest": frozen["image_digest"],
            "translation_error_mm": stats([r["translation_error_mm"] for r in samples]),
            "rotation_error_deg_raw": stats([r["rotation_error_deg_raw"] for r in samples]),
            "latency_register_seconds_cold": stats([r["register_seconds"] for r in samples]),
            "torch_peak_reserved_mib": stats([r["torch_peak_reserved_mib"] for r in samples]),
            "raw_rotation_above_90_count": sum(r["rotation_error_deg_raw"] > 90 for r in samples),
            "raw_rotation_above_150_count": sum(r["rotation_error_deg_raw"] > 150 for r in samples),
            "oracle_mask_used": True,
            "single_frame_register_each_frame": True,
            "gt_read_only_after_freeze_audit": True,
            "symmetry_aware": False,
            "bop_benchmark_score_claimed": False,
            "physical_accuracy_claimed": False,
            "publishable": False,
            "note": "Not held-out; the same frames were used in previous ICP diagnostics. Latency is per container register() and not end-to-end throughput; PyTorch reserved memory is not total GPU VRAM.",
            "samples": samples,
        }
        save_new_json(stage / "batch_summary.json", summary)
        if user.icp_summary is not None:
            icp, icp_sha = load_icp_summary(user.icp_summary, plan["frame_ids"], gt_sha)
            comparison = paired_comparison(samples, icp, icp_sha, gt_sha,
                                           summary["frozen_predictions_sha256"])
            save_new_json(stage / "paired_comparison.json", comparison)
        require(not dest.exists(), "EVALUATION_OUTPUT_ALREADY_EXISTS")
        stage.rename(dest)
        stage = None
        print("T106_FP_FIXED50_POSTHOC_PASS count=50", flush=True)
        print("TRANSLATION_MM:", json.dumps(summary["translation_error_mm"], ensure_ascii=False))
        print("RAW_ROTATION_DEG:", json.dumps(summary["rotation_error_deg_raw"], ensure_ascii=False))
        print("RAW_ROTATION_GT90:", summary["raw_rotation_above_90_count"])
        print("REGISTER_SECONDS:", json.dumps(summary["latency_register_seconds_cold"], ensure_ascii=False))
        print("TORCH_PEAK_RESERVED_MIB:", json.dumps(summary["torch_peak_reserved_mib"], ensure_ascii=False))
        print("SUMMARY:", dest / "batch_summary.json")
        if user.icp_summary is not None:
            print("PAIRED_COMPARISON:", dest / "paired_comparison.json")
        print("NOT_HELD_OUT; NOT_SYMMETRY_AWARE; NOT_PHYSICAL_ACCURACY")
        return summary
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)



def compare_existing_evaluation(user):
    """Compare an already completed FP evaluation against frozen B4 ICP, no rerun."""
    require(user.icp_summary is not None, "ICP_SUMMARY_REQUIRED_FOR_COMPARE_ONLY")
    args, plan, frozen, records = preflight(user)  # All sources still match.
    dest = Path(user.output_dir).resolve()
    summary = posthoc.read_json(dest / "batch_summary.json")
    require(summary.get("schema") == SCHEMA
            and summary.get("status") == "FULL_50_ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY"
            and summary.get("count_expected") == 50
            and summary.get("count_evaluated") == 50
            and summary.get("failed_count") == 0
            and summary.get("frame_ids") == plan["frame_ids"]
            and summary.get("symmetry_aware") is False
            and summary.get("publishable") is False,
            "FP_EVALUATION_SUMMARY_INVALID")
    require(summary.get("frozen_predictions_sha256") ==
            batch.sha(args.run_dir / "frozen_predictions.json"),
            "FP_FROZEN_BATCH_HASH_MISMATCH")
    require(summary.get("gt_annotation_sha256") ==
            batch.sha(args.scene_dir / "scene_gt.json"),
            "FP_GT_ANNOTATION_CHANGED")
    fps = summary.get("samples")
    require(isinstance(fps, list) and len(fps) == 50, "FP_SAMPLES_INCOMPLETE")
    for f, row in zip(fps, records):
        im_id = row["frame_id"]
        require(f.get("im_id") == im_id
                and f.get("candidate_sha256") == row["candidate_sha256"],
                f"FP_FRAME_OR_PREDICTION_CHANGED:{im_id}")
        evfile = dest / "frames" / f"evaluation_{im_id:06d}.json"
        require(evfile.is_file() and batch.sha(evfile) == f.get("evaluation_sha256"),
                f"FP_EVALUATION_HASH_MISMATCH:{im_id}")
        ev = posthoc.read_json(evfile)
        require(ev.get("candidate_sha256") == row["candidate_sha256"]
                and ev.get("translation_error_mm") == f.get("translation_error_mm")
                and ev.get("rotation_error_deg_raw") == f.get("rotation_error_deg_raw")
                and ev.get("gt_annotation_sha256") == summary["gt_annotation_sha256"],
                f"FP_EVALUATION_RECORD_INVALID:{im_id}")
    icp, icp_sha = load_icp_summary(user.icp_summary, plan["frame_ids"],
                                    summary["gt_annotation_sha256"])
    result = paired_comparison(fps, icp, icp_sha,
                               summary["gt_annotation_sha256"],
                               summary["frozen_predictions_sha256"])
    target = dest / "paired_comparison.json"
    require(not target.exists(), "PAIRED_COMPARISON_ALREADY_EXISTS")
    save_new_json(target, result)
    print("T106_FP_VS_ICP_PAIRED_COMPARISON_PASS count=50")
    print("FP_LOWER_TRANSLATION_FRAMES:",
          result["paired_frame_counts"]["fp_lower_translation_error"])
    print("FP_LOWER_ROTATION_FRAMES:",
          result["paired_frame_counts"]["fp_lower_raw_rotation_error"])
    print("COMPARISON:", target)
    print("NOT_HELD_OUT; NOT_SYMMETRY_AWARE; NOT_PHYSICAL_ACCURACY")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=Path("outputs/t1_06_a/fp_fixed50_v1"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/t1_06_a/fp_fixed50_eval_v1"))
    p.add_argument("--scene-dir", type=Path, default=Path("data/raw/bop/lm/test/000001"))
    p.add_argument("--models-dir", type=Path, default=Path("data/raw/bop/lm/models"))
    p.add_argument("--adapter", type=Path, default=Path("src/evaluation/t1_06_a_fp_single_frame.py"))
    p.add_argument("--posthoc", type=Path, default=Path("src/evaluation/t1_06_a_fp_posthoc_eval.py"))
    p.add_argument("--fp-root", type=Path, default=Path(r"D:\AI_Lab\_external\FoundationPose"))
    p.add_argument("--cache-root", type=Path, default=Path(r"D:\AI_Lab\_external\fp_torch_extensions"))
    p.add_argument("--icp-summary", type=Path, default=None)
    p.add_argument("--compare-only", action="store_true",
                   help="use existing completed FP evaluation; only create paired_comparison.json")
    user = p.parse_args()
    if user.compare_only:
        compare_existing_evaluation(user)
    else:
        evaluate_fixed50(user)


if __name__ == "__main__":
    main()
