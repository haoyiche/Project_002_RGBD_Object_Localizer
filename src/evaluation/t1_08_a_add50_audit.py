"""CPU-only ADD/ADD-S audit of T1-06-A's 50 frozen FoundationPose poses.

No inference, no candidate reselection, no modification to frozen evidence.
ADD-S here is a geometry diagnostic, not a declared BOP symmetry policy.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree

from src.evaluation import t1_06_a_fp_fixed_batch as batch
from src.evaluation import t1_06_a_fp_posthoc_eval as posthoc
from src.evaluation.t1_04_a_pose_metrics import PoseM2C, compare_pose_m2c


def require(condition, message):
    if not condition:
        raise ValueError(message)


def point_metrics_mm(vertices_m, estimate, gt):
    p = np.asarray(vertices_m, dtype=np.float64)
    require(p.ndim == 2 and p.shape[1] == 3 and len(p) >= 3,
            "INVALID_MODEL_VERTICES")
    require(np.isfinite(p).all(), "NONFINITE_MODEL_VERTICES")
    require(isinstance(estimate, PoseM2C) and isinstance(gt, PoseM2C),
            "POSE_M2C_REQUIRED")
    a = p @ estimate.R.T + estimate.t_m
    b = p @ gt.R.T + gt.t_m
    add = float(np.linalg.norm(a - b, axis=1).mean() * 1000)
    adds = float(cKDTree(b).query(a, k=1)[0].mean() * 1000)
    require(np.isfinite([add, adds]).all() and adds <= add + 1e-7,
            "POINT_METRIC_INVALID")
    return add, adds


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    require(a.shape == (50,) and np.isfinite(a).all(), "INVALID_STATS")
    return {"mean": float(a.mean()), "median": float(np.median(a)),
            "p95_linear": float(np.percentile(a, 95)), "max": float(a.max())}


def run():
    import open3d as o3d

    root = Path("outputs/t1_06_a/fp_fixed50_v1")
    eval_root = Path("outputs/t1_06_a/fp_fixed50_eval_v1")
    scene = Path("data/raw/bop/lm/test/000001")
    models = Path("data/raw/bop/lm/models")
    dest = Path("outputs/t1_08_a/fp_add50_v1.json")
    require(not dest.exists(), "OUTPUT_EXISTS_USE_NEW_VERSION")

    plan_path = root / "selection_manifest.json"
    frozen_path = root / "frozen_predictions.json"
    summary_path = eval_root / "batch_summary.json"
    plan = posthoc.read_json(plan_path)
    frozen = posthoc.read_json(frozen_path)
    summary = posthoc.read_json(summary_path)
    ids = plan.get("frame_ids")
    require(plan.get("schema") == batch.PLAN_SCHEMA and
            plan.get("count") == 50 and plan.get("obj_id") == 1 and
            plan.get("gt_index") == 0 and plan.get("refine_iterations") == 5 and
            plan.get("gt_read_during_prediction") is False and
            plan.get("oracle_mask_used") is True,
            "PLAN_PROTOCOL_INVALID")
    require(isinstance(ids, list) and len(ids) == 50 and
            ids == sorted(set(ids)), "INVALID_FRAME_IDS")
    require(frozen.get("schema") == batch.BATCH_SCHEMA and
            frozen.get("frame_count") == 50 and
            frozen.get("gt_evaluated") is False and
            frozen.get("publishable") is False and
            frozen.get("selection_manifest_sha256") == batch.sha(plan_path),
            "FROZEN_BATCH_INVALID")
    require(summary.get("count_expected") == 50 and
            summary.get("count_evaluated") == 50 and
            summary.get("failed_count") == 0 and
            summary.get("frame_ids") == ids and
            summary.get("frozen_predictions_sha256") == batch.sha(frozen_path),
            "OLD_EVAL_SUMMARY_INVALID")
    rows = frozen.get("candidates")
    old_samples = summary.get("samples")
    require(isinstance(rows, list) and isinstance(old_samples, list) and
            len(rows) == len(old_samples) == 50, "INCOMPLETE_50")

    args = SimpleNamespace(obj_id=1, gt_index=0, iteration=5,
                           scene_dir=scene, models_dir=models)
    estimates = []
    for im_id, record, old in zip(ids, rows, old_samples):
        require(record.get("frame_id") == im_id and old.get("im_id") == im_id,
                "FRAME_ORDER_MISMATCH")
        frame_dir = root / "frames" / f"{im_id:06d}"
        bundle = root / "input_bundles" / f"frame_{im_id:06d}"
        expected = plan["frame_input_sha256"][str(im_id)]
        verified = batch.verify_frame(args, im_id, bundle, frame_dir, expected)
        require(verified == record, "FROZEN_RECORD_CHANGED")
        for label, path in batch.get_paths(args, im_id)[1].items():
            require(path.is_file() and batch.sha(path) == expected[label],
                    f"ORIGINAL_INPUT_CHANGED:{im_id}:{label}")
        candidate = posthoc.read_json(root / record["candidate_file"])
        _, _, pose, identity = posthoc.validate_candidate(candidate)
        require(identity == (im_id, 1, 0), "POSE_ID_MISMATCH")
        estimates.append(pose)

    print("T108_A_FROZEN_50_PREFLIGHT_PASS: GT_NOT_READ_YET")

    gt_path = scene / "scene_gt.json"
    require(batch.sha(gt_path) == summary.get("gt_annotation_sha256"),
            "GT_ANNOTATION_CHANGED")
    gt_rows = posthoc.read_json(gt_path)
    info = posthoc.read_json(models / "models_info.json")["1"]
    sym_declared = ("symmetries_discrete" in info or
                    "symmetries_continuous" in info)
    model_path = models / "obj_000001.ply"
    vertices_mm = np.asarray(o3d.io.read_triangle_mesh(str(model_path)).vertices)
    vertices_m = np.asarray(vertices_mm, dtype=np.float64) * 0.001
    require(vertices_m.ndim == 2 and vertices_m.shape[1] == 3 and
            len(vertices_m) >= 3 and np.isfinite(vertices_m).all(),
            "INVALID_MESH")
    out_rows = []
    for im_id, estimate, old, frozen_row in zip(ids, estimates, old_samples, rows):
        entries = gt_rows.get(str(im_id))
        require(isinstance(entries, list) and len(entries) >= 1 and
                entries[0].get("obj_id") == 1, "GT_ID_MISMATCH")
        entry = entries[0]
        gt = PoseM2C.from_bop(
            np.asarray(entry["cam_R_m2c"], dtype=np.float64).reshape(3, 3),
            entry["cam_t_m2c"])
        original = compare_pose_m2c(estimate, gt, symmetry_policy="raw")
        require(abs(original["translation_error_mm"] -
                    old["translation_error_mm"]) < 1e-5 and
                abs(original["rotation_error_deg_raw"] -
                    old["rotation_error_deg_raw"]) < 1e-5,
                "OLD_POSE_METRIC_MISMATCH")
        add, adds = point_metrics_mm(vertices_m, estimate, gt)
        out_rows.append({"im_id": im_id, "add_mm": add,
                         "adds_nearest_neighbor_mm": adds,
                         "candidate_sha256": frozen_row["candidate_sha256"]})

    report = {
        "schema": "t1-08-a-fp-fixed50-add-mesh-vertices-v1",
        "scope": "previously_seen_BOP_LM_scene1_obj1_oracle_mask_50frames",
        "count": len(out_rows), "model_vertex_count": len(vertices_m),
        "model_vertex_units_input": "mm", "model_vertex_units_metric": "m",
        "model_mesh_sha256": batch.sha(model_path),
        "frozen_predictions_sha256": batch.sha(frozen_path),
        "gt_annotation_sha256": batch.sha(gt_path),
        "symmetry_declared_in_models_info": sym_declared,
        "add": stats([x["add_mm"] for x in out_rows]),
        "adds_nearest_neighbor_diagnostic": stats(
            [x["adds_nearest_neighbor_mm"] for x in out_rows]),
        "samples": out_rows, "prediction_rerun": False,
        "held_out": False, "oracle_mask_used": True,
        "bop_benchmark_score_claimed": False,
        "physical_accuracy_claimed": False,
        "publishable": False,
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("VERTICES:", report["model_vertex_count"])
    print("SYMMETRY_DECLARED:", sym_declared)
    print("ADD_MM:", report["add"])
    print("ADD_S_DIAGNOSTIC_MM:", report["adds_nearest_neighbor_diagnostic"])
    print("OUTPUT:", dest)
    print("T108_A_ADD50_PASS")


if __name__ == "__main__":
    run()
