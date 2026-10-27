"""OS-01-C2: evaluate a verified manifest through the real diagnostic pipeline.

Run from the project root:
    python -m src.evaluation.os01_c2_batch_evaluate

The reported distances are synthetic nominal-reference discrepancies, NOT
independent physical localization accuracy or robot publication permission.
"""

from collections import Counter
import csv
from hashlib import sha256
import inspect
import json
from pathlib import Path

import numpy as np

from src.evaluation.os01_c1_metrics import evaluate_case
from src.evaluation.os01_c2_generate import validate_manifest
from src.geometry.cluster_localization import LocalizationGates
from src.pipeline.os01_single_sample import localize_single_sample


CSV_FIELDS = (
    "sample_id", "scenario_group", "source_kind", "input_path",
    "input_sha256", "reference_kind", "reference_frame_id",
    "reference_position_m", "pipeline_params", "gate_params",
    "pipeline_status", "geometry_gate_status", "geometry_valid_count",
    "geometry_invalid_count", "geometry_passed", "filtered_point_count",
    "cluster_count", "cluster_noise_count", "gate_reasons",
    "accepted_diagnostic_centroid_m", "eligible", "exclusion_reason",
    "error_mm", "publishable",
)


def _hash_file(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_input(path):
    with np.load(path, allow_pickle=False) as sample:
        return {
            "rgb": sample["rgb"],
            "depth_m": sample["depth_m"],
            "target_mask": sample["target_mask"],
            "fx": sample["fx"].item(),
            "fy": sample["fy"].item(),
            "cx": sample["cx"].item(),
            "cy": sample["cy"].item(),
            "frame_id": sample["frame_id"].item(),
        }


def _evaluate_sample(spec, root):
    if spec["source_kind"] != "synthetic":
        raise ValueError(f"unsupported sample source: {spec['sample_id']}")
    if spec["reference_kind"] not in ("synthetic_nominal_patch_center", "none"):
        raise ValueError(f"unsupported reference type: {spec['sample_id']}")

    inputs = _load_input(root / spec["input_path"])
    gate_params = spec["gate_params"]
    gates = LocalizationGates(**gate_params) if gate_params is not None else None
    report = localize_single_sample(
        **inputs,
        **spec["pipeline_params"],
        localization_gates=gates,
    )
    # This evaluator is diagnostic-only, regardless of the local gate outcome.
    if report["position_m"] is not None or report["publishable"] is not False:
        raise RuntimeError(f"unsafe publication contract: {spec['sample_id']}")

    evaluation = evaluate_case(
        sample_id=spec["sample_id"],
        report=report,
        reference_position_m=spec["reference_position_m"],
        reference_frame_id=spec["reference_frame_id"],
    )
    gate = report["diagnostics"]["geometry_gate"]
    objects = [] if gate is None else gate["objects"]
    valid = [obj for obj in objects if obj["local_gate_passed"]]
    geometry_passed = bool(valid)
    if geometry_passed != (report["status"] == "DIAGNOSTIC_ONLY"):
        raise ValueError(f"inconsistent gate/pipeline status: {spec['sample_id']}")

    # A single centroid is representable only if there is a unique valid cluster.
    diagnostic_centroid = valid[0]["centroid_camera_m"] if len(valid) == 1 else None
    reasons = sorted({reason for obj in objects for reason in obj["reasons"]})
    error = evaluation["error_mm"]
    if error is not None and (not np.isfinite(error) or error < 0):
        raise ValueError(f"invalid computed error: {spec['sample_id']}")

    return {
        "sample_id": spec["sample_id"],
        "scenario_group": spec["scenario_group"],
        "source_kind": spec["source_kind"],
        "input_path": spec["input_path"],
        "input_sha256": spec["input_sha256"],
        "reference_kind": spec["reference_kind"],
        "reference_frame_id": spec["reference_frame_id"],
        "reference_position_m": spec["reference_position_m"],
        "pipeline_params": spec["pipeline_params"],
        "gate_params": gate_params,
        "pipeline_status": report["status"],
        "geometry_gate_status": None if gate is None else gate["status"],
        "geometry_valid_count": 0 if gate is None else gate["valid_count"],
        "geometry_invalid_count": 0 if gate is None else gate["invalid_count"],
        "geometry_passed": geometry_passed,
        "filtered_point_count": report["filtered_point_count"],
        "cluster_count": report["diagnostics"]["cluster_count"],
        "cluster_noise_count": report["diagnostics"]["cluster_noise_count"],
        "gate_reasons": reasons,
        "accepted_diagnostic_centroid_m": diagnostic_centroid,
        "eligible": evaluation["eligible"],
        "exclusion_reason": evaluation["reason"],
        "error_mm": error,
        "publishable": False,
    }


def _summary(rows, manifest, manifest_digest):
    n = len(rows)
    geometry_count = sum(row["geometry_passed"] for row in rows)
    eligible = [row for row in rows if row["eligible"]]
    errors = np.asarray([row["error_mm"] for row in eligible], dtype=np.float64)
    stats = {
        "mean": float(np.mean(errors)) if errors.size else None,
        "median": float(np.median(errors)) if errors.size else None,
        "p95": float(np.percentile(errors, 95)) if errors.size else None,
        "max": float(np.max(errors)) if errors.size else None,
    }
    code_sha256 = {
        "generator": _hash_file(inspect.getfile(validate_manifest)),
        "batch_evaluator": _hash_file(__file__),
        "c1_evaluator": _hash_file(inspect.getfile(evaluate_case)),
        "pipeline": _hash_file(inspect.getfile(localize_single_sample)),
    }
    return {
        "schema_version": "os01_c2_evaluation_v1",
        "metric_name": "synthetic_nominal_reference_discrepancy_mm",
        "physical_accuracy_claimed": False,
        "source_manifest_schema": manifest["schema_version"],
        "manifest_sha256": manifest_digest,
        "code_sha256": code_sha256,
        "total_samples": n,
        "geometry_pass_count": geometry_count,
        "geometry_pass_rate": geometry_count / n if n else None,
        "geometry_nonpass_count": n - geometry_count,
        "evaluable_count": len(eligible),
        "evaluable_rate": len(eligible) / n if n else None,
        "excluded_count": n - len(eligible),
        "missing_reference_count": sum(s["reference_position_m"] is None for s in manifest["samples"]),
        "pipeline_status_counts": dict(sorted(Counter(r["pipeline_status"] for r in rows).items())),
        "exclusion_reason_counts": dict(sorted(Counter(
            r["exclusion_reason"] for r in rows if not r["eligible"]
        ).items())),
        "scenario_counts": dict(sorted(Counter(r["scenario_group"] for r in rows).items())),
        "error_stats_mm": stats,
        "percentile_method": "numpy.percentile(method='linear')",
        "note": (
            "Metrics describe deterministic synthetic nominal-reference differences "
            "among eligible cases only. Missing/rejected cases are excluded, not zero. "
            "No independent physical ground truth or robot authorization."
        ),
    }


def run_batch(
    project_root=Path("."),
    manifest_path=Path("data/manifests/os01_c_samples.json"),
    output_dir=Path("outputs/os01_c"),
):
    """Verify inputs, evaluate 20 cases, write auditable CSV + JSON, return data."""
    root = Path(project_root).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = root / manifest_file
    manifest = validate_manifest(manifest_file, root)
    manifest_digest = _hash_file(manifest_file)

    rows = [_evaluate_sample(spec, root) for spec in manifest["samples"]]
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate sample IDs")
    summary = _summary(rows, manifest, manifest_digest)

    dest = Path(output_dir)
    if not dest.is_absolute():
        dest = root / dest
    dest.mkdir(parents=True, exist_ok=True)
    csv_path = dest / "evaluation_results.csv"
    json_path = dest / "evaluation_summary.json"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            out = {
                k: (_json_cell(v) if isinstance(v, (list, dict)) else v)
                for k, v in row.items()
            }
            writer.writerow(out)
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return rows, summary, csv_path, json_path


def main():
    rows, summary, csv_path, json_path = run_batch()
    print("Evaluated:", summary["total_samples"])
    print("Geometry passed:", summary["geometry_pass_count"])
    print("Eligible:", summary["evaluable_count"])
    print("Pipeline status counts:", summary["pipeline_status_counts"])
    print("Exclusion reasons:", summary["exclusion_reason_counts"])
    print("Synthetic discrepancy stats (mm):", summary["error_stats_mm"])
    print("Physical accuracy claimed:", summary["physical_accuracy_claimed"])
    print("CSV:", csv_path)
    print("JSON:", json_path)


if __name__ == "__main__":
    main()
