"""Level 2 Final-2: reproducible configuration + fail-closed case archive.

Uses the existing Final-1 interface. No new geometry or publication gateway.
Outputs are deterministic synthetic diagnostics, not physical accuracy evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from src.geometry.rigid_transform import RigidTransform
from src.pipeline import os01_d3_coordinate_chain as d3
from src.pipeline import os01_final1_demo_report as final1


CONFIG_PATH = Path("configs/level2_final_cases.json")
OUTPUT_PATH = Path("outputs/level2_final/failure_cases.json")
VALID_SCENARIOS = frozenset({
    "baseline",
    "different_frame_uid",
    "modified_depth_same_uid",
    "expired_validation",
    "swapped_target_background_equal_counts",
    "inverse_transform",
    "calibration_claim_true",
})
VALID_STATUSES = frozenset({
    "DIAGNOSTIC_ONLY",
    "FRAME_UID_MISMATCH",
    "INPUT_DATA_HASH_MISMATCH",
    "STALE_VALIDATION_TICKET",
    "UPSTREAM_REJECTED",
    "FRAME_MISMATCH",
})


def _read_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    cfg = json.loads(raw.decode("utf-8"))
    if not isinstance(cfg, dict) or cfg.get("schema_version") != "os01_level2_final2_cases_v1":
        raise ValueError("UNSUPPORTED_CONFIG_SCHEMA")
    cases = cfg.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("EMPTY_OR_INVALID_CASES")
    ids = set()
    scenarios = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("INVALID_CASE")
        case_id = case.get("case_id")
        scenario = case.get("scenario")
        expected = case.get("expected_status")
        if not isinstance(case_id, str) or not case_id.strip() or case_id in ids:
            raise ValueError("DUPLICATE_OR_INVALID_CASE_ID")
        if scenario not in VALID_SCENARIOS or scenario in scenarios:
            raise ValueError("DUPLICATE_OR_UNKNOWN_SCENARIO")
        if expected not in VALID_STATUSES:
            raise ValueError("UNKNOWN_EXPECTED_STATUS")
        ids.add(case_id)
        scenarios.add(scenario)
    # All seven cases are required for the designed Final-2 evidence set.
    if scenarios != VALID_SCENARIOS:
        raise ValueError("INCOMPLETE_CASE_COVERAGE")
    return cfg, hashlib.sha256(raw).hexdigest()


def _run_scenario(scenario: str) -> dict:
    # Create independent inputs for each scenario; no reuse of mutable arrays.
    capture = final1.make_demo_capture()
    kwargs = {"capture": capture}

    if scenario == "different_frame_uid":
        kwargs["candidate"] = replace(capture, frame_uid="unvalidated-frame-B")
    elif scenario == "modified_depth_same_uid":
        modified = capture.depth_m.copy()
        modified[20, 30] += 0.05
        kwargs["candidate"] = replace(capture, depth_m=modified)
    elif scenario == "expired_validation":
        # D4 default TTL is 5_000_000_000 ns; boundary +1 must reject.
        kwargs["now_ns"] = 7_000_000_001
    elif scenario == "swapped_target_background_equal_counts":
        modified = capture.depth_m.copy()
        modified[20, 30] = 2.0
        modified[0, 0] = 1.2
        kwargs["capture"] = replace(capture, depth_m=modified)
    elif scenario == "inverse_transform":
        kwargs["transform"] = d3.demo_transform().inverse()
    elif scenario == "calibration_claim_true":
        base = d3.demo_transform()
        kwargs["transform"] = RigidTransform(
            base.source_frame_id,
            base.target_frame_id,
            base.R,
            base.t_m,
            calibration_verified=True,
        )
    elif scenario != "baseline":
        raise ValueError("UNKNOWN_SCENARIO")

    return final1.build_demo_report(**kwargs)


def run_audit(config_path: Path = CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    cfg, cfg_digest = _read_config(config_path)
    rows = []
    for case in cfg["cases"]:
        report = _run_scenario(case["scenario"])
        status = report["status"]
        if status != case["expected_status"]:
            raise AssertionError(
                f"{case['case_id']} expected {case['expected_status']}, got {status}"
            )
        if report["publishable"] is not False or report["position_m"] is not None:
            raise AssertionError(f"{case['case_id']}: FAIL_CLOSED_CONTRACT_BROKEN")
        if report["physical_accuracy_claimed"] is not False:
            raise AssertionError(f"{case['case_id']}: INVALID_PHYSICAL_ACCURACY_CLAIM")
        # A rejected path must not expose final-frame diagnostic geometry.
        if status != "DIAGNOSTIC_ONLY":
            if report["centroid_world_demo_m"] is not None or report["bbox_extent_world_demo_m"] is not None:
                raise AssertionError(f"{case['case_id']}: REJECTED_GEOMETRY_LEAK")
        rows.append({
            "case_id": case["case_id"],
            "scenario": case["scenario"],
            "expected_status": case["expected_status"],
            "observed_status": status,
            "passed": True,
            "reasons": report["reasons"],
            "frame_uid": report["frame_uid"],
            "input_sha256": report["input_sha256"],
            "target_point_count": report["target_point_count"],
            "centroid_camera_m": report["centroid_camera_m"],
            "centroid_world_demo_m": report["centroid_world_demo_m"],
            "bbox_extent_camera_m": report["bbox_extent_camera_m"],
            "bbox_extent_world_demo_m": report["bbox_extent_world_demo_m"],
            "position_m": report["position_m"],
            "publishable": report["publishable"],
        })
    counts = Counter(row["observed_status"] for row in rows)
    return {
        "schema_version": "os01_level2_final2_audit_v1",
        "reference_kind": "deterministic_synthetic_demo",
        "physical_accuracy_claimed": False,
        "config_path": config_path.as_posix(),
        "config_sha256": cfg_digest,
        "case_count": len(rows),
        "diagnostic_count": counts["DIAGNOSTIC_ONLY"],
        "rejected_count": len(rows) - counts["DIAGNOSTIC_ONLY"],
        "status_counts": dict(sorted(counts.items())),
        "all_cases_passed": True,
        "cases": rows,
    }


def main() -> None:
    summary = run_audit()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("Cases:", summary["case_count"])
    print("Diagnostic only:", summary["diagnostic_count"])
    print("Rejected:", summary["rejected_count"])
    print("Status counts:", summary["status_counts"])
    print("All cases passed:", summary["all_cases_passed"])
    print("Physical accuracy claimed:", summary["physical_accuracy_claimed"])
    print("Report:", OUTPUT_PATH)
    print("Level 2 Final-2 failure audit: PASS")


if __name__ == "__main__":
    main()
