"""EXP035-E: test report provenance, fail-closed demo positions and frame chain."""
from __future__ import annotations

import copy
import importlib.util
import json

import numpy as np
import pytest

from src.experiments.exp034_label_contamination_b2 import make_label_contaminated_scene
from src.experiments.exp035_safe_pipeline_e import (
    BASE_ORACLE, CAMERA_ORACLE, WORLD_ORACLE, compare_open3d,
    run, verify_e1_to_e5, with_calibration_claim,
)
from src.experiments.exp035_se3_contract_c import make_demo_chain
from src.geometry.cluster_localization import (
    extract_publishable_positions, validate_cluster_localization,
)
from src.geometry.localization_report_audit import audit_and_extract
from src.geometry.rigid_transform import RigidTransform, inspect_numeric_position
from src.geometry.validated_position_pipeline import run_safe_pipeline


@pytest.fixture
def report():
    pts, labels = make_label_contaminated_scene()
    raw = validate_cluster_localization(pts, labels)
    raw["published_positions"] = extract_publishable_positions(raw)
    return raw


@pytest.fixture
def chain():
    return make_demo_chain()


def _run(report, transforms):
    return run_safe_pipeline(report, transforms)


def test_e1_canonical_single_b(report):
    extracted = audit_and_extract(report)
    assert [x["cluster_label"] for x in extracted] == [1]
    np.testing.assert_allclose(extracted[0]["position_m"], CAMERA_ORACLE, atol=1e-12, rtol=0)


def test_e1_invalid_a_is_never_candidate(report, chain):
    result = _run(report, [chain[0]])
    assert result["candidate_count"] == 1
    assert result["rejected_source_cluster_labels"] == [0]
    assert [x["cluster_label"] for x in result["objects"]] == [1]


def test_e2_numeric_base_oracle(report, chain):
    result = _run(report, [chain[0]])
    np.testing.assert_allclose(result["objects"][0]["diagnostics"]["p_target_demo_m"],
                               BASE_ORACLE, atol=1e-12, rtol=0)


def test_e2_expected_blocked_status(report, chain):
    result = _run(report, [chain[0]])
    assert result["status"] == "BLOCKED_UNCALIBRATED"
    assert result["objects"][0]["status"] == "BLOCKED_UNCALIBRATED"


def test_e2_no_published_position(report, chain):
    result = _run(report, [chain[0]])
    assert result["publishable"] is False
    assert result["published_positions"] == []
    assert result["objects"][0]["position_m"] is None


def test_e2_reason_mentions_unverified_edge(report, chain):
    reasons = _run(report, [chain[0]])["objects"][0]["reasons"]
    assert reasons == ["UNVERIFIED_EXTRINSIC",
                       "UNVERIFIED_TRANSFORM: camera_optical_frame -> robot_base"]


def test_e3_claim_true_never_authorizes_publication(report, chain):
    bc = with_calibration_claim(chain[0], True)
    result = _run(report, [bc])
    assert result["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert result["publishable"] is False
    assert result["published_positions"] == []
    assert result["objects"][0]["position_m"] is None


def test_e3_exact_intermediate_reason(report, chain):
    bc = with_calibration_claim(chain[0], True)
    obj = _run(report, [bc])["objects"][0]
    assert obj["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert obj["reasons"] == ["GLOBAL_VALIDATION_REQUIRED"]


def test_e3_numeric_inspector_matches_actual_contract(chain):
    bc = with_calibration_claim(chain[0], True)
    diag = inspect_numeric_position(CAMERA_ORACLE, frame_id="camera_optical_frame", transform=bc)
    assert diag["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert diag["reasons"] == ["GLOBAL_VALIDATION_REQUIRED"]
    assert diag["publishable"] is False and diag["position_m"] is None


def test_e4_legacy_audit_ignores_cache(report):
    report["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.160],
    })
    assert [x["cluster_label"] for x in audit_and_extract(report)] == [1]


def test_e4_added_invalid_a_in_cache_rejected(report, chain):
    report["published_positions"].append({
        "cluster_label": 0,
        "frame_id": "camera_optical_frame",
        "position_m": [0.099, 0.059, 1.16],
    })
    with pytest.raises(ValueError, match="published_positions mismatch"):
        _run(report, [chain[0]])


def test_e4_absent_cache_is_reconstructed(report, chain):
    del report["published_positions"]
    result = _run(report, [chain[0]])
    assert result["cache_audit"] == "ABSENT_RECOMPUTED"
    assert result["candidate_count"] == 1


def test_e4_present_valid_cache_is_checked(report, chain):
    assert _run(report, [chain[0]])["cache_audit"] == "VERIFIED_AGAINST_OBJECTS"


@pytest.mark.parametrize("field,bad", [
    ("cluster_label", -1), ("cluster_label", True), ("cluster_label", 0),
    ("cluster_label", 0.0), ("cluster_label", None),
    ("frame_id", "world"), ("frame_id", ""),
    ("position_m", [9, 9, 9]), ("position_m", None),
    ("position_m", [True, 0, 1]), ("position_m", [np.nan, 0, 1]),
    ("position_m", [1, 2]),
])
def test_e4_corrupt_cached_record_is_rejected(report, chain, field, bad):
    report["published_positions"][0][field] = bad
    with pytest.raises(ValueError):
        _run(report, [chain[0]])


def test_e4_duplicate_cache_label_rejected(report, chain):
    report["published_positions"].append(copy.deepcopy(report["published_positions"][0]))
    with pytest.raises(ValueError, match="published_positions mismatch"):
        _run(report, [chain[0]])


def test_e4_wrong_report_frame_rejected(report, chain):
    report["frame_id"] = "world"
    with pytest.raises(ValueError, match="frame mismatch"):
        _run(report, [chain[0]])


def test_e4_wrong_item_frame_rejected(report, chain):
    report["objects"][1]["frame_id"] = "world"
    report["published_positions"][0]["frame_id"] = "world"
    with pytest.raises(ValueError, match="frame mismatch"):
        _run(report, [chain[0]])


def test_e4_wrong_transform_start_frame_rejected(report, chain):
    bad = RigidTransform("world", "base", np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError, match="frame mismatch"):
        _run(report, [bad])


def test_e4_disconnected_transform_chain_rejected(report, chain):
    bad = RigidTransform("tool_frame", "world", np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError, match="frame mismatch"):
        _run(report, [chain[0], bad])


@pytest.mark.parametrize("transforms", [None, [], (), "invalid", [42], [None]])
def test_e4_invalid_transform_list_rejected(report, transforms):
    with pytest.raises(ValueError):
        _run(report, transforms)


@pytest.mark.parametrize("field", ["status", "valid_count", "invalid_count", "objects"])
def test_e4_missing_report_fields_rejected(report, chain, field):
    del report[field]
    with pytest.raises(ValueError):
        _run(report, [chain[0]])


@pytest.mark.parametrize("field", ["frame_id"])
def test_e4_missing_frame_rejected(report, chain, field):
    del report[field]
    with pytest.raises(ValueError, match="report.frame_id"):
        _run(report, [chain[0]])


def test_e5_world_hand_oracle(report, chain):
    bc = with_calibration_claim(chain[0], True)
    result = _run(report, [bc, chain[1]])
    np.testing.assert_allclose(result["objects"][0]["diagnostics"]["p_target_demo_m"],
                               WORLD_ORACLE, atol=1e-12, rtol=0)


def test_e5_base_stage_oracle(report, chain):
    bc = with_calibration_claim(chain[0], True)
    steps = _run(report, [bc, chain[1]])["objects"][0]["diagnostics"]["steps"]
    assert len(steps) == 2
    np.testing.assert_allclose(steps[0]["p_target_demo_m"], BASE_ORACLE,
                               atol=1e-12, rtol=0)


def test_e5_unverified_world_stays_blocked(report, chain):
    bc = with_calibration_claim(chain[0], True)
    result = _run(report, [bc, chain[1]])
    assert result["status"] == "BLOCKED_UNCALIBRATED"
    assert result["published_positions"] == []
    assert result["objects"][0]["reasons"][-1] == "UNVERIFIED_TRANSFORM: robot_base -> world"


def test_e5_claimed_true_base_stays_blocked(report, chain):
    bc = with_calibration_claim(chain[0], True)
    assert _run(report, [bc])["objects"][0]["publishable"] is False


def test_e5_both_claim_true_still_cannot_publish(report, chain):
    bc = with_calibration_claim(chain[0], True)
    wb = with_calibration_claim(chain[1], True)
    result = _run(report, [bc, wb])
    assert result["status"] == "READY_FOR_GLOBAL_VALIDATION"
    assert result["published_positions"] == []
    assert result["objects"][0]["position_m"] is None


def test_e5_claim_propagation_not_trust_evidence(report, chain):
    bc = with_calibration_claim(chain[0], True)
    result = _run(report, [bc, chain[1]])
    assert [step["calibration_verified_claim"] for step in result["transform_chain"]] == [True, False]
    assert all(step["trusted_calibration_record_available"] is False
               for step in result["transform_chain"])


def test_all_invalid_source_produces_no_candidate(report, chain):
    report["objects"][1]["status"] = "INVALID_CLUSTER"
    report["objects"][1]["publishable"] = False
    report["objects"][1]["position_m"] = None
    report["objects"][1]["reasons"] = ["TEST_INJECTED"]
    report["valid_count"] = 0
    report["invalid_count"] = 2
    report["status"] = "ALL_INVALID"
    report["published_positions"] = []
    result = _run(report, [chain[0]])
    assert result["status"] == "ALL_INVALID"
    assert result["objects"] == [] and result["published_positions"] == []


def test_no_objects_source_produces_no_candidate(report, chain):
    report["objects"] = []
    report["valid_count"] = 0
    report["invalid_count"] = 0
    report["status"] = "NO_OBJECTS"
    report["published_positions"] = []
    result = _run(report, [chain[0]])
    assert result["status"] == "NO_OBJECTS"
    assert result["objects"] == [] and result["candidate_count"] == 0


def test_does_not_mutate_report(report, chain):
    frozen = copy.deepcopy(report)
    _run(report, chain)
    assert report == frozen


def test_does_not_mutate_transforms(report, chain):
    before = [t.as_matrix().copy() for t in chain]
    _run(report, chain)
    for i, t in enumerate(chain):
        np.testing.assert_array_equal(before[i], t.as_matrix())


def test_independent_oracles_e1_to_e5(report):
    results = verify_e1_to_e5(report)
    assert set(results) == {"E2_unverified_base", "E3_claimed_base", "E5_partial_chain"}


def test_experiment_json_round_trip(tmp_path, report):
    path = tmp_path / "results/exp034/exp034_C_validated_localization.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    result = run(tmp_path)
    assert result["open3d_comparison"] in ("PASS", "SKIPPED")
    output = tmp_path / "results/exp035/exp035_E_safe_pipeline.json"
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["scenarios"]["E2_unverified_base"]["published_positions"] == []


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None,
                    reason="Open3D is not available in this environment")
def test_open3d_comparison(report):
    compare_open3d(report)
