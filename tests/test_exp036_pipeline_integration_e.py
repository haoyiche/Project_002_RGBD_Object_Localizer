"""EXP036-E: real EXP034/035 gate integration, independent numeric oracles."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.experiments.exp036_pipeline_integration_e import (
    BASE, CAMERA, make_report, make_exact_camera_report,
    identity_camera_transform, camera_to_base,
    synthetic_reference, run,
)
from src.geometry.localization_evaluation_adapter import (
    DERIVED, INDEPENDENT, evaluate_localization_pipeline,
)


def run_adapter(report=None, tr=None, references=None):
    if report is None:
        report = make_report()
    if tr is None:
        tr = [identity_camera_transform()]
    if references is None:
        references = {2: synthetic_reference()}
    return evaluate_localization_pipeline(report, tr, references)


def test_e1_user_oracle_sqrt405_mm():
    d = run_adapter()
    np.testing.assert_allclose(d["comparison_rows"][0]["error_mm"],
                               np.sqrt(405.0), atol=1e-10, rtol=0)
    assert d["synthetic_statistics"]["count"] == 1
    assert d["source_frame_id"] == CAMERA and d["final_frame_id"] == CAMERA


def test_e2_invalid_cluster_not_zero():
    d = run_adapter()
    assert d["comparison_rows"][1]["reason"] == "INVALID_CLUSTER"
    assert d["comparison_rows"][1]["error_mm"] is None
    assert d["comparison_rows"][1]["evaluated"] is False
    assert d["evaluated_count"] == 1 and d["excluded_count"] == 1


def test_e2_invalid_preserves_reason():
    d = run_adapter()
    assert d["comparison_rows"][1]["cluster_reasons"] == [
        "EXTENT_EXCEEDED", "CENTROID_BBOX_GAP"]


@pytest.mark.parametrize("change", [
    {"publishable": True},
    {"position_m": [1.0, 2.0, 3.0]},
    {"status": "VALID"},
])
def test_e2_contract_contradiction_fails(change):
    r = make_report()
    r["objects"][1].update(change)
    with pytest.raises(ValueError):
        run_adapter(report=r)


def test_e2_all_invalid_empty_not_fabricated_zero():
    r = make_report()
    r["status"] = "ALL_INVALID"
    r["valid_count"] = 0
    r["invalid_count"] = 1
    r["objects"] = [r["objects"][1]]
    r["published_positions"] = []
    d = run_adapter(report=r, references={})
    assert d["evaluated_count"] == 0
    assert d["synthetic_statistics"] is None
    assert d["comparison_rows"][0]["reason"] == "INVALID_CLUSTER"


def test_e3_actual_EXP035_diagnostic_unverified_claim():
    d = run_adapter(report=make_exact_camera_report(), tr=[camera_to_base()],
                    references={2: synthetic_reference(frame_id=BASE,
                                gt=[0.4, -0.05, 1.2])})
    row = d["comparison_rows"][0]
    assert row["reason"] == "UNVERIFIED_TRANSFORM_FOR_CROSS_FRAME_GT"
    assert d["synthetic_statistics"] is None
    np.testing.assert_allclose(row["demo_only_position_m"],
                               [0.4, -0.05, 1.2], atol=1e-12, rtol=0)


def test_e4_numerical_cross_frame_false_error_never_evaluated():
    d = run_adapter(report=make_exact_camera_report(), tr=[camera_to_base()],
                    references={2: synthetic_reference(frame_id=BASE,
                                gt=[0.4, -0.05, 1.2])})
    assert d["comparison_rows"][0]["error_mm"] is None
    assert d["evaluated_count"] == 0


def test_e4_nominal_claim_only_allows_synthetic_metrics():
    d = run_adapter(report=make_exact_camera_report(),
                    tr=[camera_to_base(nominal_verified_claim=True)],
                    references={2: synthetic_reference(frame_id=BASE,
                                gt=[0.4, -0.05, 1.2])})
    np.testing.assert_allclose(d["synthetic_statistics"]["mean_mm"],
                               0.0, atol=1e-10, rtol=0)
    assert d["calibration_authenticated"] is False
    assert d["publishable"] is False and d["published_positions"] == []


def test_e4_wrong_reference_frame_rejected():
    d = run_adapter(tr=[camera_to_base(nominal_verified_claim=True)],
                    references={2: synthetic_reference(frame_id=CAMERA)})
    assert d["comparison_rows"][0]["reason"] == "FRAME_MISMATCH"
    assert d["synthetic_statistics"] is None


def test_e5_cyclic_derived_gt_returns_demo_zero_not_metric():
    d = run_adapter(report=make_exact_camera_report(), tr=[camera_to_base()],
                    references={2: synthetic_reference(
                        frame_id=BASE, gt=[0.4, -0.05, 1.2],
                        gt_source=DERIVED)})
    row = d["comparison_rows"][0]
    assert row["reason"] == "SELF_DERIVED_REFERENCE"
    np.testing.assert_allclose(row["demo_numerical_error_mm"], 0, atol=1e-10, rtol=0)
    assert d["synthetic_statistics"] is None


@pytest.mark.parametrize("change,reason", [
    ({"sample_id": "wrong"}, "PROVENANCE_MISMATCH"),
    ({"source_sample_id": "wrong"}, "PROVENANCE_MISMATCH"),
    ({"reference_kind": "measured_gt"}, "UNSUPPORTED_GT_KIND"),
    ({"unit": "mm"}, "UNCONFIRMED_UNITS"),
    ({"frame_id": "robot_base"}, "FRAME_MISMATCH"),
    ({"gt_source": "unknown"}, "UNKNOWN_GT_PROVENANCE"),
    ({"gt_independent": "true"}, "UNKNOWN_GT_PROVENANCE"),
    ({"gt_independent": False}, "CONTRADICTORY_GT_PROVENANCE"),
    ({"synthetic_gt_m": [0.1, float("nan"), 1.0]}, "INVALID_GROUND_TRUTH"),
    ({"synthetic_gt_m": [True, 0.05, 1.0]}, "INVALID_GROUND_TRUTH"),
    ({"synthetic_gt_m": [0.1, 0.05]}, "INVALID_GROUND_TRUTH"),
    ({"synthetic_gt_m": None}, "MISSING_GROUND_TRUTH"),
])
def test_invalid_reference_excluded_with_diagnostic(change, reason):
    r = synthetic_reference()
    r.update(change)
    d = run_adapter(references={2: r})
    assert d["evaluated_count"] == 0 and d["excluded_count"] == 2
    assert d["synthetic_statistics"] is None
    assert d["comparison_rows"][0]["reason"] == reason
    assert d["comparison_rows"][0]["error_mm"] is None


def test_missing_reference_does_not_invent_0mm():
    d = run_adapter(references={})
    assert d["comparison_rows"][0]["reason"] == "MISSING_GROUND_TRUTH"
    assert d["synthetic_statistics"] is None


def test_unknown_cluster_reference_rejected():
    with pytest.raises(ValueError, match="unknown"):
        run_adapter(references={999: synthetic_reference()})


def test_duplicate_cluster_fails_exp034_audit():
    r = make_report()
    r["objects"][1]["cluster_label"] = 2
    with pytest.raises(ValueError):
        run_adapter(report=r)


def test_tampered_cached_position_fails_exp035_gate():
    r = make_report()
    r["published_positions"][0]["position_m"][0] += 0.05
    with pytest.raises(ValueError, match="position_m mismatch"):
        run_adapter(report=r)


def test_report_object_frame_mismatch_fails_exp035_gate():
    r = make_report()
    r["objects"][1]["frame_id"] = "world"
    with pytest.raises(ValueError, match="frame_id"):
        run_adapter(report=r)


def test_transform_source_frame_mismatch_fails_exp035_gate():
    from src.geometry.rigid_transform import RigidTransform
    tr = RigidTransform("world", BASE, np.eye(3), [0, 0, 0])
    with pytest.raises(ValueError, match="frame"):
        run_adapter(tr=[tr])


def test_missing_report_sample_id_fails():
    r = make_report()
    del r["sample_id"]
    with pytest.raises(ValueError, match="sample_id"):
        run_adapter(report=r)


def test_inputs_not_mutated():
    r = make_report()
    refs = {2: synthetic_reference()}
    before_r = copy.deepcopy(r)
    before_refs = copy.deepcopy(refs)
    run_adapter(report=r, references=refs)
    assert r == before_r and refs == before_refs


def test_pipeline_json_deterministic_and_nonpublishable(tmp_path):
    a = run(tmp_path)
    p = tmp_path / "results/exp036/exp036_E_pipeline_integration.json"
    assert p.exists()
    blob = p.read_bytes()
    b = run(tmp_path)
    assert a == b and blob == p.read_bytes()
    assert json.loads(blob)["physical_accuracy_verified"] is False
    for item in (a["E1_E2"], a["E3"], a["E4"], a["E5"]):
        assert item["publishable"] is False and item["published_positions"] == []
