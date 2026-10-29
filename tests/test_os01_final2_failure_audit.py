"""Level 2 Final-2: config completeness, case audit and output contract."""

import json
from pathlib import Path

import pytest

from src.pipeline import os01_final2_failure_audit as final2


@pytest.fixture
def config_copy(tmp_path):
    cfg = json.loads(final2.CONFIG_PATH.read_text(encoding="utf-8"))
    path = tmp_path / "level2_cases.json"
    path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    return path


def test_final2_all_7_cases_and_fail_closed():
    result = final2.run_audit()
    assert result["case_count"] == 7
    assert result["diagnostic_count"] == 2
    assert result["rejected_count"] == 5
    assert result["all_cases_passed"] is True
    assert result["physical_accuracy_claimed"] is False
    assert len({c["case_id"] for c in result["cases"]}) == 7
    assert all(c["position_m"] is None and c["publishable"] is False
               for c in result["cases"])
    assert all(c["observed_status"] == c["expected_status"]
               for c in result["cases"])
    assert all(c["centroid_world_demo_m"] is None
               for c in result["cases"] if c["observed_status"] != "DIAGNOSTIC_ONLY")
    json.dumps(result, allow_nan=False)


def test_final2_config_rejects_duplicate_case_id(config_copy):
    cfg = json.loads(config_copy.read_text(encoding="utf-8"))
    cfg["cases"][1]["case_id"] = cfg["cases"][0]["case_id"]
    config_copy.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="DUPLICATE_OR_INVALID_CASE_ID"):
        final2.run_audit(config_copy)


def test_final2_config_rejects_missing_case(config_copy):
    cfg = json.loads(config_copy.read_text(encoding="utf-8"))
    cfg["cases"].pop()
    config_copy.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="INCOMPLETE_CASE_COVERAGE"):
        final2.run_audit(config_copy)


def test_final2_detects_changed_expected_status(config_copy):
    cfg = json.loads(config_copy.read_text(encoding="utf-8"))
    cfg["cases"][0]["expected_status"] = "FRAME_MISMATCH"
    config_copy.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(AssertionError, match="expected FRAME_MISMATCH"):
        final2.run_audit(config_copy)


def test_final2_config_sha256_changes_with_content(config_copy):
    before = final2._read_config(config_copy)[1]
    cfg = json.loads(config_copy.read_text(encoding="utf-8"))
    cfg["purpose"] += " / edited"
    config_copy.write_text(json.dumps(cfg), encoding="utf-8")
    after = final2._read_config(config_copy)[1]
    assert before != after
