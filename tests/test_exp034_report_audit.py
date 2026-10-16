"""EXP034 Boss C: RED/PATCH/GREEN regression suite.

Run from Project_002_RGBD_Object_Localizer repo root:
    python -m pytest -q tests/test_exp034_report_audit.py
"""
from src.geometry.localization_report_audit import audit_and_extract
import copy
import pytest

report_ok = {
    "status": "OK",
    "valid_count": 1,
    "invalid_count": 0,
    "objects": [
        {
            "cluster_label": 0,
            "frame_id": "camera_optical_frame",
            "status": "VALID",
            "publishable": True,
            "position_m": [0.1, 0.2, 1.0],
        }
    ],
}

import copy
import math
import pytest


def test_missing_status_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    del bad["status"]
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_boolean_coordinate_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    bad["objects"][0]["position_m"] = [True, 0.2, 1.0]
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_missing_valid_count_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    del bad["valid_count"]
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_missing_invalid_count_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    del bad["invalid_count"]
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_missing_objects_must_be_rejected():
    # Consistent NO_OBJECTS counters: old .get("objects", []) silently passed.
    bad = {"status": "NO_OBJECTS", "valid_count": 0, "invalid_count": 0}
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_negative_cluster_label_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    bad["objects"][0]["cluster_label"] = -1
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_float_cluster_label_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    bad["objects"][0]["cluster_label"] = 0.5
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_whitespace_frame_id_must_be_rejected():
    bad = copy.deepcopy(report_ok)
    bad["objects"][0]["frame_id"] = "   "
    with pytest.raises(ValueError):
        audit_and_extract(bad)

def test_normal_publish():
    out = audit_and_extract(report_ok)
    assert len(out) == 1
    assert out[0]["position_m"] == [0.1, 0.2, 1.0]
    # 深拷贝，不共享引用
    out[0]["position_m"][0] = 999.0
    assert report_ok["objects"][0]["position_m"][0] == 0.1


def test_all_invalid():
    report = {
        "status": "ALL_INVALID",
        "valid_count": 0,
        "invalid_count": 1,
        "objects": [{
            "cluster_label": 0,
            "frame_id": "cam",
            "status": "INVALID_CLUSTER",
            "publishable": False,
            "position_m": None,
        }],
    }
    assert audit_and_extract(report) == []


def test_all_noise():
    report = {
        "status": "NO_OBJECTS",
        "valid_count": 0,
        "invalid_count": 0,
        "objects": [],
    }
    assert audit_and_extract(report) == []


def test_status_conflict():
    bad = copy.deepcopy(report_ok)
    bad["status"] = "ALL_INVALID"
    with pytest.raises(ValueError):
        audit_and_extract(bad)


def test_duplicate_label():
    bad = copy.deepcopy(report_ok)
    bad["objects"].append(copy.deepcopy(bad["objects"][0]))
    bad["valid_count"] = 2
    with pytest.raises(ValueError):
        audit_and_extract(bad)
