import pytest

from src.evaluation.os01_c1_metrics import evaluate_case
from pathlib import Path

import numpy as np

from src.experiments.os01_a_synthetic_rgbd_sample import (
    main as generate_sample,
)
from src.pipeline.os01_single_sample import (
    localize_single_sample,
)

def make_report(status, centroid):
    objects = []

    if centroid is not None:
        objects.append({
            "local_gate_passed": True,
            "centroid_camera_m": centroid,
        })

    return {
        "frame_id": "camera_optical_frame",
        "status": status,
        "position_m": None,
        "publishable": False,
        "diagnostics": {
            "geometry_gate": {
                "objects": objects,
            },
        },
    }


@pytest.mark.parametrize(
    "sample_id,status,centroid,reference,error_mm,reason",
    [
        (
            "S001",
            "DIAGNOSTIC_ONLY",
            [0.007, 0.001, 1.0],
            [0.005, 0.001, 1.0],
            2.0,
            None,
        ),
        (
            "S002",
            "DIAGNOSTIC_ONLY",
            [0.020, 0.0, 1.0],
            [0.015, 0.0, 1.0],
            5.0,
            None,
        ),
        (
            "S003",
            "GEOMETRY_GATE_REJECTED",
            None,
            [0.010, 0.0, 1.0],
            None,
            "PIPELINE_REJECTED",
        ),
        (
            "S004",
            "DIAGNOSTIC_ONLY",
            [0.012, 0.0, 1.0],
            None,
            None,
            "MISSING_REFERENCE",
        ),
    ],
)
def test_c1_four_cases(
    sample_id,
    status,
    centroid,
    reference,
    error_mm,
    reason,
):
    report = make_report(status, centroid)

    result = evaluate_case(
        sample_id=sample_id,
        report=report,
        reference_position_m=reference,
        reference_frame_id="camera_optical_frame",
    )

    assert result["sample_id"] == sample_id
    assert result["reason"] == reason

    if error_mm is None:
        assert result["eligible"] is False
        assert result["error_mm"] is None
    else:
        assert result["eligible"] is True
        assert result["error_mm"] == pytest.approx(
            error_mm,
            abs=1e-10,
        )


def test_c1_frame_mismatch():
    report = make_report(
        "DIAGNOSTIC_ONLY",
        [0.007, 0.001, 1.0],
    )

    result = evaluate_case(
        sample_id="S005",
        report=report,
        reference_position_m=[0.005, 0.001, 1.0],
        reference_frame_id="base_link",
    )

    assert result["eligible"] is False
    assert result["reason"] == "FRAME_MISMATCH"
    assert result["error_mm"] is None


def test_c1_multiple_valid_clusters():
    report = make_report(
        "DIAGNOSTIC_ONLY",
        [0.007, 0.001, 1.0],
    )

    report["diagnostics"]["geometry_gate"]["objects"].append({
        "local_gate_passed": True,
        "centroid_camera_m": [0.100, 0.001, 1.0],
    })

    result = evaluate_case(
        sample_id="S006",
        report=report,
        reference_position_m=[0.005, 0.001, 1.0],
        reference_frame_id="camera_optical_frame",
    )

    assert result["eligible"] is False
    assert result["reason"] == "NO_UNIQUE_VALID_CLUSTER"
    assert result["error_mm"] is None


@pytest.mark.parametrize(
    "bad_reference",
    [
        [float("nan"), 0.0, 1.0],
        [float("inf"), 0.0, 1.0],
        [0.0, 1.0],
    ],
)
def test_c1_invalid_reference(bad_reference):
    report = make_report(
        "DIAGNOSTIC_ONLY",
        [0.007, 0.001, 1.0],
    )

    with pytest.raises(ValueError):
        evaluate_case(
            sample_id="S007",
            report=report,
            reference_position_m=bad_reference,
            reference_frame_id="camera_optical_frame",
        )

def test_c1_with_actual_pipeline_report(tmp_path, monkeypatch):
    # Generate the original OS-01-A sample in an isolated directory.
    monkeypatch.chdir(tmp_path)
    generate_sample()

    sample_path = Path(
        "data/raw/synthetic/os01_sample_000.npz"
    )

    # Stage A: load synthetic RGB-D data.
    with np.load(sample_path, allow_pickle=False) as sample:
        report = localize_single_sample(
            sample["rgb"],
            sample["depth_m"],
            sample["target_mask"],
            fx=sample["fx"].item(),
            fy=sample["fy"].item(),
            cx=sample["cx"].item(),
            cy=sample["cy"].item(),
            frame_id=sample["frame_id"].item(),
        )

    # Stage B: verify the actual Pipeline output.
    assert report["status"] == "DIAGNOSTIC_ONLY"
    assert report["filtered_point_count"] == 5

    gate = report["diagnostics"]["geometry_gate"]
    assert gate["valid_count"] == 1

    # Stage C: evaluate against a deliberately offset
    # synthetic comparison coordinate.
    # This is NOT independent physical ground truth.
    result = evaluate_case(
        sample_id="OS01_C1_001",
        report=report,
        reference_position_m=[0.005, 0.001, 1.0],
        reference_frame_id="camera_optical_frame",
    )

    assert result["eligible"] is True
    assert result["reason"] is None
    assert result["error_mm"] == pytest.approx(
        2.0,
        abs=1e-10,
    )

    # Evaluating diagnostics must never enable robot output.
    assert report["position_m"] is None
    assert report["publishable"] is False