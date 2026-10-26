from pathlib import Path

import numpy as np
import pytest

from src.experiments.os01_a_synthetic_rgbd_sample import (
    main as generate_sample,
)
from src.pipeline.os01_single_sample import (
    localize_single_sample,
)


@pytest.fixture
def sample_inputs(tmp_path, monkeypatch):
    # 每个测试在独立临时目录生成样本
    monkeypatch.chdir(tmp_path)
    generate_sample()

    sample_path = Path(
        "data/raw/synthetic/os01_sample_000.npz"
    )

    with np.load(sample_path, allow_pickle=False) as sample:
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


@pytest.mark.parametrize(
    "min_neighbors, expected_count, expected_indices, "
    "expected_centroid, expected_extent",
    [
        (
            1,
            5,
            [0, 1, 2, 3, 4],
            [0.007, 0.001, 1.0],
            [0.020, 0.010, 0.0],
        ),
        (
            2,
            4,
            [0, 1, 3, 4],
            [0.010, 0.000, 1.0],
            [0.010, 0.010, 0.0],
        ),
    ],
)
def test_pipeline_parameter_ablation(
    sample_inputs,
    min_neighbors,
    expected_count,
    expected_indices,
    expected_centroid,
    expected_extent,
):
    report = localize_single_sample(
        **sample_inputs,
        radius_m=0.011,
        min_neighbors=min_neighbors,
    )

    assert report["mask_pixel_count"] == 6
    assert report["valid_point_count"] == 5
    assert report["filtered_point_count"] == expected_count

    # assert report["status"] == "DIAGNOSTIC_ONLY"
    expected_status = (
        "DIAGNOSTIC_ONLY"
        if min_neighbors == 1
        else "GEOMETRY_GATE_REJECTED"
    )

    assert report["status"] == expected_status
    gate = report["diagnostics"]["geometry_gate"]

    if min_neighbors == 1:
        assert gate["status"] == "OK"
        assert gate["valid_count"] == 1
        assert gate["invalid_count"] == 0
    else:
        assert gate["status"] == "ALL_INVALID"
        assert gate["valid_count"] == 0
        assert gate["invalid_count"] == 1
        assert gate["objects"][0]["reasons"] == [
            "POINT_COUNT_OUT_OF_RANGE"
        ]
    assert report["position_m"] is None
    assert report["publishable"] is False

    diagnostics = report["diagnostics"]

    assert diagnostics["neighbor_counts"] == [2, 2, 1, 3, 2]
    assert diagnostics["kept_indices"] == expected_indices

    np.testing.assert_allclose(
        diagnostics["centroid_camera_m"],
        expected_centroid,
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        diagnostics["bbox_extent_m"],
        expected_extent,
        rtol=0,
        atol=1e-12,
    )


def test_pipeline_empty_mask(sample_inputs):
    inputs = dict(sample_inputs)
    inputs["target_mask"] = np.zeros_like(
        sample_inputs["target_mask"],
        dtype=bool,
    )

    report = localize_single_sample(**inputs)

    assert report["status"] == "NO_OBJECTS"
    assert report["mask_pixel_count"] == 0
    assert report["valid_point_count"] == 0
    assert report["filtered_point_count"] == 0
    assert report["position_m"] is None
    assert report["publishable"] is False
    assert report["diagnostics"]["centroid_camera_m"] is None


def test_pipeline_no_valid_depth(sample_inputs):
    inputs = dict(sample_inputs)

    depth_modified = sample_inputs["depth_m"].copy()
    depth_modified[sample_inputs["target_mask"]] = 0.0

    inputs["depth_m"] = depth_modified

    report = localize_single_sample(**inputs)

    assert report["status"] == "NO_VALID_DEPTH"
    assert report["mask_pixel_count"] == 6
    assert report["valid_point_count"] == 0
    assert report["filtered_point_count"] == 0
    assert report["position_m"] is None
    assert report["publishable"] is False
    assert report["diagnostics"]["centroid_camera_m"] is None


def test_pipeline_no_points_after_filter(sample_inputs):
    report = localize_single_sample(
        **sample_inputs,
        radius_m=0.011,
        min_neighbors=4,
    )

    assert report["status"] == "NO_POINTS_AFTER_FILTER"
    assert report["mask_pixel_count"] == 6
    assert report["valid_point_count"] == 5
    assert report["filtered_point_count"] == 0
    assert report["position_m"] is None
    assert report["publishable"] is False
    assert report["diagnostics"]["centroid_camera_m"] is None