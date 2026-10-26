import numpy as np

from src.geometry.cluster_localization import LocalizationGates
from src.pipeline.os01_single_sample import localize_single_sample


def test_rejected_cluster_must_be_explicit_and_fail_closed():
    # Reproduce the OS-01-A synthetic sample
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[2:4, 3:6] = [200, 60, 40]

    depth_m = np.zeros((6, 8), dtype=np.float64)
    depth_m[2:4, 3:6] = 1.0
    depth_m[2, 3] = 0.0

    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True

    gates = LocalizationGates(
        min_points=6,
        max_points=20,
        max_extent_m=(0.04, 0.04, 0.02),
        max_centroid_bbox_gap_m=0.02,
    )

    report = localize_single_sample(
        rgb,
        depth_m,
        mask,
        fx=100.0,
        fy=100.0,
        cx=3.5,
        cy=2.5,
        frame_id="camera_optical_frame",
        localization_gates=gates,
    )

    geometry_gate = report["diagnostics"]["geometry_gate"]

    # Stage 1–5: point cloud and clustering still succeed
    assert report["valid_point_count"] == 5
    assert report["filtered_point_count"] == 5
    assert report["diagnostics"]["cluster_labels"] == [0] * 5

    # Stage 6: the local geometry gate rejects the cluster
    assert geometry_gate["status"] == "ALL_INVALID"
    assert geometry_gate["valid_count"] == 0
    assert geometry_gate["invalid_count"] == 1

    obj = geometry_gate["objects"][0]

    assert obj["local_gate_passed"] is False
    assert obj["reasons"] == ["POINT_COUNT_OUT_OF_RANGE"]

    # Rejected geometry remains available for diagnostics
    np.testing.assert_allclose(
        report["diagnostics"]["centroid_camera_m"],
        [0.007, 0.001, 1.0],
        rtol=0,
        atol=1e-12,
    )

    # Fail-closed robot publication
    assert report["position_m"] is None
    assert report["publishable"] is False

    # New contract: explicit rejection at pipeline level
    assert report["status"] == "GEOMETRY_GATE_REJECTED"


def test_all_dbscan_noise_must_report_no_clusters():
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)

    depth_m = np.zeros((6, 8), dtype=np.float64)
    depth_m[2:4, 3:6] = 1.0
    depth_m[2, 3] = 0.0

    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:6] = True

    report = localize_single_sample(
        rgb,
        depth_m,
        mask,
        fx=100.0,
        fy=100.0,
        cx=3.5,
        cy=2.5,
        frame_id="camera_optical_frame",
        radius_m=0.011,
        min_neighbors=1,
        dbscan_eps_m=0.011,
        dbscan_min_points=6,
    )

    diagnostics = report["diagnostics"]
    gate = diagnostics["geometry_gate"]

    # 1. Point cloud and radius filtering succeeded.
    assert report["valid_point_count"] == 5
    assert report["filtered_point_count"] == 5

    # 2. DBSCAN classified all points as noise.
    assert diagnostics["cluster_labels"] == [-1] * 5
    assert diagnostics["cluster_count"] == 0
    assert diagnostics["cluster_noise_count"] == 5

    # 3. EXP034 received no clusters.
    assert gate["status"] == "NO_OBJECTS"
    assert gate["valid_count"] == 0
    assert gate["invalid_count"] == 0
    assert gate["objects"] == []

    # 4. Overall geometry remains diagnostic data.
    np.testing.assert_allclose(
        diagnostics["centroid_camera_m"],
        [0.007, 0.001, 1.0],
        rtol=0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        diagnostics["bbox_extent_m"],
        [0.020, 0.010, 0.0],
        rtol=0,
        atol=1e-12,
    )

    # 5. Robot publication remains forbidden.
    assert report["position_m"] is None
    assert report["publishable"] is False

    # 6. New pipeline-level status contract.
    assert report["status"] == "NO_CLUSTERS"