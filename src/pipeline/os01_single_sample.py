from src.geometry.dbscan import dbscan_numpy
from pathlib import Path

import numpy as np
from src.geometry.cluster_localization import (
    LocalizationGates,
    validate_cluster_localization,
)
from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_filter import radius_outlier_filter
from src.geometry.pointcloud_statistics import compute_pointcloud_statistics


def localize_single_sample(
    rgb,
    depth_m,
    target_mask,
    *,
    fx,
    fy,
    cx,
    cy,
    frame_id,
    radius_m=0.011,
    min_neighbors=1,
    dbscan_eps_m=0.011,
    dbscan_min_points=3,
    localization_gates=None,
):
    if not isinstance(frame_id, str) or not frame_id.strip():
        raise ValueError("frame_id must be a nonempty string")

    # Stage 1: RGB-D -> point cloud
    points, colors, selected = mask_to_pointcloud(
        depth_m, rgb, target_mask,
        fx, fy, cx, cy,
    )

    # Stage 2: remove radius outliers
    filtered = radius_outlier_filter(
        points,
        radius=radius_m,
        min_neighbors=min_neighbors,
    )

    filtered_points = filtered["filtered_points"]

    # Stage 3: construct a diagnostic-only report
    report = {
        "frame_id": frame_id,
        "status": "DIAGNOSTIC_ONLY",
        "publishable": False,
        "position_m": None,
        "mask_pixel_count": int(np.count_nonzero(target_mask)),
        "valid_point_count": int(points.shape[0]),
        "filtered_point_count": int(filtered_points.shape[0]),
        "diagnostics": {
            "neighbor_counts": filtered["neighbor_counts"].tolist(),
            "kept_indices": filtered["kept_indices"].tolist(),
            "centroid_camera_m": None,
            "bbox_extent_m": None,
            "cluster_labels": [],
            "cluster_count": 0,
            "cluster_noise_count": 0,
            "geometry_gate": None,
        },
    }

    # Handle empty cases before calculating statistics
    if report["mask_pixel_count"] == 0:
        report["status"] = "NO_OBJECTS"
        return report

    if report["valid_point_count"] == 0:
        report["status"] = "NO_VALID_DEPTH"
        return report

    if report["filtered_point_count"] == 0:
        report["status"] = "NO_POINTS_AFTER_FILTER"
        return report

    # Stage 4: geometric statistics
    stats = compute_pointcloud_statistics(filtered_points)

    report["diagnostics"]["centroid_camera_m"] = (
        stats["centroid"].tolist()
    )
    report["diagnostics"]["bbox_extent_m"] = (
        stats["bbox_extent"].tolist()
    )


    # Stage 5: DBSCAN clustering on filtered points
    labels = dbscan_numpy(
        filtered_points,
        eps=dbscan_eps_m,
        min_points=dbscan_min_points,
    )

    cluster_ids = np.unique(labels[labels >= 0])

    report["diagnostics"]["cluster_labels"] = labels.tolist()

    report["diagnostics"]["cluster_count"] = int(
        cluster_ids.size
    )

    report["diagnostics"]["cluster_noise_count"] = int(
        np.count_nonzero(labels == -1)
    )

    # Stage 6: EXP034 local geometry validation
    # Default thresholds are educational examples, not production limits.
    gate_profile = (
        "EXP034_DEMO"
        if localization_gates is None
        else "CUSTOM"
    )

    gates = (
        LocalizationGates()
        if localization_gates is None
        else localization_gates
    )

    local_report = validate_cluster_localization(
        filtered_points,
        labels,
        frame_id=frame_id,
        gates=gates,
    )

    # Convert the local gate result into diagnostic-only information.
    # Never forward EXP034's local `publishable` as robot permission.
    report["diagnostics"]["geometry_gate"] = {
        "profile": gate_profile,
        "status": local_report["status"],
        "valid_count": local_report["valid_count"],
        "invalid_count": local_report["invalid_count"],
        "objects": [
            {
                "cluster_label": obj["cluster_label"],
                "local_gate_passed": obj["status"] == "VALID",
                "reasons": obj["reasons"],
                "centroid_camera_m": (
                    obj["diagnostics"]["centroid_m"]
                ),
                "bbox_extent_m": (
                    obj["diagnostics"]["extent_m"]
                ),
            }
            for obj in local_report["objects"]
        ],
    }

    # Robot publication remains forbidden.
    report["position_m"] = None
    report["publishable"] = False


    # Stage 7: propagate local gate rejection to pipeline status
    # if local_report["status"] == "ALL_INVALID":
    #     report["status"] = "GEOMETRY_GATE_REJECTED"
    # Stage 7: map local geometry status to pipeline status
    if local_report["status"] == "NO_OBJECTS":
        report["status"] = "NO_CLUSTERS"

    elif local_report["status"] == "ALL_INVALID":
        report["status"] = "GEOMETRY_GATE_REJECTED"

        # Fail-closed: no robot position publication
    report["position_m"] = None
    report["publishable"] = False

    return report


def main():
    sample_path = Path(
        "data/raw/synthetic/os01_sample_000.npz"
    )

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

    diagnostics = report["diagnostics"]

    print("Frame:", report["frame_id"])
    print("Mask pixels:", report["mask_pixel_count"])
    print("Valid points:", report["valid_point_count"])
    print("Neighbor counts:", diagnostics["neighbor_counts"])
    print("Filtered points:", report["filtered_point_count"])

    print(
        "Centroid (m):",
        diagnostics["centroid_camera_m"],
    )
    print(
        "Bbox extent (m):",
        diagnostics["bbox_extent_m"],
    )

    print("Status:", report["status"])
    print("Publishable:", report["publishable"])
    print(
        "DBSCAN labels:",
        diagnostics["cluster_labels"],
    )
    print(
        "Cluster count:",
        diagnostics["cluster_count"],
    )
    print(
        "Cluster noise count:",
        diagnostics["cluster_noise_count"],
    )
    gate = diagnostics["geometry_gate"]

    print("Local gate status:", gate["status"])
    print("Local gate valid count:", gate["valid_count"])
    print(
        "Local gate reasons:",
        [obj["reasons"] for obj in gate["objects"]],
    )


if __name__ == "__main__":
    main()