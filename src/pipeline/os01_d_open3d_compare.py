from pathlib import Path

import numpy as np
import open3d as o3d

from src.geometry.mask_to_pointcloud import mask_to_pointcloud
from src.geometry.pointcloud_filter import radius_outlier_filter
from src.geometry.dbscan import dbscan_numpy


def main():
    path = Path("data/raw/synthetic/os01_sample_000.npz")

    with np.load(path, allow_pickle=False) as sample:
        points, colors, selected = mask_to_pointcloud(
            sample["depth_m"],
            sample["rgb"],
            sample["target_mask"],
            fx=float(sample["fx"]),
            fy=float(sample["fy"]),
            cx=float(sample["cx"]),
            cy=float(sample["cy"]),
        )

    # Reuse the production Radius Filter.
    filtered = radius_outlier_filter(
        points,
        radius=0.011,
        min_neighbors=1,
    )

    xyz = filtered["filtered_points"]
    rgb = colors[filtered["kept_indices"]]

    # Construct an Open3D PointCloud.
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(xyz)
    cloud.colors = o3d.utility.Vector3dVector(rgb)

    # Compare two DBSCAN implementations.
    labels_numpy = dbscan_numpy(
        xyz,
        eps=0.011,
        min_points=3,
    )

    labels_open3d = np.asarray(
        cloud.cluster_dbscan(
            eps=0.011,
            min_points=3,
            print_progress=False,
        ),
        dtype=np.int64,
    )

    # Test Open3D voxel downsampling.
    downsampled = cloud.voxel_down_sample(
        voxel_size=0.005,
    )

    print("Open3D version:", o3d.__version__)
    print("Original points:", points.shape)
    print("Filtered points:", xyz.shape)
    print("Open3D points:", np.asarray(cloud.points).shape)
    print("Open3D colors:", np.asarray(cloud.colors).shape)
    print("NumPy labels:", labels_numpy.tolist())
    print("Open3D labels:", labels_open3d.tolist())
    print(
        "Downsampled points:",
        np.asarray(downsampled.points).shape,
    )

    # Check geometry and cluster membership without
    # depending on arbitrary cluster ID numbering.
    assert np.allclose(np.asarray(cloud.points), xyz)
    assert len(labels_numpy) == len(xyz)
    assert len(labels_open3d) == len(xyz)

    assert np.count_nonzero(labels_numpy == -1) == 0
    assert np.count_nonzero(labels_open3d == -1) == 0

    assert np.unique(labels_numpy).size == 1
    assert np.unique(labels_open3d).size == 1

    assert len(downsampled.points) == 5
    # D1-B：主动修改体素大小
    coarse = cloud.voxel_down_sample(
        voxel_size=0.020
    )

    xyz_coarse = np.asarray(coarse.points)

    print("\n=== D1 Voxel Ablation ===")
    print("Original XYZ:\n", xyz)
    print("Coarse XYZ:\n", xyz_coarse)

    print("Original centroid:", xyz.mean(axis=0))
    print("Coarse centroid:", xyz_coarse.mean(axis=0))

    print("Original count:", len(xyz))
    print("Coarse count:", len(xyz_coarse))
    print("Open3D comparison: PASS")


if __name__ == "__main__":
    main()