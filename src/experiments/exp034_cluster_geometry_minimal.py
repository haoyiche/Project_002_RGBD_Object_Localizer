"""EXP034 minimum experiment: per-cluster 3D centroid and AABB.

Predict the output BEFORE running. This script deliberately reuses the 12-point
EXP033 case and the independent NumPy DBSCAN implementation from that unit.
Coordinates and all geometric outputs are in meters in the camera frame.
"""

import numpy as np
import open3d as o3d

from exp033_dbscan_numpy import dbscan_numpy


object_a = np.array(
    [
        [0.00, 0.00, 1.15],
        [0.02, 0.00, 1.15],
        [0.00, 0.02, 1.15],
        [0.02, 0.02, 1.15],
        [0.01, 0.01, 1.16],
    ],
    dtype=np.float64,
)
object_b = object_a.copy()
object_b[:, 0] += 0.50
noise = np.array(
    [[0.25, 0.25, 1.15], [0.90, 0.50, 1.20]], dtype=np.float64
)
points = np.vstack([object_a, object_b, noise])  # (12, 3)

EPS_M = 0.06
MIN_POINTS = 3
FRAME_ID = "camera"

labels = dbscan_numpy(points, eps=EPS_M, min_points=MIN_POINTS)
assert labels.shape == (len(points),)
assert labels.dtype == np.int64

cluster_ids = sorted(int(label) for label in np.unique(labels) if label >= 0)
noise_indices = np.flatnonzero(labels == -1).tolist()

print("Open3D version:", o3d.__version__)
print("Point cloud:", points.shape, points.dtype)
print("Labels:", labels.tolist())
print("Frame:", FRAME_ID, "| Unit: m")
print("Cluster IDs:", cluster_ids)
print("Noise indices:", noise_indices)

for cluster_id in cluster_ids:
    cluster_points = points[labels == cluster_id]  # (M, 3)
    centroid = cluster_points.mean(axis=0)  # (3,)
    xyz_min = cluster_points.min(axis=0)  # (3,)
    xyz_max = cluster_points.max(axis=0)  # (3,)
    aabb_center = (xyz_min + xyz_max) / 2.0
    aabb_extent = xyz_max - xyz_min

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster_points)
    open3d_aabb = pcd.get_axis_aligned_bounding_box()

    np.testing.assert_allclose(open3d_aabb.get_min_bound(), xyz_min)
    np.testing.assert_allclose(open3d_aabb.get_max_bound(), xyz_max)
    np.testing.assert_allclose(open3d_aabb.get_center(), aabb_center)
    np.testing.assert_allclose(open3d_aabb.get_extent(), aabb_extent)

    print(f"\n=== Cluster {cluster_id} ===")
    print("Shape:", cluster_points.shape)
    print("Centroid (m):", centroid.tolist())
    print("AABB min (m):", xyz_min.tolist())
    print("AABB max (m):", xyz_max.tolist())
    print("AABB center (m):", aabb_center.tolist())
    print("AABB extent (m):", aabb_extent.tolist())
    print("Centroid - AABB center (m):", (centroid - aabb_center).tolist())
    print("Open3D AABB check: PASS")

print("\nEXP034 MINIMAL EXPERIMENT FINISHED")
