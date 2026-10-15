"""EXP033: Minimal Open3D DBSCAN experiment, using a synthetic point cloud.

Study protocol: record predictions before running this file.
Units: meters. Each row is one XYZ point in camera/world-consistent coordinates.
"""
from collections import Counter

import numpy as np
import open3d as o3d


# Five nearby points in object A (N_A=5, 3D XYZ in meters).
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

# Object B has exactly the same internal geometry, shifted 0.50 m in X.
object_b = object_a.copy()
object_b[:, 0] += 0.50

# Two isolated points. These are synthetic clutter, not pre-filtered.
clutter = np.array(
    [
        [0.25, 0.25, 1.15],
        [0.90, 0.50, 1.20],
    ],
    dtype=np.float64,
)

points = np.vstack([object_a, object_b, clutter])
assert points.shape == (12, 3)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

eps_m = 0.018
min_points = 5
labels = np.asarray(
    pcd.cluster_dbscan(
        eps=eps_m,
        min_points=min_points,
        print_progress=False,
    ),
    dtype=np.int64,
)

print("Open3D version:", o3d.__version__)
print("Input shape:", points.shape, "dtype:", points.dtype)
print("eps:", eps_m, "m; min_points:", min_points)
print("P00-P01 distance (m):", np.linalg.norm(points[0] - points[1]))
print("P00-P04 distance (m):", np.linalg.norm(points[0] - points[4]))
print("\nIndex | X | Y | Z | DBSCAN label")
for i, (p, label) in enumerate(zip(points, labels)):
    print(f"P{i:02d} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {label}")

counts = Counter(map(int, labels))
cluster_ids = sorted(label for label in counts if label >= 0)
noise_count = counts.get(-1, 0)
print("\nCluster count:", len(cluster_ids))
print("Cluster sizes (label:size):", {k: counts[k] for k in cluster_ids})
print("Noise count (-1):", noise_count)
print("Noise indices:", np.flatnonzero(labels == -1).tolist())
print("Total accounted for:", len(labels))
