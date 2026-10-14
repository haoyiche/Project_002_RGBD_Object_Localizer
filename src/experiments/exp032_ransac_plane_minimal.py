
import numpy as np
import open3d as o3d

np.set_printoptions(precision=5, suppress=True)

# =====================================
# 1. 构建 9 个桌面点
#    Shape: (9, 3)
#    单位: m
# =====================================
table_points = np.array([
    [x, y, 1.0]
    for y in (0.0, 0.1, 0.2)
    for x in (0.0, 0.1, 0.2)
], dtype=np.float64)

# 3 个非平面点
object_points = np.array([
    [0.05, 0.04, 1.15],
    [0.14, 0.17, 1.20],
    [0.22, 0.08, 1.25],
], dtype=np.float64)

# =====================================
# 2. 拼接输入点云
#    Shape: (12, 3)
# =====================================
points = np.vstack([
    table_points,
    object_points,
])

# =====================================
# 3. 构造 Open3D 点云
# =====================================
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# =====================================
# 4. RANSAC 平面分割
# =====================================
distance_threshold = 0.01
ransac_n = 3
num_iterations = 1000

plane_model, inliers = pcd.segment_plane(
    distance_threshold=distance_threshold,
    ransac_n=ransac_n,
    num_iterations=num_iterations,
)

# =====================================
# 5. 拆解平面系数
# =====================================
a, b, c, d = plane_model

normal = np.array([a, b, c])
normal_norm = np.linalg.norm(normal)

# 点到平面的距离
# Shape: (12,)
distances = np.abs(
    points @ normal + d
) / normal_norm

# =====================================
# 6. 取得内点 / 非内点
# =====================================
inliers = np.array(
    sorted(inliers), dtype=np.int64
)

all_indices = np.arange(len(points))

outliers = np.setdiff1d(
    all_indices,
    inliers,
)

plane_pcd = pcd.select_by_index(
    inliers.tolist()
)

other_pcd = pcd.select_by_index(
    inliers.tolist(),
    invert=True,
)

# =====================================
# 7. 输出诊断结果
# =====================================
print("Open3D version:", o3d.__version__)

print("\nInput shape:", points.shape)
print("Plane coefficients:", plane_model)

print("\nInlier indices:", inliers)
print("Outlier indices:", outliers)

print("Inlier count:", len(inliers))
print("Outlier count:", len(outliers))

print("\nDistances to fitted plane:")
for i, distance in enumerate(distances):
    print(f"P{i:02d}: {distance:.6f} m")

print("\nPlane cloud shape:",
      np.asarray(plane_pcd.points).shape)

print("Other cloud shape:",
      np.asarray(other_pcd.points).shape)
# 与真实桌面 z=1 的法向量比较
true_normal = np.array([0.0, 0.0, 1.0])

cos_angle = abs(
    np.dot(normal, true_normal)
) / normal_norm

cos_angle = np.clip(cos_angle, -1.0, 1.0)

angle_deg = np.degrees(np.arccos(cos_angle))

# 真实桌面上的参考点
reference_point = np.array([0.0, 0.0, 1.0])

plane_error_m = abs(
    np.dot(normal, reference_point) + d
) / normal_norm

print("\n=== Plane Quality ===")
print("Normal angle error:", angle_deg, "deg")
print("Plane reference error:", plane_error_m, "m")
# =============================
# EXP032-A2：真实桌面分割质量
# =============================

true_table_indices = set(range(9))
true_object_indices = set(range(9, 12))

predicted_inliers = set(inliers.tolist())

table_detected = len(
    predicted_inliers & true_table_indices
)

object_leakage = len(
    predicted_inliers & true_object_indices
)

table_recall = table_detected / 9.0

print("\n=== Segmentation Quality ===")
print("Table points detected:", table_detected, "/ 9")
print("Object points wrongly included:", object_leakage, "/ 3")
print("Table recall:", table_recall)