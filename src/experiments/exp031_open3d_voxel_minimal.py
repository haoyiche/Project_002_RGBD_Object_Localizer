
import numpy as np
import open3d as o3d

# 单位：m；所有点都在相机坐标系中
points = np.array([
    [0.00, 0.00, 1.00],
    [0.04, 0.00, 1.00],
    [0.08, 0.00, 1.00],
    [0.09, 0.00, 1.00],
    [1.00, 0.00, 1.00],
], dtype=np.float64)

colors = np.array([
    [1.0, 0.0, 0.0],  # A red
    [0.0, 1.0, 0.0],  # B green
    [0.0, 0.0, 1.0],  # C blue
    [1.0, 1.0, 1.0],  # D white
    [1.0, 0.0, 1.0],  # E magenta
], dtype=np.float64)

# EXP031-C：仅修改 D 点的颜色
# D 是原始数组中的第 4 个点，索引为 3
colors[3] = [1.0, 0.0, 0.0]

# EXP031-B: 仅增加 C 点的观测数量
extra_points = np.repeat(points[2:3], 2, axis=0)
extra_colors = np.repeat(colors[2:3], 2, axis=0)

points = np.vstack([points, extra_points])
colors = np.vstack([colors, extra_colors])

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points)
cloud.colors = o3d.utility.Vector3dVector(colors)

np.set_printoptions(precision=4, suppress=True)

print("Open3D version:", o3d.__version__)
print("Input shape:", points.shape)
print("Input point count:", len(points))

for voxel_size in [0.05, 0.20]:
    origin = points.min(axis=0) - voxel_size * 0.5

    voxel_indices = np.floor(
        (points - origin) / voxel_size
    ).astype(np.int64)

    print("\nVoxel origin:", origin)

    # for name, index in zip("ABCDE", voxel_indices):
    #     print(f"Point {name}: voxel index = {index}")
    point_labels = ["A", "B", "C", "D", "E", "C2", "C3"]

    assert len(point_labels) == len(voxel_indices)

    for name, index in zip(point_labels, voxel_indices, strict=True):
        print(f"Point {name}: voxel index = {index}")
        filtered = cloud.voxel_down_sample(
            voxel_size=voxel_size
        )

    xyz_out = np.asarray(filtered.points)
    rgb_out = np.asarray(filtered.colors)

    # 仅为了让打印顺序容易对照
    order = np.argsort(xyz_out[:, 0])

    print("\n======================")
    print("voxel_size =", voxel_size, "m")
    print("Output shape:", xyz_out.shape)
    print("Output XYZ:\n", xyz_out[order])
    print("Output RGB:\n", rgb_out[order])
    print("Input centroid:", points.mean(axis=0))
    print("Output centroid:", xyz_out.mean(axis=0))