import numpy as np

# 第一步：准备深度图和目标 Mask
depth_m = np.array([
    [1.0, 0.0],
    [2.0, 3.0],
])

target_mask = np.array([
    [True, True],
    [False, True],
])

# 第二步：只保留目标中的有效深度
valid = np.isfinite(depth_m) & (depth_m > 0)
select_mask = target_mask & valid

# 第三步：提取像素坐标和深度
v, u = np.nonzero(select_mask)
z = depth_m[select_mask]

# 第四步：像素反投影
fx, fy = 2.0, 2.0
cx, cy = 0.0, 0.0

x = (u - cx) * z / fx
y = (v - cy) * z / fy

# 第五步：组合 XYZ
points = np.column_stack((x, y, z))

print("select_mask:")
print(select_mask)

print("v =", v)
print("u =", u)
print("z =", z)

print("points:")
print(points)

print("shape:", points.shape)

expected = np.array([
    [0.0, 0.0, 1.0],
    [1.5, 1.5, 3.0],
])

np.testing.assert_allclose(points, expected)
assert points.shape == (2, 3)

print("EXP028 Mask to XYZ: PASS")
