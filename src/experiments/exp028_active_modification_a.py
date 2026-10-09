import numpy as np

# 1. 原始数据
depth_m = np.array([
    [1.0, 0.0],
    [2.0, 3.0],
])

target_mask = np.array([
    [True, True],
    [False, True],
])

# 2. 主动修改 A：新增一个目标像素
target_mask[1, 0] = True

# 3. 选择属于目标且深度有效的像素
valid = np.isfinite(depth_m) & (depth_m > 0)
select_mask = target_mask & valid

# 4. 获取像素坐标和深度
v, u = np.nonzero(select_mask)
z = depth_m[select_mask]

# 5. 反投影到三维
fx, fy = 2.0, 2.0
cx, cy = 0.0, 0.0

x = (u - cx) * z / fx
y = (v - cy) * z / fy

points = np.column_stack((x, y, z))

# 6. 独立参考答案
expected = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 2.0],
    [1.5, 1.5, 3.0],
])

print("select_mask:")
print(select_mask)

print("u =", u)
print("v =", v)
print("z =", z)

print("points:")
print(points)
print("shape:", points.shape)

assert points.shape == (3, 3)
np.testing.assert_allclose(
    points, expected, rtol=0, atol=1e-12
)

print("EXP028 Modification A: PASS")
