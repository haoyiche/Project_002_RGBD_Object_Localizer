import numpy as np

# 原始深度
depth_before = np.array([
    [1.0, 0.0],
    [2.0, 3.0],
])

# 修改 B：仅改变右下角深度
depth_after = depth_before.copy()
depth_after[1, 1] = 6.0

target_mask = np.ones((2, 2), dtype=bool)

def to_points(depth):
    valid = np.isfinite(depth) & (depth > 0)
    select = target_mask & valid

    v, u = np.nonzero(select)
    z = depth[select]

    x = u * z / 2.0
    y = v * z / 2.0

    return select, np.column_stack((x, y, z))

mask_before, points_before = to_points(depth_before)
mask_after, points_after = to_points(depth_after)

delta = points_after - points_before

print("Points before:")
print(points_before)

print("Points after:")
print(points_after)

print("Delta:")
print(delta)

# 独立参考
expected_after = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 2.0],
    [3.0, 3.0, 6.0],
])

assert np.array_equal(mask_before, mask_after)
assert points_after.shape == (3, 3)

np.testing.assert_allclose(points_after, expected_after)
np.testing.assert_allclose(delta[2], [1.5, 1.5, 3.0])
np.testing.assert_allclose(delta[:2], 0.0)

print("EXP028 Modification B: PASS")
