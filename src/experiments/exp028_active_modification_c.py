import numpy as np

depth_before = np.array([
    [1.0, 0.0],
    [2.0, 6.0],
])

depth_after = depth_before.copy()
depth_after[1, 0] = 0.0

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

print("Mask before:")
print(mask_before)

print("Mask after:")
print(mask_after)

print("Points before:")
print(points_before)

print("Points after:")
print(points_after)

# Independent geometry references
expected_before = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 2.0],
    [3.0, 3.0, 6.0],
])

expected_after = np.array([
    [0.0, 0.0, 1.0],
    [3.0, 3.0, 6.0],
])

np.testing.assert_allclose(points_before, expected_before)
np.testing.assert_allclose(points_after, expected_after)

assert points_before.shape == (3, 3)
assert points_after.shape == (2, 3)

# Deliberate error: incompatible shapes
try:
    delta = points_after - points_before
except ValueError as error:
    print("Expected broadcasting error detected:")
    print(error)
else:
    raise AssertionError("Expected ValueError was not raised")

print("EXP028 Modification C: PASS")
