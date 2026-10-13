"""EXP031-D acceptance tests: hand-calculated oracle + Open3D comparison."""

import numpy as np
import pytest

from src.experiments.exp031_voxel_numpy import voxel_downsample_numpy


def seven_point_case():
    """EXP031-C: D is red and C occurs three times."""
    points = np.array([
        [0.00, 0.00, 1.00],  # A red
        [0.04, 0.00, 1.00],  # B green
        [0.08, 0.00, 1.00],  # C blue
        [0.09, 0.00, 1.00],  # D red
        [1.00, 0.00, 1.00],  # E magenta
        [0.08, 0.00, 1.00],  # C2 blue
        [0.08, 0.00, 1.00],  # C3 blue
    ], dtype=np.float64)
    colors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return points, colors


def assert_close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def sort_xyz_with_rgb(xyz, rgb):
    """Sort two arrays with ONE shared ordering, preserving their association."""
    order = np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))
    return xyz[order], rgb[order]


def test_hand_calculated_oracle():
    pts = np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0]], dtype=float)
    cols = np.eye(3, dtype=float)
    keys, xyz, rgb, counts = voxel_downsample_numpy(pts, cols, 0.05)
    np.testing.assert_array_equal(keys, [[0, 0, 0], [20, 0, 0]])
    np.testing.assert_array_equal(counts, [2, 1])
    assert_close(xyz, [[0.005, 0, 0], [1, 0, 0]])
    assert_close(rgb, [[0.5, 0.5, 0], [0, 0, 1]])


@pytest.mark.parametrize("size,expected_x,expected_rgb,expected_keys,expected_counts", [
    (0.05,
     [0.0, 0.04, 0.0825, 1.0],
     [[1, 0, 0], [0, 1, 0], [0.25, 0, 0.75], [1, 0, 1]],
     [[0, 0, 0], [1, 0, 0], [2, 0, 0], [20, 0, 0]],
     [1, 1, 4, 1]),
    (0.20,
     [0.37 / 6.0, 1.0],
     [[2/6, 1/6, 3/6], [1, 0, 1]],
     [[0, 0, 0], [5, 0, 0]],
     [6, 1]),
])
def test_seven_points_exact_expected(size, expected_x, expected_rgb,
                                     expected_keys, expected_counts):
    pts, cols = seven_point_case()
    keys, xyz, rgb, counts = voxel_downsample_numpy(pts, cols, size)
    np.testing.assert_array_equal(keys, expected_keys)
    np.testing.assert_array_equal(counts, expected_counts)
    assert_close(xyz[:, 0], expected_x)
    assert_close(xyz[:, 1:], np.tile([0.0, 1.0], (len(xyz), 1)))
    assert_close(rgb, expected_rgb)
    # Counts restore the original input centroid exactly up to rounding.
    assert_close(np.average(xyz, axis=0, weights=counts), pts.mean(axis=0))


@pytest.mark.parametrize("size", [0.05, 0.20])
def test_matches_open3d_xyz_rgb_pairing(size):
    o3d = pytest.importorskip("open3d")
    pts, cols = seven_point_case()
    _, xyz_np, rgb_np, _ = voxel_downsample_numpy(pts, cols, size)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(cols)
    down = cloud.voxel_down_sample(voxel_size=size)
    xyz_o3d = np.asarray(down.points)
    rgb_o3d = np.asarray(down.colors)

    # Compare geometry AND corresponding colors; never sort RGB separately.
    np_xyz, np_rgb = sort_xyz_with_rgb(xyz_np, rgb_np)
    o3d_xyz, o3d_rgb = sort_xyz_with_rgb(xyz_o3d, rgb_o3d)
    assert_close(np_xyz, o3d_xyz)
    assert_close(np_rgb, o3d_rgb)


def test_empty_cloud_shapes():
    pts = np.empty((0, 3), dtype=float)
    keys, xyz, rgb, counts = voxel_downsample_numpy(pts, pts, 0.05)
    assert keys.shape == xyz.shape == rgb.shape == (0, 3)
    assert counts.shape == (0,)
    assert keys.dtype == counts.dtype == np.int64
    assert xyz.dtype == rgb.dtype == np.float64


@pytest.mark.parametrize("bad", [0, -0.1, float("nan"), float("inf"),
                                  -float("inf"), True, False, "0.05", None,
                                  [0.05], 1 + 1j])
def test_invalid_voxel_size_rejected(bad):
    pts = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError):
        voxel_downsample_numpy(pts, np.zeros_like(pts), bad)


@pytest.mark.parametrize("bad_pts,bad_cols", [
    (np.array([0.0, 0.0, 0.0]), np.zeros((1, 3))),
    (np.zeros((1, 2)), np.zeros((1, 3))),
    (np.zeros((1, 3)), np.zeros((2, 3))),
    (np.zeros((1, 3)), np.zeros((1, 2))),
    (np.array([[np.nan, 0, 0]]), np.zeros((1, 3))),
    (np.array([[np.inf, 0, 0]]), np.zeros((1, 3))),
    (np.zeros((1, 3)), np.array([[np.nan, 0, 0]])),
    (np.zeros((1, 3)), np.array([[np.inf, 0, 0]])),
    (np.array([[1j, 0, 0]]), np.zeros((1, 3))),
])
def test_invalid_point_or_color_arrays_rejected(bad_pts, bad_cols):
    with pytest.raises(ValueError):
        voxel_downsample_numpy(bad_pts, bad_cols, 0.05)


def test_single_point_keeps_position_and_color():
    pts = np.array([[0.30, -0.1, 1.2]])
    cols = np.array([[0.2, 0.7, 0.9]])
    _, xyz, rgb, counts = voxel_downsample_numpy(pts, cols, 0.05)
    assert_close(xyz, pts)
    assert_close(rgb, cols)
    np.testing.assert_array_equal(counts, [1])


def test_input_permutation_does_not_change_output():
    pts, cols = seven_point_case()
    idx = np.array([6, 2, 4, 0, 5, 1, 3])
    for size in (0.05, 0.20):
        original = voxel_downsample_numpy(pts, cols, size)
        shuffled = voxel_downsample_numpy(pts[idx], cols[idx], size)
        np.testing.assert_array_equal(original[0], shuffled[0])
        np.testing.assert_array_equal(original[3], shuffled[3])
        assert_close(original[1], shuffled[1])
        assert_close(original[2], shuffled[2])


def test_exact_voxel_boundary_uses_floor():
    # Binary-exact case: size=1/8, origin=-1/16.
    # x=1/16 falls exactly on the boundary between key 0 and key 1.
    pts = np.array([[0.0, 0, 0], [0.0625, 0, 0], [0.125, 0, 0]])
    cols = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    keys, xyz, rgb, counts = voxel_downsample_numpy(pts, cols, 0.125)
    np.testing.assert_array_equal(keys, [[0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(counts, [1, 2])
    assert_close(xyz[:, 0], [0.0, 0.09375])
    assert_close(rgb, [[1, 0, 0], [0, 0.5, 0.5]])


def test_color_only_change_keeps_geometry():
    pts, cols = seven_point_case()
    original = voxel_downsample_numpy(pts, cols, 0.05)
    recolored = cols.copy()
    recolored[3] = [1, 1, 1]  # D becomes white again
    changed = voxel_downsample_numpy(pts, recolored, 0.05)
    np.testing.assert_array_equal(original[0], changed[0])
    np.testing.assert_array_equal(original[3], changed[3])
    assert_close(original[1], changed[1])
    assert not np.allclose(original[2], changed[2])
    assert_close(changed[2][2], [0.25, 0.25, 1.0])


def test_extreme_index_is_rejected_not_silently_overflowed():
    pts = np.array([[0, 0, 0], [1e18, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="voxel index"):
        voxel_downsample_numpy(pts, np.zeros_like(pts), 0.05)
