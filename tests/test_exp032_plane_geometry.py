"""EXP032-D independent acceptance tests for three-point plane geometry.

Project location: tests/test_exp032_plane_geometry.py
Implementation: src/experiments/exp032_plane_numpy.py

This is the BASELINE suite. Scale-invariant degeneracy is a separate
predict-before-run engineering challenge; see the mentor instructions.
"""

import numpy as np
import pytest

from src.experiments.exp032_plane_numpy import (
    plane_from_three_points,
    point_plane_distances,
)

P0 = np.array([0.00, 0.00, 1.000])
P1 = np.array([0.10, 0.00, 1.000])
P2 = np.array([0.00, 0.10, 1.000])
FIVE = np.array([
    [0.00, 0.00, 1.000],
    [0.10, 0.00, 1.000],
    [0.00, 0.10, 1.000],
    [0.10, 0.10, 1.006],
    [0.10, 0.10, 1.020],
])


def test_hand_oracle_plane_and_dtype():
    plane = plane_from_three_points(P0, P1, P2)
    assert plane.shape == (4,)
    assert plane.dtype == np.float64
    np.testing.assert_allclose(plane, [0., 0., 1., -1.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(plane[:3]), 1.0, atol=1e-12)


def test_hand_oracle_distances_and_inlier_mask():
    plane = plane_from_three_points(P0, P1, P2)
    distances = point_plane_distances(FIVE, plane)
    assert distances.shape == (5,)
    np.testing.assert_allclose(distances, [0., 0., 0., .006, .020], rtol=0, atol=1e-12)
    assert (distances <= .01).tolist() == [True, True, True, True, False]


def test_swapping_two_points_reverses_coefficients_but_not_distances():
    p = plane_from_three_points(P0, P1, P2)
    rev = plane_from_three_points(P0, P2, P1)
    np.testing.assert_allclose(rev, -p, atol=1e-12)
    np.testing.assert_allclose(
        point_plane_distances(FIVE, rev), point_plane_distances(FIVE, p), atol=1e-12
    )


@pytest.mark.parametrize("factor", [2.0, -7.0, 0.5])
def test_unnormalized_and_sign_flipped_plane_has_same_distances(factor):
    p = plane_from_three_points(P0, P1, P2)
    expected = point_plane_distances(FIVE, p)
    actual = point_plane_distances(FIVE, factor * p)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("triple", [
    (P0, P0, P1),
    (np.array([0., 0., 1.]), np.array([0.1, 0., 1.]), np.array([0.2, 0., 1.])),
])
def test_repeated_or_collinear_points_rejected(triple):
    with pytest.raises(ValueError):
        plane_from_three_points(*triple)


@pytest.mark.parametrize("invalid", [
    np.array([0., 0.]),
    np.array([[0., 0., 1.]]),
    np.array([np.nan, 0., 1.]),
    np.array([np.inf, 0., 1.]),
])
def test_invalid_three_point_input_rejected(invalid):
    with pytest.raises(ValueError):
        plane_from_three_points(invalid, P1, P2)


@pytest.mark.parametrize("bad_points", [
    np.array([0., 0., 1.]),
    np.zeros((2, 2)),
    np.array([[0., 0., np.nan]]),
    np.array([[0., np.inf, 1.]]),
])
def test_invalid_distance_points_rejected(bad_points):
    with pytest.raises(ValueError):
        point_plane_distances(bad_points, [0, 0, 1, -1])


@pytest.mark.parametrize("bad_plane", [
    np.array([0., 0., 1.]),
    np.array([0., 0., 0., 1.]),
    np.array([np.nan, 0., 1., -1.]),
    np.array([np.inf, 0., 1., -1.]),
])
def test_invalid_plane_rejected(bad_plane):
    with pytest.raises(ValueError):
        point_plane_distances(FIVE, bad_plane)


def test_empty_pointcloud_yields_empty_distances():
    result = point_plane_distances(np.empty((0, 3)), [0, 0, 1, -1])
    assert result.shape == (0,)
    assert result.dtype == np.float64


def test_independent_oblique_plane_oracle():
    # Plane x + 2y + 2z - 6 = 0, normal length 3.
    # (0,0,3), (2,0,2), (0,1,2) all lie on it.
    a = [0., 0., 3.]
    b = [2., 0., 2.]
    c = [0., 1., 2.]
    plane = plane_from_three_points(a, b, c)
    np.testing.assert_allclose(np.linalg.norm(plane[:3]), 1.0, atol=1e-12)
    # Point (0,0,0) is at a known distance | -6 | / 3 = 2.
    np.testing.assert_allclose(
        point_plane_distances(np.array([[0., 0., 0.]]), plane), [2.], atol=1e-12
    )
    np.testing.assert_allclose(
        point_plane_distances(np.array([a, b, c]), plane), [0., 0., 0.], atol=1e-12
    )


def test_matches_open3d_three_point_model_geometry():
    o3d = pytest.importorskip("open3d")
    pts = np.stack((P0, P1, P2))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    from_o3d, idx = pcd.segment_plane(
        distance_threshold=.001, ransac_n=3, num_iterations=100, probability=1.0
    )
    assert len(idx) == 3
    from_numpy = plane_from_three_points(P0, P1, P2)
    # Normalize and align signs before comparing plane coefficients.
    from_o3d = np.asarray(from_o3d, dtype=float)
    from_o3d /= np.linalg.norm(from_o3d[:3])
    if np.dot(from_o3d[:3], from_numpy[:3]) < 0:
        from_o3d = -from_o3d
    np.testing.assert_allclose(from_o3d, from_numpy, rtol=0, atol=1e-9)
