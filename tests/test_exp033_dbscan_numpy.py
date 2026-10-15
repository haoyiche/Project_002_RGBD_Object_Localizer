"""EXP033-D independent acceptance tests for the learner's NumPy DBSCAN.

Contract:
  points: (N, 3), finite, coordinates in meters
  eps: finite positive radius in meters
  min_points: positive integer (bool is not an integer for this API)
  output: (N,) np.int64, noise=-1, cluster labels >=0

Run from P02 root: python -m pytest -v tests/test_exp033_dbscan_numpy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src" / "experiments"))

from exp033_dbscan_numpy import dbscan_numpy


def assert_partition(labels, expected_groups, expected_noise):
    """Compare exact partitions without relying on arbitrary cluster IDs.

    Noise gets its own explicit check: treating every -1 as a regular cluster
    would hide an important class of segmentation errors.
    """
    labels = np.asarray(labels)
    assert labels.dtype == np.dtype("int64"), f"expected int64, got {labels.dtype}"
    assert labels.ndim == 1
    actual_noise = set(np.flatnonzero(labels == -1).tolist())
    assert actual_noise == set(expected_noise)
    ids = sorted(int(x) for x in np.unique(labels) if x >= 0)
    assert ids == list(range(len(ids))), "cluster IDs must be consecutive from 0"
    actual = {frozenset(np.flatnonzero(labels == cid).tolist()) for cid in ids}
    expected = {frozenset(group) for group in expected_groups}
    assert actual == expected


def scene(b_x=0.50):
    a = np.array([
        [0.00, 0.00, 1.15],
        [0.02, 0.00, 1.15],
        [0.00, 0.02, 1.15],
        [0.02, 0.02, 1.15],
        [0.01, 0.01, 1.16],
    ], dtype=np.float64)
    b = a.copy()
    b[:, 0] += b_x
    clutter = np.array([
        [0.25, 0.25, 1.15],
        [0.90, 0.50, 1.20],
    ], dtype=np.float64)
    return np.vstack((a, b, clutter))


@pytest.mark.parametrize("eps,min_points,groups,noise", [
    (0.060, 3, [range(0, 5), range(5, 10)], [10, 11]),
    (0.015, 3, [], list(range(12))),
    (0.018, 3, [range(0, 5), range(5, 10)], [10, 11]),
    (0.018, 5, [range(0, 5), range(5, 10)], [10, 11]),
    (0.018, 6, [], list(range(12))),
])
def test_abc_baseline_oracles(eps, min_points, groups, noise):
    labels = dbscan_numpy(scene(), eps=eps, min_points=min_points)
    assert labels.shape == (12,)
    assert_partition(labels, groups, noise)


@pytest.mark.parametrize("b_x,groups", [
    (0.040, [range(0, 5), range(5, 10)]),
    (0.035, [range(0, 10)]),
])
def test_c1_c2_bridge_oracles(b_x, groups):
    labels = dbscan_numpy(scene(b_x), eps=0.018, min_points=3)
    assert_partition(labels, groups, [10, 11])


def test_provisional_noise_is_absorbed_by_later_core():
    points = np.array([[0.00, 0, 0], [0.01, 0, 0], [0.02, 0, 0]])
    labels = dbscan_numpy(points, eps=0.011, min_points=3)
    assert_partition(labels, [[0, 1, 2]], [])


def test_border_does_not_expand_to_other_noncore_points():
    # First 4 points form a dense core; P4 is a border point; P5 has only P4
    # as a neighbor. P5 must stay noise, otherwise the implementation
    # incorrectly expands through a border point.
    points = np.array([
        [0.00,  0.00, 0],
        [0.00,  0.10, 0],
        [0.00, -0.10, 0],
        [0.10,  0.00, 0],
        [1.00,  0.00, 0],
        [1.85,  0.00, 0],
    ])
    labels = dbscan_numpy(points, eps=0.95, min_points=4)
    assert_partition(labels, [[0, 1, 2, 3, 4]], [5])


def test_exact_eps_boundary_is_included():
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    labels = dbscan_numpy(points, eps=1.0, min_points=2)
    assert_partition(labels, [[0, 1]], [])


def test_one_point_min_points_one_is_its_own_cluster():
    labels = dbscan_numpy(np.array([[0, 0, 0.5]]), eps=0.1, min_points=1)
    assert_partition(labels, [[0]], [])


def test_one_point_min_points_three_is_noise():
    labels = dbscan_numpy(np.array([[0, 0, 0.5]]), eps=0.1, min_points=3)
    assert_partition(labels, [], [0])


def test_empty_point_cloud_returns_int64_empty():
    labels = dbscan_numpy(np.empty((0, 3)), eps=0.01, min_points=3)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (0,)
    assert labels.dtype == np.int64


@pytest.mark.parametrize("bad_points", [
    np.array([0., 0., 0.]),
    np.zeros((3, 2)),
    np.zeros((2, 3, 1)),
    np.array([[0.0, np.nan, 0.0]]),
    np.array([[0.0, np.inf, 0.0]]),
])
def test_bad_point_shape_and_nonfinite_rejected(bad_points):
    with pytest.raises(ValueError):
        dbscan_numpy(bad_points, eps=0.01, min_points=3)


@pytest.mark.parametrize("bad_eps", [0.0, -1.0, float("nan"), float("inf")])
def test_bad_eps_rejected(bad_eps):
    with pytest.raises(ValueError):
        dbscan_numpy(np.zeros((1, 3)), eps=bad_eps, min_points=3)


@pytest.mark.parametrize("bad_min_points", [0, -1, 1.5, True, False])
def test_bad_min_points_rejected(bad_min_points):
    with pytest.raises(ValueError):
        dbscan_numpy(np.zeros((1, 3)), eps=0.1, min_points=bad_min_points)


def test_np_integer_min_points_is_accepted():
    labels = dbscan_numpy(np.array([[0.0, 0.0, 0.0]]), eps=0.1,
                          min_points=np.int64(1))
    assert_partition(labels, [[0]], [])


def test_unambiguous_results_agree_with_open3d():
    o3d = pytest.importorskip("open3d")
    points = scene()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    open3d_labels = np.asarray(
        pcd.cluster_dbscan(eps=0.018, min_points=3, print_progress=False),
        dtype=np.int64,
    )
    own_labels = dbscan_numpy(points, eps=0.018, min_points=3)

    # In this specific synthetic scene there are no ambiguous border points.
    # Compare only partitions and noise masks, not numeric cluster IDs.
    for labels in (own_labels, open3d_labels):
        assert_partition(labels, [range(0, 5), range(5, 10)], [10, 11])
