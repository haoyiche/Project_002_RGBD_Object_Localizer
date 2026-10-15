
"""EXP033-E1: numerical stability regression tests."""

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(PROJECT / "src" / "experiments"),
)

from exp033_dbscan_numpy import dbscan_numpy


def test_large_finite_distance():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1e155, 0.0, 0.0],
    ])

    labels = dbscan_numpy(
        points,
        eps=2e155,
        min_points=2,
    )

    # 两点相距 1e155，小于 eps。
    # 应属于同一簇。
    assert labels.tolist() == [0, 0]


def test_tiny_finite_distance():
    points = np.array([
        [0.0, 0.0, 0.0],
        [1e-200, 0.0, 0.0],
    ])

    labels = dbscan_numpy(
        points,
        eps=5e-201,
        min_points=2,
    )

    # 两点相距 1e-200，大于 eps。
    # 两点均应为噪声。
    assert labels.tolist() == [-1, -1]
