"""EXP032-D: independent numerical-degeneracy regression tests."""

import numpy as np
import pytest

from src.experiments.exp032_plane_numpy import (
    plane_from_three_points,
    point_plane_distances,
)


def test_tiny_perpendicular_triangle_is_valid():
    p = plane_from_three_points(
        [0.0, 0.0, 0.0], [1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0]
    )
    np.testing.assert_allclose(p, [0.0, 0.0, 1.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(p[:3]), 1.0, rtol=0, atol=1e-12)


def test_large_near_collinear_triangle_is_rejected():
    with pytest.raises(ValueError):
        plane_from_three_points(
            [0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 1e-10, 0.0]
        )


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_right_triangle_is_valid_at_multiple_scales(scale):
    pts = np.array(
        [[0.0, 0.0, 0.0], [1e-7, 0.0, 0.0], [0.0, 1e-7, 0.0]]
    ) * scale
    p = plane_from_three_points(*pts)
    np.testing.assert_allclose(p, [0.0, 0.0, 1.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(point_plane_distances(pts, p), 0.0, rtol=0, atol=1e-12)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_near_collinear_is_rejected_at_multiple_scales(scale):
    pts = np.array(
        [[0.0, 0.0, 0.0], [1000.0, 0.0, 0.0], [1000.0, 1e-10, 0.0]]
    ) * scale
    with pytest.raises(ValueError):
        plane_from_three_points(*pts)


def test_p1_equals_p2_is_rejected():
    with pytest.raises(ValueError):
        plane_from_three_points([0., 0., 0.], [1., 2., 3.], [1., 2., 3.])


@pytest.mark.parametrize("offset,should_reject", [(1e-7, False), (1e-9, True)])
def test_relative_sine_threshold(offset, should_reject):
    args = ([0., 0., 0.], [1., 0., 0.], [1., offset, 0.])
    if should_reject:
        with pytest.raises(ValueError):
            plane_from_three_points(*args)
    else:
        p = plane_from_three_points(*args)
        np.testing.assert_allclose(np.linalg.norm(p[:3]), 1.0, atol=1e-12)
