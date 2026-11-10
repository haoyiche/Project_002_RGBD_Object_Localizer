import numpy as np
import pytest

from src.experiments.t1_07_a_e1_c_depth_noise import (
    sigma_depth_mm,
    centroid_std_mm,
)


def test_depth_noise_anchor_points():
    actual = sigma_depth_mm([1.0, 2.0])
    np.testing.assert_allclose(actual, [2.0, 8.0])


def test_near_and_far_centroid_std():
    assert np.isclose(
        centroid_std_mm(np.full(100, 1.0)), 0.2
    )
    assert np.isclose(
        centroid_std_mm(np.full(100, 2.0)), 0.8
    )
    assert np.isclose(
        centroid_std_mm(np.full(400, 2.0)), 0.4
    )


def test_mixed_depth_uses_individual_variances():
    depths = np.r_[np.ones(50), np.full(50, 2.0)]

    expected = np.sqrt(50 * 2**2 + 50 * 8**2) / 100
    observed = centroid_std_mm(depths)

    assert np.isclose(observed, expected)
    assert observed > 0.45


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [0.0],
        [-1.0],
        [float("nan")],
        [float("inf")],
        [[1.0]],
    ],
)
def test_invalid_depth_rejected(invalid):
    with pytest.raises(ValueError):
        centroid_std_mm(invalid)
