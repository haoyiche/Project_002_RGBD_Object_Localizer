import numpy as np
import pytest
from src.evaluation.t1_04_a_pose_metrics import PoseM2C
from src.evaluation.t1_08_a_add50_audit import point_metrics_mm


def square():
    a = 0.05
    return np.array([[-a, -a, 0], [a, -a, 0],
                     [a, a, 0], [-a, a, 0]], dtype=float)


def test_half_turn_square():
    gt = PoseM2C(np.eye(3), np.zeros(3))
    est = PoseM2C(np.diag([-1.0, -1.0, 1.0]), np.zeros(3))
    add, adds = point_metrics_mm(square(), est, gt)
    assert add == pytest.approx(100 * np.sqrt(2))
    assert adds == pytest.approx(0.0)


def test_translation_only():
    gt = PoseM2C(np.eye(3), np.zeros(3))
    est = PoseM2C(np.eye(3), np.array([0.01, 0, 0]))
    add, adds = point_metrics_mm(square(), est, gt)
    assert add == pytest.approx(10)
    assert adds == pytest.approx(10)


def test_invalid_vertices_rejected():
    gt = PoseM2C(np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="INVALID_MODEL_VERTICES"):
        point_metrics_mm(np.zeros((0, 3)), gt, gt)


def test_nan_vertices_rejected():
    gt = PoseM2C(np.eye(3), np.zeros(3))
    pts = square()
    pts[0, 0] = np.nan
    with pytest.raises(ValueError, match="NONFINITE_MODEL_VERTICES"):
        point_metrics_mm(pts, gt, gt)


def test_adds_not_larger_than_add():
    gt = PoseM2C(np.eye(3), np.zeros(3))
    est = PoseM2C(np.diag([-1.0, -1.0, 1.0]), np.array([0.012, 0, 0]))
    add, adds = point_metrics_mm(square(), est, gt)
    assert adds <= add
