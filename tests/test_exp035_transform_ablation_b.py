"""EXP035-B: independent prediction checks, controlled variables, frame chain."""

import importlib.util

import numpy as np
import pytest

from src.experiments.exp035_se3_minimal import (
    EXPECTED_CAMERA_M,
    make_homogeneous,
    make_synthetic_transform,
)
from src.experiments.exp035_transform_ablation_b import (
    ORACLE_B0_BASE_M,
    ORACLE_B1_BASE_M,
    ORACLE_B1_DELTA_M,
    ORACLE_B2_BASE_M,
    ORACLE_NORM_SQUARED_M2,
    ORACLE_REVERSED_DEMO_M,
    ORACLE_WORLD_FROM_CAMERA_R,
    ORACLE_WORLD_FROM_CAMERA_T_M,
    ORACLE_WORLD_M,
    compare_with_open3d,
    compute_variants,
    make_world_from_base,
    serialize_case,
    verify_hand_oracles,
)

TOL = 1e-12


@pytest.fixture
def case():
    return compute_variants(EXPECTED_CAMERA_M)


def test_original_input_unchanged(case):
    np.testing.assert_array_equal(case["p_camera_m"], EXPECTED_CAMERA_M)
    assert case["p_camera_m"] is not EXPECTED_CAMERA_M


def test_reference_rotation_translation_unchanged():
    R, t = make_synthetic_transform()
    np.testing.assert_array_equal(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(t, [0.30, -0.10, 0.20], rtol=0, atol=TOL)


def test_b0_preserved(case):
    np.testing.assert_allclose(case["B0"]["p_base_m"], ORACLE_B0_BASE_M, rtol=0, atol=TOL)


def test_b1_identity_rotation(case):
    np.testing.assert_allclose(case["B1"]["T_base_from_camera"][:3, :3],
                               np.eye(3), atol=TOL, rtol=0)


def test_b1_translation_only_hand_oracle(case):
    np.testing.assert_allclose(case["B1"]["p_base_m"], ORACLE_B1_BASE_M, atol=TOL, rtol=0)


def test_b1_difference_from_a_hand_oracle(case):
    np.testing.assert_allclose(case["B1"]["delta_vs_A_m"], ORACLE_B1_DELTA_M, atol=TOL, rtol=0)


def test_b1_inverse_subtracts_translation(case):
    np.testing.assert_allclose(case["B1"]["p_camera_roundtrip_m"], EXPECTED_CAMERA_M,
                               atol=TOL, rtol=0)


def test_b2_zero_translation(case):
    np.testing.assert_array_equal(case["B2"]["T_base_from_camera"][:3, 3], np.zeros(3))


def test_b2_rotation_only_hand_oracle(case):
    np.testing.assert_allclose(case["B2"]["p_base_m"], ORACLE_B2_BASE_M, atol=TOL, rtol=0)


def test_b2_squared_norm_invariance(case):
    assert case["B2"]["norm_squared_camera_m2"] == pytest.approx(ORACLE_NORM_SQUARED_M2, abs=TOL)
    assert case["B2"]["norm_squared_rotated_m2"] == pytest.approx(ORACLE_NORM_SQUARED_M2, abs=TOL)


def test_b2_inverse_rotates_back(case):
    np.testing.assert_allclose(case["B2"]["p_camera_roundtrip_m"], EXPECTED_CAMERA_M,
                               atol=TOL, rtol=0)


def test_b3_second_transform_parameters():
    R, t = make_world_from_base()
    np.testing.assert_array_equal(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(t, [-0.20, 0.40, 0], atol=TOL, rtol=0)


def test_b3_sequential_world_hand_oracle(case):
    np.testing.assert_allclose(case["B3"]["p_world_sequential_m"], ORACLE_WORLD_M,
                               atol=TOL, rtol=0)


def test_b4_composite_rotation_hand_oracle(case):
    np.testing.assert_allclose(case["B3"]["R_world_from_camera"],
                               ORACLE_WORLD_FROM_CAMERA_R, atol=TOL, rtol=0)


def test_b4_composite_translation_hand_oracle(case):
    np.testing.assert_allclose(case["B3"]["t_world_from_camera_m"],
                               ORACLE_WORLD_FROM_CAMERA_T_M, atol=TOL, rtol=0)


def test_b4_homogeneous_vs_sequential(case):
    np.testing.assert_allclose(case["B3"]["p_world_homogeneous_m"], ORACLE_WORLD_M,
                               atol=TOL, rtol=0)
    np.testing.assert_allclose(case["B3"]["p_world_composed_m"], ORACLE_WORLD_M,
                               atol=TOL, rtol=0)


def test_b4_matrix_order(case):
    Rbc, tbc = make_synthetic_transform()
    Rwb, twb = make_world_from_base()
    expected = make_homogeneous(Rwb, twb) @ make_homogeneous(Rbc, tbc)
    np.testing.assert_allclose(case["B3"]["T_world_from_camera"], expected,
                               atol=TOL, rtol=0)


def test_b4_wrong_order_differs_by_one_meter(case):
    np.testing.assert_allclose(case["B3"]["p_reversed_wrong_demo_m"],
                               ORACLE_REVERSED_DEMO_M, atol=TOL, rtol=0)
    np.testing.assert_allclose(case["B3"]["p_reversed_wrong_demo_m"] -
                               case["B3"]["p_world_sequential_m"],
                               [0, -1, 0], atol=TOL, rtol=0)


def test_rotation_blocks_commute_but_full_transforms_do_not(case):
    a = case["B3"]["T_world_from_camera"]
    b = case["B3"]["T_reversed_wrong_demo"]
    np.testing.assert_allclose(a[:3, :3], b[:3, :3], atol=TOL, rtol=0)
    assert not np.allclose(a, b, atol=TOL, rtol=0)


def test_b3_composed_inverse_round_trip(case):
    np.testing.assert_allclose(case["B3"]["p_camera_roundtrip_m"], EXPECTED_CAMERA_M,
                               atol=TOL, rtol=0)


def test_prediction_oracles_all_pass(case):
    assert verify_hand_oracles(case) is True


def test_b5_hypothetical_unverified_blocks_publication(case):
    data = serialize_case(case)
    chain = data["B5_hypothetical_chain"]
    assert [edge["verified"] for edge in chain["transforms"]] == [True, False]
    assert chain["status"] == "BLOCKED_UNCALIBRATED"
    assert chain["publishable"] is False and chain["position_m"] is None
    assert chain["reasons"] == ["UNVERIFIED_TRANSFORM: robot_base -> world"]


def test_all_actual_transforms_are_synthetic_not_publishable(case):
    data = serialize_case(case)
    assert data["calibration_verified"] is False
    assert data["publishable"] is False and data["position_m"] is None
    assert "SYNTHETIC" in data["warning"]


@pytest.mark.parametrize("invalid", [
    [1.0, 2.0],
    [[1.0, 2.0, 3.0]],
    [float("nan"), 0.0, 0.0],
    [0.0, float("inf"), 0.0],
])
def test_reject_bad_point_shape_or_value(invalid):
    with pytest.raises(ValueError):
        compute_variants(invalid)


@pytest.mark.skipif(importlib.util.find_spec("open3d") is None, reason="Open3D not installed")
def test_open3d_matches_three_variants(case):
    assert compare_with_open3d(case) is True
