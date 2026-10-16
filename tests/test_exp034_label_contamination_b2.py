"""Independent regression tests for EXP034-B2 controlled-label experiment."""
import numpy as np
import pytest

from src.experiments.exp034_cluster_localization_minimal import (
    compare_with_open3d,
    localize_clusters,
)
from src.experiments.exp034_density_bias_b1 import make_density_biased_scene
from src.experiments.exp034_label_contamination_b2 import (
    P11_INDEX,
    build_report,
    demo_gates,
    make_label_contaminated_scene,
)


def _objects(points, labels):
    return {o['cluster_label']: o for o in localize_clusters(points, labels)}


def test_only_p11_label_changes_not_coordinates_or_any_other_label():
    initial_pts, initial_labels = make_density_biased_scene()
    pts, labels = make_label_contaminated_scene()
    np.testing.assert_array_equal(pts, initial_pts)
    np.testing.assert_array_equal(np.flatnonzero(labels != initial_labels), [P11_INDEX])
    assert initial_labels[11] == -1 and labels[11] == 0
    assert labels[10] == -1


def test_exact_counts_and_two_clusters():
    pts, labs = make_label_contaminated_scene()
    assert len(pts) == 16
    obj = _objects(pts, labs)
    assert sorted(obj) == [0, 1]
    assert (obj[0]['point_count'], obj[1]['point_count']) == (10, 5)
    assert np.count_nonzero(labs == -1) == 1


def test_centroid_hand_oracle():
    pts, labs = make_label_contaminated_scene()
    obj = _objects(pts, labs)
    np.testing.assert_allclose(obj[0]['centroid_m'], [0.099, 0.059, 1.160], atol=1e-12, rtol=0)


def test_centroid_shift_is_89_49_and_4p444_mm():
    before_pts, before_lab = make_density_biased_scene()
    pts, labs = make_label_contaminated_scene()
    old = _objects(before_pts, before_lab)[0]
    new = _objects(pts, labs)[0]
    delta_mm = (np.array(new['centroid_m']) - np.array(old['centroid_m'])) * 1000
    np.testing.assert_allclose(delta_mm, [89.0, 49.0, 40/9], atol=1e-9, rtol=0)


def test_bounds_and_bbox_center_oracle():
    pts, labs = make_label_contaminated_scene()
    A = _objects(pts, labs)[0]
    np.testing.assert_allclose(A['bbox_min_m'], [0, 0, 1.15], atol=1e-12, rtol=0)
    np.testing.assert_allclose(A['bbox_max_m'], [0.90, 0.50, 1.20], atol=1e-12, rtol=0)
    np.testing.assert_allclose(A['bbox_center_m'], [0.45, 0.25, 1.175], atol=1e-12, rtol=0)
    np.testing.assert_allclose(A['extent_m'], [0.90, 0.50, 0.05], atol=1e-12, rtol=0)


def test_extent_expansion_factors():
    before_pts, before_lab = make_density_biased_scene()
    pts, labs = make_label_contaminated_scene()
    before = _objects(before_pts, before_lab)[0]
    after = _objects(pts, labs)[0]
    ratio = np.asarray(after['extent_m']) / np.asarray(before['extent_m'])
    np.testing.assert_allclose(ratio, [45.0, 25.0, 5.0], atol=1e-10, rtol=0)


def test_unaffected_control_object_b():
    old_pts, old_labs = make_density_biased_scene()
    pts, labs = make_label_contaminated_scene()
    old_b, new_b = _objects(old_pts, old_labs)[1], _objects(pts, labs)[1]
    assert old_b == new_b


def test_count_gate_misses_while_two_geometry_gates_catch():
    old_pts, old_labs = make_density_biased_scene()
    pts, labs = make_label_contaminated_scene()
    before = _objects(old_pts, old_labs)[0]
    after = _objects(pts, labs)[0]
    before_gates = demo_gates(before, before)
    after_gates = demo_gates(before, after)
    assert before_gates['point_count_gate_5_to_20_pass']
    assert after_gates['point_count_gate_5_to_20_pass']
    assert before_gates['extent_gate_max_twice_B1_pass']
    assert before_gates['centroid_bbox_distance_gate_0p02m_pass']
    assert not after_gates['extent_gate_max_twice_B1_pass']
    assert not after_gates['centroid_bbox_distance_gate_0p02m_pass']
    assert after_gates['centroid_to_bbox_center_distance_m'] > 0.39


def test_point_permutation_does_not_change_the_statistics():
    pts, labs = make_label_contaminated_scene()
    first = _objects(pts, labs)
    order = np.arange(len(pts))[::-1]
    second = _objects(pts[order], labs[order])
    for idx in (0, 1):
        for field in ('centroid_m', 'bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
            np.testing.assert_allclose(first[idx][field], second[idx][field], atol=1e-12, rtol=0)
        assert first[idx]['point_count'] == second[idx]['point_count']


def test_full_report_self_consistency():
    result, pts, labs = build_report()
    assert result['noise_count_before'] == 2
    assert result['noise_count_after'] == 1
    assert len(result['objects_after']) == 2
    assert result['objects_after'][0]['point_count'] == 10
    np.testing.assert_allclose(result['A_centroid_shift_mm'], [89, 49, 40/9], atol=1e-9, rtol=0)
    assert len(pts) == len(labs) == 16


def test_open3d_comparison_on_polluted_scene():
    pytest.importorskip('open3d')
    pts, labs = make_label_contaminated_scene()
    compare_with_open3d(pts, labs, list(_objects(pts, labs).values()))
