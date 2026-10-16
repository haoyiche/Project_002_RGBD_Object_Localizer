"""EXP034-B1 controlled experiment tests (no live sensor needed)."""
import numpy as np

from src.experiments.exp034_cluster_localization_minimal import localize_clusters, make_scene
from src.experiments.exp034_density_bias_b1 import make_density_biased_scene


def _objects(points, labels):
    return {x['cluster_label']: x for x in localize_clusters(points, labels)}


def test_b1_retains_original_points_and_labels():
    before_p, before_l = make_scene()
    after_p, after_l = make_density_biased_scene()
    assert after_p.shape == (16, 3)
    assert after_l.shape == (16,)
    np.testing.assert_array_equal(after_p[:12], before_p)
    np.testing.assert_array_equal(after_l[:12], before_l)


def test_b1_only_adds_four_identical_points_to_a():
    points, labels = make_density_biased_scene()
    np.testing.assert_array_equal(points[12:], np.tile([0.01, 0.01, 1.160], (4, 1)))
    np.testing.assert_array_equal(labels[12:], [0, 0, 0, 0])
    assert np.count_nonzero(labels == 0) == 9
    assert np.count_nonzero(labels == 1) == 5


def test_b1_keeps_noise_labels_and_two_objects():
    points, labels = make_density_biased_scene()
    np.testing.assert_array_equal(np.flatnonzero(labels == -1), [10, 11])
    assert set(_objects(points, labels)) == {0, 1}


def test_b1_hand_derived_new_centroid():
    points, labels = make_density_biased_scene()
    a = _objects(points, labels)[0]
    independent_expected = [0.09 / 9, 0.09 / 9, 10.4 / 9]
    np.testing.assert_allclose(a['centroid_m'], independent_expected, rtol=0, atol=1e-12)
    assert a['point_count'] == 9


def test_b1_aabb_is_unchanged_in_all_four_fields():
    bp, bl = make_scene()
    ap, al = make_density_biased_scene()
    before = _objects(bp, bl)[0]
    after = _objects(ap, al)[0]
    for field in ('bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
        np.testing.assert_allclose(after[field], before[field], rtol=0, atol=1e-12)
    np.testing.assert_allclose(after['extent_m'], [0.02, 0.02, 0.01], rtol=0, atol=1e-12)


def test_b1_centroid_crosses_aabb_center():
    bp, bl = make_scene()
    ap, al = make_density_biased_scene()
    before = _objects(bp, bl)[0]
    after = _objects(ap, al)[0]
    assert before['centroid_m'][2] < before['bbox_center_m'][2]
    assert after['centroid_m'][2] > after['bbox_center_m'][2]
    shift_mm = (after['centroid_m'][2] - before['centroid_m'][2]) * 1000
    np.testing.assert_allclose(shift_mm, 32/9, atol=1e-10, rtol=0)


def test_b1_object_b_is_unchanged():
    bp, bl = make_scene()
    ap, al = make_density_biased_scene()
    b0 = _objects(bp, bl)[1]
    b1 = _objects(ap, al)[1]
    assert b0['point_count'] == b1['point_count'] == 5
    for field in ('centroid_m', 'bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
        np.testing.assert_allclose(b1[field], b0[field], rtol=0, atol=1e-12)


def test_b1_permutation_does_not_change_geometry():
    points, labels = make_density_biased_scene()
    order = np.array([15, 0, 10, 6, 13, 11, 4, 7, 12, 3, 8, 5, 1, 14, 2, 9])
    direct = _objects(points, labels)
    permuted = _objects(points[order], labels[order])
    for cluster_id in (0, 1):
        assert direct[cluster_id]['point_count'] == permuted[cluster_id]['point_count']
        for field in ('centroid_m', 'bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
            np.testing.assert_allclose(direct[cluster_id][field], permuted[cluster_id][field], rtol=0, atol=1e-12)
