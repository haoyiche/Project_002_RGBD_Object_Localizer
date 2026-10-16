"""EXP034-C localization-contract tests: oracle, errors, and safe output rules."""
import json

import numpy as np
import pytest

from src.geometry.cluster_localization import (
    LocalizationGates,
    extract_publishable_positions,
    validate_cluster_localization,
)
from src.experiments.exp034_cluster_localization_minimal import (
    compare_with_open3d,
    make_scene,
)
from src.experiments.exp034_density_bias_b1 import make_density_biased_scene
from src.experiments.exp034_label_contamination_b2 import make_label_contaminated_scene


def _lookup(report):
    return {o['cluster_label']: o for o in report['objects']}


def test_b2_status_is_partial_and_counts_match():
    pts, labs = make_label_contaminated_scene()
    report = validate_cluster_localization(pts, labs)
    assert report['status'] == 'PARTIAL_INVALID'
    assert (report['valid_count'], report['invalid_count'], report['noise_count']) == (1, 1, 1)
    assert sorted(_lookup(report)) == [0, 1]
    assert report['frame_id'] == 'camera_optical_frame'


def test_invalid_A_has_no_publishable_position_but_keeps_diagnostics():
    pts, labs = make_label_contaminated_scene()
    obj = _lookup(validate_cluster_localization(pts, labs))[0]
    assert obj['status'] == 'INVALID_CLUSTER'
    assert obj['publishable'] is False
    assert obj['position_m'] is None
    assert obj['reasons'] == ['EXTENT_EXCEEDED', 'CENTROID_BBOX_GAP']
    np.testing.assert_allclose(obj['diagnostics']['centroid_m'], [0.099, 0.059, 1.160], atol=1e-12, rtol=0)
    np.testing.assert_allclose(obj['diagnostics']['extent_m'], [0.90, 0.50, 0.05], atol=1e-12, rtol=0)
    assert obj['diagnostics']['point_count'] == 10


def test_valid_B_can_still_publish_exact_oracle():
    pts, labs = make_label_contaminated_scene()
    report = validate_cluster_localization(pts, labs)
    B = _lookup(report)[1]
    assert B['status'] == 'VALID' and B['publishable'] is True
    assert B['reasons'] == []
    np.testing.assert_allclose(B['position_m'], [0.51, 0.01, 1.152], atol=1e-12, rtol=0)
    np.testing.assert_allclose(B['diagnostics']['extent_m'], [0.02, 0.02, 0.01], atol=1e-12, rtol=0)
    assert extract_publishable_positions(report) == [
        {'cluster_label': 1, 'frame_id': 'camera_optical_frame', 'position_m': B['position_m']}
    ]


def test_B1_before_contamination_is_all_valid():
    pts, labs = make_density_biased_scene()
    report = validate_cluster_localization(pts, labs)
    assert report['status'] == 'OK'
    assert report['valid_count'] == 2 and report['invalid_count'] == 0
    assert report['noise_count'] == 2
    assert [o['cluster_label'] for o in extract_publishable_positions(report)] == [0, 1]


def test_A_baseline_has_correct_centroid_and_box_center_gap():
    pts, labs = make_scene()
    A = _lookup(validate_cluster_localization(pts, labs))[0]
    np.testing.assert_allclose(A['position_m'], [0.01, 0.01, 1.152], atol=1e-12, rtol=0)
    np.testing.assert_allclose(A['diagnostics']['bbox_center_m'], [0.01, 0.01, 1.155], atol=1e-12, rtol=0)
    np.testing.assert_allclose(A['diagnostics']['centroid_bbox_distance_m'], 0.003, atol=1e-12, rtol=0)


def test_all_noise_returns_no_objects():
    pts = np.array([[0.0, 0.0, 1.1], [1.0, 1.0, 1.2]])
    report = validate_cluster_localization(pts, np.array([-1, -1]))
    assert report == {'frame_id': 'camera_optical_frame', 'status': 'NO_OBJECTS', 'objects': [],
                      'valid_count': 0, 'invalid_count': 0, 'noise_count': 2}
    assert extract_publishable_positions(report) == []


def test_explicit_empty_Nx3_integer_labels_returns_no_objects():
    report = validate_cluster_localization(np.empty((0, 3)), np.empty((0,), dtype=np.int64))
    assert report['status'] == 'NO_OBJECTS'
    assert report['noise_count'] == 0 and report['objects'] == []


@pytest.mark.parametrize('bad_points,bad_labels', [
    (np.array([[np.nan, 0.0, 1.0]]), np.array([0])),
    (np.array([[np.inf, 0.0, 1.0]]), np.array([0])),
    (np.array([[0.0, 0.0, 1.0]]), np.array([0, 0])),
    (np.array([0.0, 1.0, 2.0]), np.array([0])),
    (np.empty((1, 2)), np.array([0])),
    (np.array([[0.0, 0.0, 1.0]]), np.array([-5])),
    (np.array([[0.0, 0.0, 1.0]]), np.array([0.0])),
    (np.array([[0.0, 0.0, 1.0]]), np.array([True])),
    (np.array([[True, False, True]]), np.array([0])),
    (np.array([['x', 'y', 'z']]), np.array([0])),
])
def test_bad_input_raises_value_error_not_invalid_cluster(bad_points, bad_labels):
    with pytest.raises(ValueError):
        validate_cluster_localization(bad_points, bad_labels)


@pytest.mark.parametrize('bad_frame', ['', ' ', None, 17])
def test_frame_id_contract(bad_frame):
    pts, labs = make_scene()
    with pytest.raises(ValueError):
        validate_cluster_localization(pts, labs, frame_id=bad_frame)


@pytest.mark.parametrize('kwargs', [
    {'min_points': 0},
    {'max_points': 4, 'min_points': 5},
    {'min_points': True},
    {'max_points': 5.5},
    {'max_extent_m': [0.02, 0.01]},
    {'max_extent_m': [0.04, np.nan, 0.02]},
    {'max_extent_m': [-0.04, 0.04, 0.02]},
    {'max_centroid_bbox_gap_m': -0.01},
    {'max_centroid_bbox_gap_m': np.inf},
    {'max_centroid_bbox_gap_m': True},
])
def test_invalid_gate_config_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalizationGates(**kwargs)


def test_wrong_gate_instance_rejected():
    pts, labs = make_scene()
    with pytest.raises(ValueError):
        validate_cluster_localization(pts, labs, gates={})


def test_count_gate_rejects_small_cluster_and_no_position():
    pts = np.array([[0.0, 0.0, 1.0], [0.01, 0.0, 1.0]])
    report = validate_cluster_localization(pts, np.array([3, 3]))
    assert report['status'] == 'ALL_INVALID'
    item = report['objects'][0]
    assert item['cluster_label'] == 3
    assert item['reasons'] == ['POINT_COUNT_OUT_OF_RANGE']
    assert item['position_m'] is None
    assert extract_publishable_positions(report) == []


def test_stricter_extent_config_rejects_B_without_polluting_A():
    pts, labs = make_scene()
    custom = LocalizationGates(max_extent_m=(0.01, 0.01, 0.02))
    report = validate_cluster_localization(pts, labs, gates=custom)
    assert report['status'] == 'ALL_INVALID'
    assert all('EXTENT_EXCEEDED' in obj['reasons'] for obj in report['objects'])


def test_custom_frame_propagated_without_coordinate_transform():
    pts, labs = make_scene()
    report = validate_cluster_localization(pts, labs, frame_id='camera_test_frame')
    assert report['frame_id'] == 'camera_test_frame'
    assert all(obj['frame_id'] == 'camera_test_frame' for obj in report['objects'])
    np.testing.assert_allclose(report['objects'][0]['position_m'], [0.01, 0.01, 1.152], atol=1e-12)


def test_nonconsecutive_label_ids_sorted():
    pts, labs = make_scene()
    labels = np.where(labs == 0, 7, np.where(labs == 1, 4, -1))
    report = validate_cluster_localization(pts, labels)
    assert [obj['cluster_label'] for obj in report['objects']] == [4, 7]


def test_permutation_invariance_for_b2():
    pts, labs = make_label_contaminated_scene()
    old = validate_cluster_localization(pts, labs)
    perm = np.arange(len(pts))[::-1]
    new = validate_cluster_localization(pts[perm], labs[perm])
    for old_obj, new_obj in zip(old['objects'], new['objects']):
        assert old_obj['status'] == new_obj['status']
        assert old_obj['reasons'] == new_obj['reasons']
        assert old_obj['publishable'] == new_obj['publishable']
        for k in ('centroid_m', 'bbox_min_m', 'bbox_max_m', 'bbox_center_m', 'extent_m'):
            np.testing.assert_allclose(old_obj['diagnostics'][k], new_obj['diagnostics'][k], atol=1e-12, rtol=0)


def test_report_json_has_no_nan_and_invalid_position_serializes_to_null():
    pts, labs = make_label_contaminated_scene()
    result = validate_cluster_localization(pts, labs)
    text = json.dumps(result, allow_nan=False)
    restored = json.loads(text)
    assert restored['objects'][0]['position_m'] is None
    assert restored['objects'][1]['position_m'] is not None


def test_configured_policy_cannot_prove_true_object_center():
    pts, labs = make_density_biased_scene()
    A = _lookup(validate_cluster_localization(pts, labs))[0]
    assert A['status'] == 'VALID'
    np.testing.assert_allclose(A['position_m'][2], 10.4 / 9, atol=1e-12, rtol=0)
    assert abs(A['position_m'][2] - A['diagnostics']['bbox_center_m'][2]) > 5e-4
    # There is no solid-object ground truth in this experiment: 'VALID' != 3D accuracy.


def test_open3d_geometry_matches_validation_diagnostics():
    pytest.importorskip('open3d')
    pts, labs = make_label_contaminated_scene()
    report = validate_cluster_localization(pts, labs)
    raw_objects = [{'cluster_label': obj['cluster_label'], **obj['diagnostics']}
                   for obj in report['objects']]
    compare_with_open3d(pts, labs, raw_objects)
