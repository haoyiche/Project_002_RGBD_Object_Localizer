"""EXP034-D independent, contract-level checks for three gate ablations."""
import json

import numpy as np
import pytest

from src.experiments.exp034_gate_ablation_d import (
    EXPECTED,
    SCENARIOS,
    run_ablation,
)
from src.experiments.exp034_label_contamination_b2 import make_label_contaminated_scene
from src.geometry.cluster_localization import (
    LocalizationGates,
    extract_publishable_positions,
    validate_cluster_localization,
)


def _scenario(name):
    pts, labels = make_label_contaminated_scene()
    report = validate_cluster_localization(pts, labels, gates=SCENARIOS[name])
    return report, {item['cluster_label']: item for item in report['objects']}


def test_input_is_fixed_B2_scene_with_only_one_noise_point():
    pts, labels = make_label_contaminated_scene()
    assert pts.shape == (16, 3) and labels.shape == (16,)
    assert np.count_nonzero(labels == 0) == 10
    assert np.count_nonzero(labels == 1) == 5
    assert np.count_nonzero(labels == -1) == 1
    np.testing.assert_allclose(pts[11], [0.90, 0.50, 1.20], rtol=0, atol=1e-12)
    assert labels[11] == 0 and labels[10] == -1


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_exact_status_and_publication_for_every_scenario(name):
    report, by_label = _scenario(name)
    expect = EXPECTED[name]
    assert report['status'] == expect['status']
    assert by_label[0]['reasons'] == expect['A_reasons']
    assert by_label[1]['reasons'] == expect['B_reasons']
    assert [x['cluster_label'] for x in extract_publishable_positions(report)] == expect['published']
    assert report['valid_count'] == len(expect['published'])
    assert report['invalid_count'] == 2 - len(expect['published'])
    assert report['noise_count'] == 1


def test_D1_clean_B_false_rejection_due_only_to_gap():
    report, by_label = _scenario('D1_strict_gap')
    assert by_label[1]['reasons'] == ['CENTROID_BBOX_GAP']
    np.testing.assert_allclose(by_label[1]['diagnostics']['centroid_bbox_distance_m'],
                               0.003, rtol=0, atol=1e-12)
    assert by_label[1]['position_m'] is None
    assert report['status'] == 'ALL_INVALID'


def test_D2_A_is_still_rejected_by_extent():
    _, by_label = _scenario('D2_relaxed_gap')
    assert by_label[0]['reasons'] == ['EXTENT_EXCEEDED']
    assert by_label[0]['position_m'] is None
    np.testing.assert_allclose(by_label[0]['diagnostics']['centroid_bbox_distance_m'],
                               np.sqrt(0.351**2+0.191**2+0.015**2), rtol=0, atol=1e-12)


def test_D3_accepts_known_contaminated_A_and_uses_OK_status():
    report, by_label = _scenario('D3_relaxed_both')
    assert report['status'] == 'OK'  # Existing public API, NOT 'ALL_VALID'.
    assert report['valid_count'] == 2 and report['invalid_count'] == 0
    assert by_label[0]['reasons'] == []
    assert by_label[0]['position_m'] is not None
    np.testing.assert_allclose(by_label[0]['position_m'], [0.099, 0.059, 1.160], rtol=0, atol=1e-12)
    assert [x['cluster_label'] for x in extract_publishable_positions(report)] == [0, 1]


def test_D0_baseline_still_matches_C_contract():
    report, by_label = _scenario('D0_default')
    assert report['status'] == 'PARTIAL_INVALID'
    assert by_label[0]['reasons'] == ['EXTENT_EXCEEDED', 'CENTROID_BBOX_GAP']
    assert by_label[1]['status'] == 'VALID'
    assert [x['cluster_label'] for x in extract_publishable_positions(report)] == [1]


def test_all_scenarios_have_identical_geometry_and_frame():
    reference = None
    for name in SCENARIOS:
        report, by_label = _scenario(name)
        assert report['frame_id'] == 'camera_optical_frame'
        geometry = {i: by_label[i]['diagnostics'] for i in (0, 1)}
        if reference is None:
            reference = geometry
        else:
            assert geometry == reference


def test_independent_manual_geometry_oracles():
    _, by_label = _scenario('D3_relaxed_both')
    np.testing.assert_allclose(by_label[0]['diagnostics']['centroid_m'],
                               [0.99/10, 0.59/10, 11.60/10], rtol=0, atol=1e-12)
    np.testing.assert_allclose(by_label[0]['diagnostics']['bbox_min_m'],
                               [0, 0, 1.150], rtol=0, atol=1e-12)
    np.testing.assert_allclose(by_label[0]['diagnostics']['bbox_max_m'],
                               [0.90, 0.50, 1.200], rtol=0, atol=1e-12)
    np.testing.assert_allclose(by_label[0]['diagnostics']['extent_m'],
                               [0.90, 0.50, 0.050], rtol=0, atol=1e-12)
    np.testing.assert_allclose(by_label[1]['diagnostics']['centroid_m'],
                               [0.51, 0.01, 1.152], rtol=0, atol=1e-12)


def test_thresholds_change_only_the_documented_values():
    base = SCENARIOS['D0_default']
    d1 = SCENARIOS['D1_strict_gap']
    d2 = SCENARIOS['D2_relaxed_gap']
    d3 = SCENARIOS['D3_relaxed_both']
    assert all(g.min_points == 5 and g.max_points == 20 for g in (base, d1, d2, d3))
    assert all(g.max_extent_m == (0.04, 0.04, 0.02) for g in (base, d1, d2))
    assert d1.max_centroid_bbox_gap_m == 0.002
    assert d2.max_centroid_bbox_gap_m == d3.max_centroid_bbox_gap_m == 0.50
    assert d3.max_extent_m == (1.0, 1.0, 0.10)


def test_D3_is_not_proof_of_accuracy():
    pts, labs = make_label_contaminated_scene()
    assert labs[11] == 0  # The label is deliberately wrong in every scenario.
    d3, _ = _scenario('D3_relaxed_both')
    assert d3['status'] == 'OK'
    assert any(x['cluster_label'] == 0 for x in extract_publishable_positions(d3))
    np.testing.assert_allclose(pts[11], [0.9, 0.5, 1.2], rtol=0, atol=1e-12)


def test_invalid_clusters_keep_diagnostics_but_no_position():
    for name in ('D0_default', 'D1_strict_gap', 'D2_relaxed_gap'):
        _, by_label = _scenario(name)
        for obj in by_label.values():
            if obj['status'] == 'INVALID_CLUSTER':
                assert obj['position_m'] is None
                assert obj['publishable'] is False
                assert 'centroid_m' in obj['diagnostics']


def test_ablation_full_report_is_json_serializable():
    result = run_ablation(compare_open3d=False)
    restored = json.loads(json.dumps(result, allow_nan=False))
    assert restored['experiment'] == 'EXP034-D'
    assert restored['scenarios']['D3_relaxed_both']['status'] == 'OK'
    assert len(restored['scenarios']['D3_relaxed_both']['published_positions']) == 2


def test_open3d_still_matches_fixed_scene_geometry():
    pytest.importorskip('open3d')
    result = run_ablation(compare_open3d=True)
    assert result['scenarios']['D0_default']['status'] == 'PARTIAL_INVALID'
