"""EXP034-D: vary localization gate thresholds on an unchanged B2 scene.

Experimental independence: the point coordinates, labels, frame_id, and
geometry code are fixed. Only explicit validation limits change.
All thresholds are pedagogical examples, not production safety limits.
"""
import json
from pathlib import Path

import numpy as np

from src.geometry.cluster_localization import (
    LocalizationGates,
    extract_publishable_positions,
    validate_cluster_localization,
)
from src.experiments.exp034_label_contamination_b2 import make_label_contaminated_scene
from src.experiments.exp034_cluster_localization_minimal import compare_with_open3d


FRAME_ID = 'camera_optical_frame'
SCENARIOS = {
    'D0_default': LocalizationGates(),
    'D1_strict_gap': LocalizationGates(max_centroid_bbox_gap_m=0.002),
    'D2_relaxed_gap': LocalizationGates(max_centroid_bbox_gap_m=0.50),
    'D3_relaxed_both': LocalizationGates(
        max_extent_m=(1.0, 1.0, 0.10),
        max_centroid_bbox_gap_m=0.50,
    ),
}

# Independently reasoned oracle, including the literal public API status
# 'OK', not a made-up 'ALL_VALID' state.
EXPECTED = {
    'D0_default': {
        'status': 'PARTIAL_INVALID', 'published': [1],
        'A_reasons': ['EXTENT_EXCEEDED', 'CENTROID_BBOX_GAP'], 'B_reasons': [],
    },
    'D1_strict_gap': {
        'status': 'ALL_INVALID', 'published': [],
        'A_reasons': ['EXTENT_EXCEEDED', 'CENTROID_BBOX_GAP'],
        'B_reasons': ['CENTROID_BBOX_GAP'],
    },
    'D2_relaxed_gap': {
        'status': 'PARTIAL_INVALID', 'published': [1],
        'A_reasons': ['EXTENT_EXCEEDED'], 'B_reasons': [],
    },
    'D3_relaxed_both': {
        'status': 'OK', 'published': [0, 1],
        'A_reasons': [], 'B_reasons': [],
    },
}


def _by_label(report):
    return {item['cluster_label']: item for item in report['objects']}


def run_ablation(*, compare_open3d=True):
    points, labels = make_label_contaminated_scene()
    original_points = points.copy()
    original_labels = labels.copy()
    assert (points.shape, labels.shape) == ((16, 3), (16,))
    assert (np.count_nonzero(labels == 0), np.count_nonzero(labels == 1),
            np.count_nonzero(labels == -1)) == (10, 5, 1)
    print('Controlled variables: points, labels, frame_id unchanged: PASS')

    reports = {}
    reference_geometry = None
    for scenario, gates in SCENARIOS.items():
        report = validate_cluster_localization(points, labels, frame_id=FRAME_ID, gates=gates)
        published = extract_publishable_positions(report)
        ids = [item['cluster_label'] for item in published]
        expected = EXPECTED[scenario]
        by_label = _by_label(report)

        assert report['status'] == expected['status'], (scenario, report['status'])
        assert ids == expected['published'], (scenario, ids)
        assert by_label[0]['reasons'] == expected['A_reasons'], scenario
        assert by_label[1]['reasons'] == expected['B_reasons'], scenario
        assert report['noise_count'] == 1
        assert (report['valid_count'], report['invalid_count']) == (len(ids), 2-len(ids))
        assert all(obj['frame_id'] == FRAME_ID for obj in report['objects'])
        assert np.array_equal(points, original_points)
        assert np.array_equal(labels, original_labels)
        for obj in report['objects']:
            assert obj['publishable'] == (obj['status'] == 'VALID')
            assert (obj['position_m'] is not None) == obj['publishable']

        # Pin the geometry to an independent, hand-computed reference rather
        # than merely checking all scenarios agree with each other.
        np.testing.assert_allclose(by_label[0]['diagnostics']['centroid_m'],
                                   [0.099, 0.059, 1.160], rtol=0, atol=1e-12)
        np.testing.assert_allclose(by_label[0]['diagnostics']['extent_m'],
                                   [0.90, 0.50, 0.050], rtol=0, atol=1e-12)
        np.testing.assert_allclose(by_label[1]['diagnostics']['centroid_m'],
                                   [0.51, 0.01, 1.152], rtol=0, atol=1e-12)
        np.testing.assert_allclose(by_label[1]['diagnostics']['extent_m'],
                                   [0.02, 0.02, 0.01], rtol=0, atol=1e-12)

        geom = {label: by_label[label]['diagnostics'] for label in (0, 1)}
        if reference_geometry is None:
            reference_geometry = geom
        else:
            assert geom == reference_geometry, 'Geometry must not change with gates'

        # Persist exactly the configured limits used for reproducibility.
        reports[scenario] = {
            'gates': {
                'min_points': gates.min_points,
                'max_points': gates.max_points,
                'max_extent_m': list(gates.max_extent_m),
                'max_centroid_bbox_gap_m': gates.max_centroid_bbox_gap_m,
            },
            **report,
            'published_positions': published,
        }
        print(f"{scenario}: PASS | status={report['status']} | "
              f"A={by_label[0]['status']} {by_label[0]['reasons']} | "
              f"B={by_label[1]['status']} {by_label[1]['reasons']} | "
              f'published={ids}')

    print('D1 false rejection of clean B: VERIFIED')
    print('D2 extent gate independently blocks polluted A: VERIFIED')
    print('D3 polluted A published after gate relaxation: VERIFIED (unsafe false acceptance)')
    print('Geometry invariance across D0/D1/D2/D3: PASS')

    # Open3D checks only the fixed point geometry; it is NOT a validation of
    # engineering gate thresholds or of a real-world physical center.
    if compare_open3d:
        c0 = _by_label(reports['D0_default'])
        o3d_geometry = [
            {'cluster_label': i, **c0[i]['diagnostics']} for i in (0, 1)
        ]
        compare_with_open3d(points, labels, o3d_geometry)

    return {
        'experiment': 'EXP034-D',
        'frame_id': FRAME_ID,
        'input_point_count': len(points),
        'input_cluster_counts': {'A': 10, 'B': 5, 'noise': 1},
        'controlled_variable': 'Only LocalizationGates changes across scenarios.',
        'known_label_pollution': 'P11 is incorrectly assigned to cluster 0 in every scenario.',
        'warning': 'VALID and OK are policy acceptance, not proof of geometric truth or safety.',
        'scenarios': reports,
    }


def main():
    report = run_ablation(compare_open3d=True)
    print('\n=== EXP034-D GATE ABLATION SUMMARY ===')
    for name, result in report['scenarios'].items():
        print(f"{name}: {result['status']} | valid={result['valid_count']} "
              f"invalid={result['invalid_count']} published={len(result['published_positions'])}")
    out = Path('results/exp034/exp034_D_gate_ablation.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved:', out)
    print('EXP034-D EXPERIMENT VERIFIED; EXP034 UNIT COMPLETION: PENDING')


if __name__ == '__main__':
    main()
