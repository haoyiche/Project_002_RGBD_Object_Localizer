"""EXP034-C smoke demo: validate B2 scene and compare raw geometry to Open3D."""
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


def run_experiment():
    points, labels = make_label_contaminated_scene()
    gates = LocalizationGates()  # B2 demonstration limits; NOT production limits.
    report = validate_cluster_localization(points, labels, gates=gates)
    by_label = {obj['cluster_label']: obj for obj in report['objects']}
    assert report['status'] == 'PARTIAL_INVALID'
    assert (report['valid_count'], report['invalid_count'], report['noise_count']) == (1, 1, 1)
    assert by_label[0]['status'] == 'INVALID_CLUSTER'
    assert by_label[0]['publishable'] is False and by_label[0]['position_m'] is None
    assert by_label[0]['reasons'] == ['EXTENT_EXCEEDED', 'CENTROID_BBOX_GAP']
    assert by_label[1]['status'] == 'VALID' and by_label[1]['publishable'] is True
    np.testing.assert_allclose(by_label[1]['position_m'], [0.51, 0.01, 1.152], atol=1e-12, rtol=0)
    published = extract_publishable_positions(report)
    assert [o['cluster_label'] for o in published] == [1]
    print('C1 independent per-cluster gate: PASS')
    print('C2 invalid position is null, diagnostics retained: PASS')

    # Geometry comparison includes the rejected cluster: its diagnostics are
    # calculated correctly even though publishing them as position is forbidden.
    raw_geometry = []
    for item in report['objects']:
        raw_geometry.append({'cluster_label': item['cluster_label'], **item['diagnostics']})
    compare_with_open3d(points, labels, raw_geometry)

    noise_pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    all_noise = validate_cluster_localization(noise_pts, np.array([-1, -1]))
    assert (all_noise['status'], all_noise['objects'], all_noise['noise_count']) == ('NO_OBJECTS', [], 2)
    print('C3 all-noise valid input: PASS')

    for bad_pts, bad_labels in (
        (np.array([[np.nan, 0.0, 1.0]]), np.array([0])),
        (np.array([[0.0, 0.0, 1.0]]), np.array([0, 0])),
    ):
        try:
            validate_cluster_localization(bad_pts, bad_labels)
        except ValueError:
            pass
        else:
            raise AssertionError('C4 invalid API input must raise ValueError')
    print('C4 input-contract failures: PASS')
    report['published_positions'] = published
    report['gate_note'] = ('Passing these demonstration gates does NOT prove '
                           'physical accuracy, correct object identity, or safe robot motion.')
    print('C5 acceptance vs accuracy distinction: PASS')
    return report


def main():
    report = run_experiment()
    print('\n=== EXP034-C VALIDATED LOCALIZATION ===')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    out = Path('results/exp034/exp034_C_validated_localization.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Saved:', out)
    print('EXP034-C EXPERIMENT VERIFIED. EXP034 UNIT COMPLETION: PENDING')


if __name__ == '__main__':
    main()
