"""T1-04-B5 tests use synthetic / monkeypatched observations, NOT LM GT."""
import json

import numpy as np
import pytest

from src.evaluation import t1_04_b5_diagnostic_ablations as m
from src.evaluation import t1_04_b_gt_free_icp as b1
from src.evaluation import t1_04_b2_independent_pose_eval as b2
from src.evaluation.t1_04_a_pose_metrics import PoseM2C


def stubbed(tmp_path, monkeypatch):
    points = np.array([[i*0.001, (i*i % 31)*0.001, 1+0.001*i]
                       for i in range(35)], dtype=float)
    calls = []
    def obs(scene_dir, *, im_id, gt_index):
        calls.append(('observation', im_id))
        return points, {
            'im_id': im_id, 'gt_index': gt_index,
            'sha256': {x: 'a'*64 for x in ('scene_camera','depth','rgb','mask_visib')},
            'oracle_mask_used': True, 'gt_pose_read_by_predictor': False,
        }
    def load(models_dir, obj_id):
        calls.append(('model', obj_id))
        return points, {
            'obj_id': obj_id, 'model_units_input': 'mm',
            'model_units_internal': 'm',
            'mesh_sha256': 'a'*64, 'models_info_sha256': 'b'*64,
        }
    def fit(model, observed, *, max_iterations, trim_fraction, support_threshold_m):
        calls.append(('fit', max_iterations, trim_fraction, support_threshold_m))
        # Pretend trim and iteration affect pose; support-only has no effect.
        t = np.array([0, 0, 1], dtype=float)
        t[0] = (0.75 - trim_fraction) * 0.01 + (max_iterations-18)*0.00001
        support = 0.60 if support_threshold_m == .004 else 0.85
        return PoseM2C(np.eye(3), t), {
            'trimmed_rmse_m': .003 + (0.75 - trim_fraction) * .001,
            'support_fraction': support,
            'support_threshold_m': support_threshold_m,
            'median_nn_distance_m': .002,
            'num_model_points': 35, 'num_observed_points': 35,
        }
    monkeypatch.setattr(b1, 'read_oracle_observation', obs)
    monkeypatch.setattr(b1, 'load_model_points', load)
    monkeypatch.setattr(b1, 'fit_pca_multistart_icp', fit)
    out = tmp_path / 'run'
    return calls, out


def test_ablation_design_one_factor_at_a_time():
    base = m.VARIANTS['baseline']
    assert len(m.VARIANTS) == 4
    for variant in ('trim_060','iter_036','support_004'):
        diff = [k for k in base if base[k] != m.VARIANTS[variant][k]]
        assert len(diff) == 1


def test_freeze_generated_only_after_all_predictions(tmp_path, monkeypatch):
    calls, out = stubbed(tmp_path, monkeypatch)
    freeze = m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                             run_dir=out, cases=(15,19))
    assert len(freeze['records']) == 8
    assert freeze['prediction_gt_pose_access'] is False
    assert len(list(out.glob('candidate_*.json'))) == 8
    assert m.audit_frozen(out) == freeze
    assert not list(out.glob('evaluation_*.json'))
    assert len([c for c in calls if c[0]=='fit']) == 8


def test_candidate_tampering_rejected_before_evaluation(tmp_path, monkeypatch):
    _, out = stubbed(tmp_path, monkeypatch)
    m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                    run_dir=out, cases=(15,))
    (out/'candidate_000015_trim_060.json').write_text('tampered', encoding='utf8')
    evaluated = []
    monkeypatch.setattr(b2, 'evaluate_saved_candidate',
                        lambda **kw: evaluated.append(kw))
    with pytest.raises(ValueError, match='FROZEN_CANDIDATE_SHA256_MISMATCH'):
        m.evaluate_cases(scene_dir=tmp_path, models_dir=tmp_path, run_dir=out)
    assert evaluated == []


def test_changed_source_hash_rejected(tmp_path, monkeypatch):
    _, out = stubbed(tmp_path, monkeypatch)
    m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                    run_dir=out, cases=(15,))
    frozen = json.loads((out/'frozen_ablations.json').read_text())
    frozen['b1_source_sha256'] = '0'*64
    (out/'frozen_ablations.json').write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match='PREDICTOR_CODE_CHANGED'):
        m.audit_frozen(out)


def test_evaluation_uses_preexisting_candidates_and_invariants(tmp_path, monkeypatch):
    calls, out = stubbed(tmp_path, monkeypatch)
    m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                    run_dir=out, cases=(15,))
    nfit = len([c for c in calls if c[0] == 'fit'])
    def evaluate(*, candidate_path, scene_dir, models_dir):
        record = json.loads(candidate_path.read_text())
        return {
            'candidate_sha256': b1._sha256(candidate_path),
            'gt_pose_read_by_evaluator': True,
            'gt_pose_used_for_estimation': False,
            'publishable': False, 'position_m': None,
            'translation_error_mm': float(record['pose_candidate_model_to_camera']['t_m'][0]),
            'rotation_error_deg_raw': 15.0,
        }
    monkeypatch.setattr(b2, 'evaluate_saved_candidate', evaluate)
    result = m.evaluate_cases(scene_dir=tmp_path, models_dir=tmp_path,
                              run_dir=out)
    assert result['record_count'] == 4
    assert result['not_held_out'] is True
    assert result['publishable'] is False
    assert len([c for c in calls if c[0] == 'fit']) == nfit
    assert (out/'diagnostic_summary.json').exists()
    with pytest.raises(ValueError, match='EVALUATION_ALREADY_EXISTS'):
        m.evaluate_cases(scene_dir=tmp_path, models_dir=tmp_path, run_dir=out)


def test_second_run_dir_rejected(tmp_path, monkeypatch):
    _, out = stubbed(tmp_path, monkeypatch)
    m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                    run_dir=out, cases=(15,))
    with pytest.raises(ValueError, match='RUN_DIRECTORY_ALREADY_EXISTS'):
        m.predict_cases(scene_dir=tmp_path, models_dir=tmp_path,
                        run_dir=out, cases=(15,))


def test_support_threshold_only_diagnostic_real_algorithm():
    rng = np.random.default_rng(351)
    model = rng.normal(size=(50,3)) * [0.02,0.03,0.01]
    obs = model + [0,0,1]
    p0, d0 = b1.fit_pca_multistart_icp(
        model, obs, max_iterations=5, trim_fraction=.75, support_threshold_m=.008)
    p1, d1 = b1.fit_pca_multistart_icp(
        model, obs, max_iterations=5, trim_fraction=.75, support_threshold_m=.004)
    np.testing.assert_array_equal(p0.R,p1.R)
    np.testing.assert_array_equal(p0.t_m,p1.t_m)
    assert d0['trimmed_rmse_m'] == d1['trimmed_rmse_m']
    assert d1['support_fraction'] <= d0['support_fraction']
