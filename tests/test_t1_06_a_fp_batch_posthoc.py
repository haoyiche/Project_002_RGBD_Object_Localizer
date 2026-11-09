"""CPU contract tests; no real BOP GT, Docker, or CUDA needed."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.evaluation import t1_06_a_fp_batch_posthoc as ev
from src.evaluation import t1_04_b4_fixed_50_batch as b4


def fixture_icp(tmp_path, *, corrupt_gt=False, drop_frame=False, corrupt_freeze=False):
    ids = list(range(50))
    freeze = tmp_path / 'frozen_predictions.json'
    freeze.write_text('immutable fixture', encoding='utf-8')
    summary_path = tmp_path / 'batch_summary.json'
    rows = [dict(im_id=i, translation_error_mm=2.0 + i/10,
                 rotation_error_deg_raw=float(i)) for i in ids]
    if drop_frame:
        rows[-1]['im_id'] = 100
    summary = dict(
        schema=b4.SUMMARY_SCHEMA, frame_count_expected=50,
        frame_count_evaluated=50, failed_count=0, is_50_frame_scope=True,
        obj_id=1, gt_index=0, raw_rotation_not_symmetry_aware=True,
        dataset_gt_used_only_in_evaluation=True, bop_benchmark_score_claimed=False,
        gt_annotation_sha256='different' if corrupt_gt else 'gt-sha',
        frozen_predictions_sha256='wrong' if corrupt_freeze else ev.batch.sha(freeze),
        samples=rows,
    )
    summary_path.write_text(json.dumps(summary), encoding='utf-8')
    return summary_path, ids


def test_stats_match_b4_definition():
    data = list(range(50))
    result = ev.stats(data)
    assert result['mean'] == 24.5
    assert result['median'] == 24.5
    assert result['p95_linear'] == pytest.approx(46.55)
    assert result['max'] == 49.0


def test_stats_reject_incomplete_scope():
    with pytest.raises(ValueError, match='STATS_SCOPE_OR_VALUE_INVALID'):
        ev.stats([1.0, 2.0])


def test_load_icp_50_matched(tmp_path):
    path, ids = fixture_icp(tmp_path)
    data, digest = ev.load_icp_summary(path, ids, 'gt-sha')
    assert len(data['samples']) == 50
    assert digest == ev.batch.sha(path)


def test_reject_gt_annotation_change(tmp_path):
    path, ids = fixture_icp(tmp_path, corrupt_gt=True)
    with pytest.raises(ValueError, match='ICP_AND_FP_GT_ANNOTATION_DIFFER'):
        ev.load_icp_summary(path, ids, 'gt-sha')


def test_reject_frame_set_mismatch(tmp_path):
    path, ids = fixture_icp(tmp_path, drop_frame=True)
    with pytest.raises(ValueError, match='ICP_FRAME_SET_OR_ORDER_DIFFERENT'):
        ev.load_icp_summary(path, ids, 'gt-sha')


def test_reject_baseline_freeze_tampering(tmp_path):
    path, ids = fixture_icp(tmp_path, corrupt_freeze=True)
    with pytest.raises(ValueError, match='ICP_FROZEN_PREDICTIONS_HASH_MISMATCH'):
        ev.load_icp_summary(path, ids, 'gt-sha')


def test_paired_comparison_no_cherry_pick(tmp_path):
    path, ids = fixture_icp(tmp_path)
    icp, digest = ev.load_icp_summary(path, ids, 'gt-sha')
    fps = [dict(im_id=i, translation_error_mm=1.0,
                rotation_error_deg_raw=2.0) for i in ids]
    out = ev.paired_comparison(fps, icp, digest, 'gt-sha', 'fp_sha')
    assert out['frame_count'] == 50
    assert out['paired_frame_counts']['fp_lower_translation_error'] == 50
    assert out['paired_frame_counts']['fp_lower_raw_rotation_error'] == 47
    assert out['samples'][0]['im_id'] == 0


def test_output_preexist_rejected_before_gt(tmp_path, monkeypatch):
    output = tmp_path / 'exists'
    output.mkdir()
    user = SimpleNamespace(output_dir=output)
    monkeypatch.setattr(ev, 'preflight', lambda x: (_ for _ in ()).throw(AssertionError('MUST_NOT_RUN')))
    with pytest.raises(ValueError, match='EVALUATION_OUTPUT_ALREADY_EXISTS'):
        ev.evaluate_fixed50(user)


def test_compare_only_missing_icp_summary_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='ICP_SUMMARY_REQUIRED_FOR_COMPARE_ONLY'):
        ev.compare_existing_evaluation(SimpleNamespace(icp_summary=None))


def test_compare_only_reads_existing_evaluations_without_rerunning(tmp_path, monkeypatch):
    icpdir = tmp_path / 'icp'
    icpdir.mkdir()
    original_scene = tmp_path / 'scene'
    original_scene.mkdir()
    (original_scene / 'scene_gt.json').write_text('{"fixture": 1}', encoding='utf-8')
    gtsha = ev.batch.sha(original_scene / 'scene_gt.json')
    baseline, ids = fixture_icp(icpdir)
    b = json.loads(baseline.read_text(encoding='utf-8'))
    b['gt_annotation_sha256'] = gtsha
    baseline.write_text(json.dumps(b), encoding='utf-8')

    run = tmp_path / 'frozen'
    run.mkdir()
    (run / 'frozen_predictions.json').write_text('frozen', encoding='utf-8')
    dest = tmp_path / 'evaluation'
    (dest / 'frames').mkdir(parents=True)
    samples, records = [], []
    for im_id in ids:
        hash_value = f'candidate-sha-{im_id}'
        record = dict(im_id=im_id, candidate_sha256=hash_value,
                      translation_error_mm=1.0, rotation_error_deg_raw=2.0)
        payload = dict(candidate_sha256=hash_value, translation_error_mm=1.0,
                       rotation_error_deg_raw=2.0, gt_annotation_sha256=gtsha)
        path = dest / 'frames' / f'evaluation_{im_id:06d}.json'
        path.write_text(json.dumps(payload), encoding='utf-8')
        record['evaluation_sha256'] = ev.batch.sha(path)
        samples.append(record)
        records.append(dict(frame_id=im_id, candidate_sha256=hash_value))
    summary = dict(schema=ev.SCHEMA,
                   status='FULL_50_ANNOTATION_RELATIVE_DIAGNOSTIC_ONLY',
                   count_expected=50, count_evaluated=50, failed_count=0,
                   frame_ids=ids, symmetry_aware=False, publishable=False,
                   frozen_predictions_sha256=ev.batch.sha(run / 'frozen_predictions.json'),
                   gt_annotation_sha256=gtsha, samples=samples)
    (dest / 'batch_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    monkeypatch.setattr(ev, 'preflight', lambda user:
                        (SimpleNamespace(run_dir=run, scene_dir=original_scene),
                         {'frame_ids': ids}, {}, records))
    user = SimpleNamespace(icp_summary=baseline, output_dir=dest)
    result = ev.compare_existing_evaluation(user)
    assert result['frame_count'] == 50
    assert (dest / 'paired_comparison.json').is_file()
    with pytest.raises(ValueError, match='PAIRED_COMPARISON_ALREADY_EXISTS'):
        ev.compare_existing_evaluation(user)
