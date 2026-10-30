"""T1-04-B1: GT-blind, known-model / ORACLE-mask PCA+ICP pose hypothesis.

Prediction inputs: scene_camera.json, RGB, depth, mask_visib, model PLY,
models_info.json, and explicit obj_id/im_id/gt_index. NEVER scene_gt.json or
any T1-02 Manifest (both contain pose GT). Units: model mm -> m, depth mm -> m.

This is an experimental geometric hypothesis, not a verified BOP 6D score.
Even a small ICP residual does not prove a correct pose, especially under
partial views, symmetry and repeated geometry. GT is evaluated separately.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
from scipy.spatial import cKDTree

from src.evaluation.t1_04_a_pose_metrics import PoseM2C


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_oracle_observation(scene_dir: Path, *, im_id: int, gt_index: int):
    """No scene_gt read. Oracle mask index is EXTERNAL supervision, not predicted.

    Returns (points_camera_m[N,3], audit_dict). RGB is validated but not used
    for registration. Only camera calibration and observed PNG bytes are read.
    """
    scene_dir = Path(scene_dir)
    if type(im_id) is not int or im_id < 0 or type(gt_index) is not int or gt_index < 0:
        raise ValueError('INVALID_IMAGE_OR_INSTANCE_INDEX')
    cam_path = scene_dir / 'scene_camera.json'
    meta = json.loads(cam_path.read_text(encoding='utf-8'))
    frame = meta.get(str(im_id))
    if not isinstance(frame, dict):
        raise ValueError('CAMERA_IMAGE_ID_NOT_FOUND')
    try:
        K = np.array(frame['cam_K'], dtype=np.float64).reshape(3, 3)
        scale = float(frame['depth_scale'])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError('INVALID_CAMERA_METADATA') from exc
    if (not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0
            or not np.allclose(K[2], [0., 0., 1.], atol=1e-8, rtol=0)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError('INVALID_CAMERA_METADATA')

    stem = f'{im_id:06d}'
    files = {
        'rgb': scene_dir / 'rgb' / f'{stem}.png',
        'depth': scene_dir / 'depth' / f'{stem}.png',
        'mask_visib': scene_dir / 'mask_visib' / f'{stem}_{gt_index:06d}.png',
    }
    if any(not x.is_file() for x in files.values()):
        raise ValueError('MISSING_RGB_DEPTH_OR_ORACLE_MASK')
    rgb = cv2.imread(str(files['rgb']), cv2.IMREAD_UNCHANGED)
    raw = cv2.imread(str(files['depth']), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(files['mask_visib']), cv2.IMREAD_UNCHANGED)
    if rgb is None or raw is None or mask is None:
        raise ValueError('PNG_DECODE_FAILED')
    if rgb.ndim != 3 or rgb.dtype != np.uint8 or rgb.shape[2] != 3:
        raise ValueError('INVALID_RGB_FORMAT')
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError('DEPTH_MUST_BE_UINT16')
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0,255)).all():
        raise ValueError('MASK_MUST_BE_BINARY_UINT8')
    if raw.shape != mask.shape or raw.shape != rgb.shape[:2]:
        raise ValueError('RGB_DEPTH_MASK_SHAPE_MISMATCH')
    visible = mask == 255
    valid = visible & (raw > 0)
    if np.count_nonzero(valid) < 30:
        raise ValueError('INSUFFICIENT_VISIBLE_DEPTH_POINTS')
    v, u = np.nonzero(valid)
    z = raw[valid].astype(np.float64) * scale / 1000.0
    # General invertible intrinsic matrix rather than assuming zero skew.
    rays = np.linalg.solve(K, np.vstack((u, v, np.ones(len(u)))))
    xyz = (rays * z).T
    if not np.isfinite(xyz).all() or np.any(z <= 0):
        raise ValueError('INVALID_OBSERVATION_GEOMETRY')
    audit = {
        'im_id': im_id, 'gt_index': gt_index,
        'visible_mask_pixels': int(visible.sum()),
        'valid_depth_points': int(len(xyz)),
        'depth_scale_mm_per_raw_unit': scale,
        'K_row_major': K.ravel().tolist(),
        'sha256': {'scene_camera': _sha256(cam_path), **{k: _sha256(p) for k,p in files.items()}},
        'oracle_mask_used': True,
        'gt_pose_read_by_predictor': False,
    }
    return xyz, audit


def check_and_scale_model(vertices_mm, model_info: dict):
    """Fail closed on model unit/extent mismatch. Returns model vertices [N,3] m."""
    x = np.asarray(vertices_mm, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or len(x) < 30 or not np.isfinite(x).all():
        raise ValueError('INVALID_MODEL_VERTICES')
    try:
        size = np.asarray([model_info[f'size_{k}'] for k in 'xyz'], dtype=np.float64)
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('INVALID_MODEL_METADATA') from exc
    if np.any(size <= 0) or not np.isfinite(size).all():
        raise ValueError('INVALID_MODEL_METADATA')
    extent = np.ptp(x, axis=0)
    if not np.allclose(extent, size, atol=1.0, rtol=0):
        raise ValueError('MODEL_SIZE_MM_MISMATCH')
    return x / 1000.0


def load_model_points(models_dir: Path, obj_id: int):
    """Open3D used only for PLY decoding; the registration core uses NumPy/SciPy."""
    if type(obj_id) is not int or obj_id <= 0:
        raise ValueError('INVALID_OBJ_ID')
    root = Path(models_dir)
    ply = root / f'obj_{obj_id:06d}.ply'
    info_path = root / 'models_info.json'
    if not ply.is_file() or not info_path.is_file():
        raise ValueError('MODEL_FILE_OR_METADATA_MISSING')
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(ply))
    if not mesh.has_triangles():
        raise ValueError('MODEL_MESH_NO_TRIANGLES')
    info = json.loads(info_path.read_text(encoding='utf-8'))
    if str(obj_id) not in info:
        raise ValueError('OBJ_ID_NOT_IN_MODELS_INFO')
    points_m = check_and_scale_model(np.asarray(mesh.vertices), info[str(obj_id)])
    return points_m, {'obj_id': obj_id, 'mesh_sha256': _sha256(ply),
                      'models_info_sha256': _sha256(info_path),
                      'model_vertex_count': int(len(points_m)),
                      'model_units_input': 'mm', 'model_units_internal': 'm'}


def _reduce(points, max_points=1800):
    # Fixed-index subsample is deterministic. No GT or random seed needed.
    if len(points) <= max_points:
        return points.copy()
    return points[np.linspace(0, len(points)-1, max_points).astype(int)].copy()


def _fit_rigid(model, camera):
    """Kabsch: model->camera, corresponding [N,3] points; no scale estimation."""
    ca = model.mean(axis=0)
    cb = camera.mean(axis=0)
    U, _, Vt = np.linalg.svd((model-ca).T @ (camera-cb))
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1., 1., np.sign(d)]) @ U.T
    t = cb - R @ ca
    return R, t


def _basis(points):
    centered = points - points.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    axes = Vt.T
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return axes


def _initial_rotations(model, observed):
    bm, bo = _basis(model), _basis(observed)
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[:, perm]
        for signs in itertools.product((-1., 1.), repeat=3):
            Q = P @ np.diag(signs)
            if np.linalg.det(Q) > 0:
                yield bo @ Q @ bm.T


def fit_pca_multistart_icp(model_m, observed_m, *, max_iterations=18,
                           trim_fraction=0.75, support_threshold_m=0.008):
    """GT-free partial-overlap registration. Returns UNVERIFIED pose hypothesis.

    Observed->nearest transformed MODEL vertex; Kabsch maps matched model
    vertices to camera points. 24 PCA right-handed axis permutations.
    Geometric support is NOT a pose-accuracy or correctness certificate.
    """
    a0 = np.asarray(model_m, dtype=np.float64)
    b0 = np.asarray(observed_m, dtype=np.float64)
    for x, label in [(a0, 'MODEL'), (b0, 'OBSERVATION')]:
        if x.ndim != 2 or x.shape[1] != 3 or len(x) < 30 or not np.isfinite(x).all():
            raise ValueError(f'INVALID_{label}_POINTS')
    if (type(max_iterations) is not int or max_iterations < 1
            or not 0.5 <= trim_fraction <= 1.0
            or not np.isfinite(support_threshold_m) or support_threshold_m <= 0):
        raise ValueError('INVALID_ICP_PARAMETERS')
    a = _reduce(a0)
    b = _reduce(b0)
    if np.linalg.matrix_rank(a - a.mean(axis=0)) < 3:
        raise ValueError('DEGENERATE_MODEL_GEOMETRY')
    if np.linalg.matrix_rank(b - b.mean(axis=0)) < 2:
        raise ValueError('DEGENERATE_OBSERVATION_GEOMETRY')
    best = None
    ntrim = max(30, int(np.ceil(trim_fraction * len(b))))
    for R0 in _initial_rotations(a, b):
        R = R0.copy()
        t = b.mean(axis=0) - R @ a.mean(axis=0)
        for _ in range(max_iterations):
            mapped = a @ R.T + t
            distance, index = cKDTree(mapped).query(b, k=1)
            chosen = np.argpartition(distance, ntrim-1)[:ntrim]
            updated_R, updated_t = _fit_rigid(a[index[chosen]], b[chosen])
            change = (np.linalg.norm(t-updated_t)
                      + np.linalg.norm(R-updated_R) * 0.05)
            R, t = updated_R, updated_t
            if change < 1e-8:
                break
        distance, _ = cKDTree(a @ R.T + t).query(b, k=1)
        # Candidate selection uses lower 75% residual, NOT GT.
        trim = np.partition(distance, ntrim-1)[:ntrim]
        trim_rmse = float(np.sqrt(np.mean(trim ** 2)))
        if best is None or trim_rmse < best['trimmed_rmse_m']:
            best = {'R': R.copy(), 't_m': t.copy(),
                    'trimmed_rmse_m': trim_rmse,
                    'support_fraction': float(np.mean(distance <= support_threshold_m)),
                    'support_threshold_m': float(support_threshold_m),
                    'median_nn_distance_m': float(np.median(distance)),
                    'num_model_points': len(a), 'num_observed_points': len(b)}
    pose = PoseM2C(best['R'], best['t_m'])
    return pose, {k:v for k,v in best.items() if k not in ('R','t_m')}


def make_prediction_report(*, scene_dir, models_dir, im_id, gt_index, obj_id):
    # The prediction code has NO call to T1-01 load_bop_frame_contract()
    # or T1-02 read_instance() / verify_manifest() (they read scene GT).
    observed, obs_info = read_oracle_observation(
        scene_dir, im_id=im_id, gt_index=gt_index)
    model, model_info = load_model_points(models_dir, obj_id)
    pose, reg = fit_pca_multistart_icp(model, observed)
    return {
        'schema': 't1-04-b1-oracle-mask-gt-blind-icp-v1',
        'method': 'PCA_24_START_PARTIAL_OVERLAP_POINT_TO_POINT_ICP',
        'status': 'GEOMETRIC_CANDIDATE_UNVERIFIED',
        'estimation_performed': True,
        'gt_pose_read_by_predictor': False,
        'gt_pose_used_to_construct_candidate': False,
        'oracle_mask_used': True,
        'oracle_supervision_used': True,
        'pose_accuracy_claimed': False,
        'physical_accuracy_claimed': False,
        'symmetry_aware': False,
        'publishable': False,
        'position_m': None,
        'pose_candidate_model_to_camera': {
            'R_row_major': pose.R.ravel().tolist(),
            't_m': pose.t_m.tolist(),
        },
        'registration_diagnostic_only': reg,
        'observation': obs_info,
        'model': model_info,
        'note': ('This is a GT-free hypothesis using oracle instance mask/ID; '
                 'low NN/ICP residual is not independent pose accuracy.'),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene-dir', type=Path, required=True)
    p.add_argument('--models-dir', type=Path, required=True)
    p.add_argument('--im-id', type=int, required=True)
    p.add_argument('--gt-index', type=int, default=0)
    p.add_argument('--obj-id', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    # Never overwrite a previous report: a failed re-run must not make an old
    # JSON look like fresh evidence. Use a new output filename per run.
    if args.output.exists():
        raise FileExistsError('OUTPUT_ALREADY_EXISTS_USE_NEW_RUN_PATH')
    report = make_prediction_report(scene_dir=args.scene_dir, models_dir=args.models_dir,
                                    im_id=args.im_id, gt_index=args.gt_index,
                                    obj_id=args.obj_id)
    # Important: do not write a PASS pose report. No GT or true correctness check.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write prevents partial reports. Separate output path per experiment.
    path = args.output
    tmp_path = None
    try:
        with NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                prefix='.t1_04_b1_', suffix='.tmp',
                                delete=False) as f:
            tmp_path = Path(f.name)
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()
    print('GT pose read in prediction:', report['gt_pose_read_by_predictor'])
    print('Oracle mask used:', report['oracle_mask_used'])
    print('Candidate t camera (m):', report['pose_candidate_model_to_camera']['t_m'])
    print('Registration diagnostics:', report['registration_diagnostic_only'])
    print('status:', report['status'])
    print('publishable:', report['publishable'], 'position_m:', report['position_m'])
    print('Prediction report:', path)


if __name__ == '__main__':
    main()
