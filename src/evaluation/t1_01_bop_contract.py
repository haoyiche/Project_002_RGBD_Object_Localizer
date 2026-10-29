"""T1-01: BOP scene metadata and coordinate/units contract (CPU-only).

Source format: https://github.com/thodan/bop_toolkit/blob/master/docs/bop_datasets_format.md
This module does NOT estimate a 6D pose or establish physical accuracy.
It intentionally does not compute pose error from a mask centroid.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


@dataclass(frozen=True)
class BopFrameContract:
    scene_dir: Path
    im_id: int
    gt_index: int
    obj_id: int
    K: np.ndarray                         # Per-image intrinsics, pixels
    depth_scale_mm_per_unit: float        # BOP depth image raw units -> mm
    R_cam_from_model: np.ndarray          # Model -> camera
    t_cam_from_model_m: np.ndarray        # Model origin in camera frame, m
    rgb_path: Path
    depth_path: Path
    mask_visib_path: Path


def _load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_bop_frame_contract(scene_dir, *, im_id, gt_index=0, require_files=True):
    """Read per-image camera calibration and ONE GT object-instance pose.

    `gt_index` selects the scene_gt.json list entry, not `obj_id`.
    `mask_visib/{im_id:06d}_{gt_index:06d}.png` refers to that same entry.
    Does not read image pixels. No model-origin/centroid equivalence assumed.
    """
    if type(im_id) is not int or im_id < 0 or type(gt_index) is not int or gt_index < 0:
        raise ValueError('im_id and gt_index must be nonnegative integers')
    root = Path(scene_dir)
    cameras = _load_json(root / 'scene_camera.json')
    poses = _load_json(root / 'scene_gt.json')
    key = str(im_id)
    if key not in cameras or key not in poses:
        raise ValueError('IMAGE_ID_NOT_FOUND')
    camera = cameras[key]
    entries = poses[key]
    if not isinstance(entries, list) or gt_index >= len(entries):
        raise ValueError('GT_INDEX_NOT_FOUND')
    pose = entries[gt_index]

    K = np.asarray(camera['cam_K'], dtype=np.float64).reshape(3, 3)
    R = np.asarray(pose['cam_R_m2c'], dtype=np.float64).reshape(3, 3)
    t_mm = np.asarray(pose['cam_t_m2c'], dtype=np.float64).reshape(3)
    scale = float(camera['depth_scale'])
    if (not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0
            or not np.allclose(K[2], [0, 0, 1], atol=1e-8)
            or abs(float(np.linalg.det(K))) < 1e-10):
        raise ValueError('INVALID_INTRINSICS')
    if (not np.isfinite(R).all()
            or not np.allclose(R.T @ R, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5)):
        raise ValueError('INVALID_ROTATION')
    if not np.isfinite(t_mm).all():
        raise ValueError('INVALID_TRANSLATION')
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('INVALID_DEPTH_SCALE')
    obj_id = pose['obj_id']
    if type(obj_id) is not int or obj_id <= 0:
        raise ValueError('INVALID_OBJECT_ID')

    stem = f'{im_id:06d}'
    paths = (
        root / 'rgb' / f'{stem}.png',
        root / 'depth' / f'{stem}.png',
        root / 'mask_visib' / f'{stem}_{gt_index:06d}.png',
    )
    if require_files and any(not p.is_file() for p in paths):
        raise ValueError('MISSING_RGB_DEPTH_OR_VISIBLE_MASK')
    return BopFrameContract(root, im_id, gt_index, obj_id, K, scale,
                            R, t_mm / 1000.0, *paths)


def depth_raw_to_m(raw_depth, contract):
    """BOP depth_mm = raw_uint16 * depth_scale; zero remains invalid.

    Returns depth in meters and an explicit valid-depth mask.
    """
    raw = np.asarray(raw_depth)
    if raw.ndim != 2 or raw.dtype != np.uint16:
        raise ValueError('DEPTH_IMAGE_MUST_BE_UINT16_HW')
    depth_m = raw.astype(np.float64) * (contract.depth_scale_mm_per_unit / 1000.0)
    valid = raw > 0
    return depth_m, valid


def pixel_to_camera_m(*, u, v, raw_depth, contract):
    """Back-project a valid BOP depth sample into the CAMERA frame in meters."""
    if type(raw_depth) not in (int, np.uint16) or not 0 < int(raw_depth) <= 65535:
        raise ValueError('INVALID_DEPTH_SAMPLE')
    z_m = float(raw_depth) * contract.depth_scale_mm_per_unit / 1000.0
    if not np.isfinite(z_m) or z_m <= 0:
        raise ValueError('INVALID_DEPTH_SAMPLE')
    pixel = np.array([float(u), float(v), 1.0], dtype=np.float64)
    if not np.isfinite(pixel).all():
        raise ValueError('INVALID_PIXEL')
    return (np.linalg.solve(contract.K, pixel) * z_m)


def main():
    """Self-contained, transparent contract demo; no real BOP data involved."""
    with TemporaryDirectory() as tmp:
        scene = Path(tmp)
        (scene / 'scene_camera.json').write_text(json.dumps({'0': {
            'cam_K': [100, 0, 1, 0, 100, 1, 0, 0, 1], 'depth_scale': 0.1,
        }}), encoding='utf-8')
        (scene / 'scene_gt.json').write_text(json.dumps({'0': [{
            'obj_id': 1, 'cam_R_m2c': np.eye(3).reshape(-1).tolist(),
            'cam_t_m2c': [10, 20, 500],
        }]}), encoding='utf-8')
        info = load_bop_frame_contract(scene, im_id=0, require_files=False)
        depth, valid = depth_raw_to_m(np.array([[5000, 0]], dtype=np.uint16), info)
        xyz = pixel_to_camera_m(u=2, v=1, raw_depth=5000, contract=info)
        print('reference_kind: LOCAL_SYNTHETIC_FORMAT_FIXTURE')
        print('depth_m:', depth.tolist(), 'valid_depth:', valid.tolist())
        print('pixel_xyz_camera_m:', xyz.tolist())
        print('gt_model_origin_camera_m:', info.t_cam_from_model_m.tolist())
        print('model_origin_is_not_pointcloud_centroid: True')
        assert np.allclose(xyz, [.005, 0, .5])
        assert np.allclose(info.t_cam_from_model_m, [.01, .02, .5])
        print('T1-01 BOP metadata/units contract: PASS')


if __name__ == '__main__':
    main()
