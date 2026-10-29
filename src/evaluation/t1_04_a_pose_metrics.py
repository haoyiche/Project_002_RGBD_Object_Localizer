"""T1-04-A: CPU-only model->camera 6D pose metric contract.

Translation error (mm) and *raw*, non-symmetry-aware rotation geodesic (deg).
No pose estimation is performed here. GT-derived perturbations are SANITY FIXTURES,
not legitimate predictions nor a BOP benchmark score.

BOP fields: cam_R_m2c / cam_t_m2c (translation in mm). Internal t: meters.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PoseM2C:
    """T_camera_from_model. R (3,3) SO(3), t_m (3,) meters.

    source_frame/target_frame are enforced to guard accidental c2m labeling.
    Values are copied into read-only arrays so external array edits cannot
    silently alter an already-validated pose.
    """

    R: np.ndarray
    t_m: np.ndarray
    source_frame: str = "model"
    target_frame: str = "camera"

    def __post_init__(self):
        if self.source_frame != "model" or self.target_frame != "camera":
            raise ValueError("INVALID_POSE_FRAME_CONVENTION")
        try:
            R = np.array(self.R, dtype=np.float64, copy=True)
            t = np.array(self.t_m, dtype=np.float64, copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("INVALID_POSE_NUMERIC_INPUT") from exc
        if R.shape != (3, 3):
            raise ValueError("INVALID_ROTATION_SHAPE")
        if t.shape != (3,):
            raise ValueError("INVALID_TRANSLATION_SHAPE")
        if not np.isfinite(R).all():
            raise ValueError("NONFINITE_ROTATION")
        if not np.isfinite(t).all():
            raise ValueError("NONFINITE_TRANSLATION")
        if (not np.allclose(R.T @ R, np.eye(3), atol=1e-5, rtol=0.0)
                or not np.isclose(np.linalg.det(R), 1.0, atol=1e-5, rtol=0.0)):
            raise ValueError("INVALID_ROTATION_MATRIX")
        R.setflags(write=False)
        t.setflags(write=False)
        object.__setattr__(self, "R", R)
        object.__setattr__(self, "t_m", t)

    @classmethod
    def from_bop(cls, cam_R_m2c, cam_t_m2c, *, translation_unit="mm",
                 source_frame="model", target_frame="camera") -> PoseM2C:
        """Convert BOP translation mm into meters exactly once."""
        if translation_unit != "mm":
            raise ValueError("BOP_TRANSLATION_UNIT_MUST_BE_MM")
        try:
            t_mm = np.asarray(cam_t_m2c, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("INVALID_POSE_NUMERIC_INPUT") from exc
        return cls(cam_R_m2c, t_mm / 1000.0,
                   source_frame=source_frame, target_frame=target_frame)


def axis_angle_rotation(axis, degrees: float) -> np.ndarray:
    """Construct a valid rotation for controlled *synthetic* tests."""
    a = np.asarray(axis, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("INVALID_ROTATION_AXIS")
    magnitude = float(np.linalg.norm(a))
    if magnitude <= 0 or not np.isfinite(degrees):
        raise ValueError("INVALID_AXIS_ANGLE")
    a = a / magnitude
    angle = math.radians(float(degrees))
    x, y, z = a
    K = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def compare_pose_m2c(estimate: PoseM2C, ground_truth: PoseM2C, *,
                     symmetry_policy: str = "raw") -> dict:
    """Compare same-frame poses; returns diagnostics, NOT an estimator score.

    The raw rotation geodesic ignores object symmetries. To avoid silent metric
    mislabeling, requests for symmetry-aware evaluation are rejected here.
    """
    if symmetry_policy != "raw":
        raise ValueError("UNSUPPORTED_SYMMETRY_POLICY")
    if not isinstance(estimate, PoseM2C) or not isinstance(ground_truth, PoseM2C):
        raise ValueError("POSE_M2C_REQUIRED")

    translation_error_mm = float(np.linalg.norm(
        estimate.t_m - ground_truth.t_m) * 1000.0)
    relative = estimate.R @ ground_truth.R.T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    skew = relative - relative.T
    sine = float(np.linalg.norm(np.array([
        skew[2, 1], skew[0, 2], skew[1, 0]
    ])) / 2.0)
    # atan2(sin(theta), cos(theta)) is stable near 0 and 180 degrees.
    rotation_error_deg_raw = math.degrees(math.atan2(sine, cosine))

    return {
        "translation_error_mm": translation_error_mm,
        "rotation_error_deg_raw": rotation_error_deg_raw,
        "symmetry_aware": False,
        "reference_kind": "POSE_METRIC_CONTRACT_DIAGNOSTIC",
        "publishable": False,
        "position_m": None,
    }


def controlled_demo(scene_dir=None, *, im_id=15, gt_index=0) -> dict:
    """Inject GT-derived perturbations for formula sanity, never pose inference."""
    if scene_dir is None:
        gt = PoseM2C(np.eye(3), np.array([0.0, 0.0, 0.5]))
        sample_kind = "SYNTHETIC_IDENTITY_GT_FIXTURE"
    else:
        from src.evaluation.t1_01_bop_contract import load_bop_frame_contract
        c = load_bop_frame_contract(Path(scene_dir), im_id=im_id, gt_index=gt_index)
        gt = PoseM2C(c.R_cam_from_model, c.t_cam_from_model_m)
        sample_kind = "REAL_BOP_GT_WITH_SYNTHETIC_GT_DERIVED_PERTURBATION"

    # Translation delta is expressed in CAMERA coordinates in meters.
    estimate = PoseM2C(
        axis_angle_rotation((0, 0, 1), 10.0) @ gt.R,
        gt.t_m + np.array([0.005, -0.002, 0.0]),
    )
    result = compare_pose_m2c(estimate, gt)
    result.update({
        "sample_kind": sample_kind,
        "estimation_performed": False,
        "pose_accuracy_claimed": False,
        "physical_accuracy_claimed": False,
        "gt_used_to_construct_candidate": True,
        "expected_translation_error_mm": math.sqrt(29.0),
        "expected_rotation_error_deg_raw": 10.0,
    })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", default=None, type=Path)
    parser.add_argument("--im-id", default=15, type=int)
    parser.add_argument("--gt-index", default=0, type=int)
    args = parser.parse_args(argv)
    report = controlled_demo(args.scene_dir, im_id=args.im_id, gt_index=args.gt_index)
    for key, val in report.items():
        print(f"{key}: {val}")
    print("T1-04-A formula / frame / units sanity: PASS (NOT pose inference)")


if __name__ == "__main__":
    main()
