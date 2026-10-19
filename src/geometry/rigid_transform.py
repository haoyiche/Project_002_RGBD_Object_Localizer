"""EXP035-C: validated, immutable SE(3) numerical transform interface.

Convention: T_target_from_source maps column-vector source coordinates into
                target coordinates, p_target = R @ p_source + t_m.
All translation and point coordinates use METERS.

`calibration_verified` records a *claim*, not authenticated calibration evidence.
This module intentionally does NOT grant robot-position publication permission.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

ROTATION_TOL = 1e-9


def _finite_array(value, shape, name):
    """Reject wrong shapes, booleans, strings and NaN/Inf before cast to float64."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape {shape} and numeric elements") from exc
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {raw.shape}")
    for item in raw.flat:
        if isinstance(item, (bool, np.bool_)) or not isinstance(item, Real):
            raise ValueError(f"{name} must contain real numbers, not booleans/strings")
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be convertible to float64") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _valid_frame_id(frame, name):
    if not isinstance(frame, str) or not frame.strip():
        raise ValueError(f"{name} must be a nonempty frame ID")
    return frame


@dataclass(frozen=True, init=False)
class RigidTransform:
    """Immutable numeric SE(3) with source/target frame labels and calibration claim.

    R and t_m properties return copies to prevent accidental mutation.
    Composition uses `outer.compose(inner)` = outer @ inner, rightmost first.
    """

    source_frame_id: str
    target_frame_id: str
    _rotation: tuple
    _translation: tuple
    calibration_verified: bool

    def __init__(self, source_frame_id, target_frame_id, R, t_m,
                 calibration_verified=False):
        source = _valid_frame_id(source_frame_id, "source_frame_id")
        target = _valid_frame_id(target_frame_id, "target_frame_id")
        rotation = _finite_array(R, (3, 3), "R")
        translation = _finite_array(t_m, (3,), "t_m")
        if type(calibration_verified) is not bool:
            raise ValueError("calibration_verified must be exactly bool")
        if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0,
                           atol=ROTATION_TOL):
            raise ValueError("R must be orthogonal: R.T @ R ~= I")
        if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0,
                          atol=ROTATION_TOL):
            raise ValueError("R must have determinant +1 (no reflection)")
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "target_frame_id", target)
        object.__setattr__(self, "_rotation", tuple(tuple(float(v) for v in row)
                                                   for row in rotation))
        object.__setattr__(self, "_translation", tuple(float(v) for v in translation))
        object.__setattr__(self, "calibration_verified", calibration_verified)

    @property
    def R(self):
        return np.array(self._rotation, dtype=np.float64)

    @property
    def t_m(self):
        return np.array(self._translation, dtype=np.float64)

    def as_matrix(self):
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.R
        result[:3, 3] = self.t_m
        return result

    def transform_point(self, point_m, *, frame_id):
        if _valid_frame_id(frame_id, "frame_id") != self.source_frame_id:
            raise ValueError(f"frame mismatch: expected {self.source_frame_id!r}, got {frame_id!r}")
        p = _finite_array(point_m, (3,), "point_m")
        return self.R @ p + self.t_m

    def transform_points(self, points_m, *, frame_id):
        if _valid_frame_id(frame_id, "frame_id") != self.source_frame_id:
            raise ValueError(f"frame mismatch: expected {self.source_frame_id!r}, got {frame_id!r}")
        try:
            raw = np.asarray(points_m, dtype=object)
        except (TypeError, ValueError) as exc:
            raise ValueError("points_m must have shape (N,3)") from exc
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"points_m must have shape (N,3), got {raw.shape}")
        points = _finite_array(points_m, raw.shape, "points_m")
        return points @ self.R.T + self.t_m

    def inverse(self):
        R_inv = self.R.T
        return RigidTransform(
            self.target_frame_id, self.source_frame_id,
            R_inv, -R_inv @ self.t_m,
            calibration_verified=self.calibration_verified,
        )

    def compose(self, inner):
        """self(T_C_B).compose(inner(T_B_A)) returns T_C_A."""
        if not isinstance(inner, RigidTransform):
            raise ValueError("inner must be RigidTransform")
        if self.source_frame_id != inner.target_frame_id:
            raise ValueError(
                "frame mismatch in composition: "
                f"outer source {self.source_frame_id!r} != "
                f"inner target {inner.target_frame_id!r}"
            )
        return RigidTransform(
            inner.source_frame_id, self.target_frame_id,
            self.R @ inner.R,
            self.R @ inner.t_m + self.t_m,
            calibration_verified=(self.calibration_verified and inner.calibration_verified),
        )


def inspect_numeric_position(point_m, *, frame_id, transform):
    """Numerical diagnostic, NEVER a production publication gateway.

    A verified bool is not authenticated calibration provenance. Even if it is
    True, this stage requires subsequent global validation (EXP035-E).
    """
    if not isinstance(transform, RigidTransform):
        raise ValueError("transform must be RigidTransform")
    value = transform.transform_point(point_m, frame_id=frame_id)
    verified = transform.calibration_verified
    return {
        "source_frame_id": transform.source_frame_id,
        "target_frame_id": transform.target_frame_id,
        "status": "READY_FOR_GLOBAL_VALIDATION" if verified else "BLOCKED_UNCALIBRATED",
        "publishable": False,
        "position_m": None,
        "reasons": ["GLOBAL_VALIDATION_REQUIRED"] if verified else ["UNVERIFIED_EXTRINSIC"],
        "diagnostics": {
            "p_target_demo_m": value.tolist(),
            "calibration_verified_claim": verified,
            "demo_only": True,
        },
    }
