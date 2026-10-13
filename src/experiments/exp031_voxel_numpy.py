"""EXP031: NumPy reference implementation of legacy Open3D voxel downsampling.

For ordinary, finite, nonempty point clouds, use the Open3D 0.20.0
legacy grid-origin convention: origin = points.min(axis=0) - voxel_size / 2.
The implementation also returns per-voxel counts, which Open3D's simple
voxel_down_sample output does not expose directly.
"""

from numbers import Real
import numpy as np


def voxel_downsample_numpy(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group 3D points by voxel and average XYZ and RGB per voxel.

    Args:
        points: (N, 3) finite XYZ values, in meters in this experiment.
        colors: (N, 3) finite RGB values, usually in [0, 1].
        voxel_size: positive finite scalar in the same units as points.

    Returns:
        voxel_keys: (M, 3) int64, lexicographically ordered voxel indices.
        xyz_out: (M, 3) float64, per-voxel mean of XYZ.
        rgb_out: (M, 3) float64, per-voxel mean of RGB.
        counts: (M,) int64, number of input points per nonempty voxel.

    Empty (0, 3) inputs produce correctly shaped empty outputs.
    Very large indices are rejected to avoid unsafe float-to-int conversion.
    """
    if isinstance(voxel_size, (bool, np.bool_)) or not isinstance(voxel_size, Real):
        raise ValueError("voxel_size must be a real numeric scalar (not bool)")
    size = float(voxel_size)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("voxel_size must be finite and > 0")

    # Reject complex inputs instead of silently dropping imaginary components.
    if np.iscomplexobj(points) or np.iscomplexobj(colors):
        raise ValueError("points and colors must be real-valued")
    try:
        pts = np.asarray(points, dtype=np.float64)
        cols = np.asarray(colors, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("points and colors must be convertible to float64") from exc

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")
    if cols.ndim != 2 or cols.shape[1] != 3:
        raise ValueError(f"colors must have shape (N,3), got {cols.shape}")
    if len(pts) != len(cols):
        raise ValueError("points and colors must have equal row counts")
    if not np.isfinite(pts).all() or not np.isfinite(cols).all():
        raise ValueError("points and colors must contain only finite values")

    if len(pts) == 0:
        return (
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        origin = pts.min(axis=0) - size / 2.0
        key_float = np.floor((pts - origin) / size)
    # Conservative bound: float64 can represent every integer exactly only
    # up to 2**53. Stay below 2**52 for a simple, safe reference implementation.
    if (
        not np.isfinite(origin).all()
        or not np.isfinite(key_float).all()
        or np.any(np.abs(key_float) >= 2**52)
    ):
        raise ValueError("voxel index is non-finite or too large for safe conversion")
    keys = key_float.astype(np.int64)

    voxel_keys, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    m = len(voxel_keys)
    xyz_sums = np.zeros((m, 3), dtype=np.float64)
    rgb_sums = np.zeros((m, 3), dtype=np.float64)
    np.add.at(xyz_sums, inverse, pts)
    np.add.at(rgb_sums, inverse, cols)
    xyz_out = xyz_sums / counts[:, None]
    rgb_out = rgb_sums / counts[:, None]
    return voxel_keys, xyz_out, rgb_out, counts
