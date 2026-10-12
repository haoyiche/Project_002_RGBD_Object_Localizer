from numbers import Real

import numpy as np


def radius_filter_independent(points, radius, min_neighbors):
    """Independent NumPy Radius Outlier Removal."""

    # Validate points
    try:
        pts = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("points must be numeric") from exc

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if not np.isfinite(pts).all():
        raise ValueError("points must be finite")

    # Validate radius
    if (
        isinstance(radius, (bool, np.bool_))
        or not isinstance(radius, Real)
    ):
        raise ValueError("radius must be a positive real scalar")

    radius = float(radius)

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and > 0")

    # Validate min_neighbors
    if (
        isinstance(min_neighbors, (bool, np.bool_))
        or not isinstance(min_neighbors, (int, np.integer))
    ):
        raise ValueError("min_neighbors must be a positive integer")

    if min_neighbors < 1:
        raise ValueError("min_neighbors must be >= 1")

    n = pts.shape[0]

    # Independent explicit-loop implementation
    neighbor_counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        count = 0

        for j in range(n):
            if i == j:
                continue

            diff = pts[i] - pts[j]
            distance = np.sqrt(np.dot(diff, diff))

            if distance <= radius:
                count += 1

        neighbor_counts[i] = count

    # One-pass filtering
    keep_mask = neighbor_counts >= min_neighbors
    kept_indices = np.flatnonzero(keep_mask)
    filtered_points = pts[kept_indices].copy()

    return {
        "filtered_points": filtered_points,
        "neighbor_counts": neighbor_counts,
        "keep_mask": keep_mask,
        "kept_indices": kept_indices,
    }
