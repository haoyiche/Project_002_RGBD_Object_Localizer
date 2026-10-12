from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree


def radius_outlier_filter(points, radius, min_neighbors):
    """
    Remove isolated points using radius neighbors.

    Contract
    --------
    points:
        [N, 3], finite numeric XYZ, empty is allowed.
    radius:
        Finite positive real scalar.
        Unit is the same as point coordinates.
    min_neighbors:
        Positive integer, excluding bool.

    Neighborhood:
        Euclidean distance <= radius.
        The point's own index is excluded.
        Duplicate coordinates at different indices
        count as distinct neighbors.

    Filtering:
        One pass on the original input.
        Original point order is preserved.

    Returns
    -------
    dict:
        filtered_points: [M, 3], float64
        keep_mask:       [N], bool
        neighbor_counts: [N], int64
        kept_indices:    [M], integer
    """

    # 1. Convert point cloud
    try:
        xyz = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "points must contain numeric XYZ values"
        ) from exc

    # 2. Validate shape
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)"
        )

    # 3. Validate coordinates
    if not np.isfinite(xyz).all():
        raise ValueError(
            "points must contain only finite values"
        )

    # 4. Validate radius
    if (
        isinstance(radius, (bool, np.bool_))
        or not isinstance(radius, Real)
    ):
        raise ValueError(
            "radius must be a finite positive real scalar"
        )

    try:
        r = float(radius)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "radius must be a finite positive real scalar"
        ) from exc

    if not np.isfinite(r) or r <= 0:
        raise ValueError(
            "radius must be finite and > 0"
        )

    # 5. Validate min_neighbors
    if (
        isinstance(min_neighbors, (bool, np.bool_))
        or not isinstance(min_neighbors, Integral)
    ):
        raise ValueError(
            "min_neighbors must be a positive integer"
        )

    k = int(min_neighbors)

    if k <= 0:
        raise ValueError(
            "min_neighbors must be >= 1"
        )

    n = xyz.shape[0]

    # 6. Initialize neighbor counts
    counts = np.zeros(n, dtype=np.int64)

    # 7. Spatial neighbor search
    if n >= 2:
        tree = cKDTree(xyz)

        # Slightly expand candidate search to reduce
        # boundary-rounding misses.
        query_radius = np.nextafter(r, np.inf)

        for i in range(n):
            candidates = tree.query_ball_point(
                xyz[i],
                query_radius,
                p=2,
                eps=0,
            )

            candidate_indices = np.asarray(
                candidates,
                dtype=np.intp,
            )

            # Exclude the point's own INDEX only
            other_indices = candidate_indices[
                candidate_indices != i
            ]

            if other_indices.size == 0:
                continue

            # Apply exact contract to candidates
            distances = np.linalg.norm(
                xyz[other_indices] - xyz[i],
                axis=1,
            )

            counts[i] = np.count_nonzero(
                distances <= r
            )

    # 8. One-pass threshold selection
    keep_mask = counts >= k

    # 9. Preserve original index order
    kept_indices = np.flatnonzero(keep_mask)

    # 10. Build filtered output
    filtered_points = xyz[keep_mask].copy()

    return {
        "filtered_points": filtered_points,
        "keep_mask": keep_mask,
        "neighbor_counts": counts,
        "kept_indices": kept_indices,
    }
