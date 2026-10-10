import numpy as np


def compute_pointcloud_statistics(points):
    """
    Compute geometric statistics for a 3D point cloud.

    Parameters
    ----------
    points:
        Non-empty [N, 3] array of finite XYZ coordinates.
        Coordinates are assumed to be in meters.

    Returns
    -------
    dict:
        centroid:      [3] float64
        min_bound:     [3] float64
        max_bound:     [3] float64
        bbox_center:   [3] float64
        bbox_extent:   [3] float64
        median_center: [3] float64

    Notes
    -----
    No outlier removal is performed.
    """

    # 1. Convert input to float64
    try:
        xyz = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "points must contain numeric XYZ coordinates"
        ) from exc

    # 2. Shape validation
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)"
        )

    # 3. Empty point cloud
    if xyz.shape[0] == 0:
        raise ValueError(
            "points must not be empty"
        )

    # 4. Finite-value validation
    if not np.isfinite(xyz).all():
        raise ValueError(
            "points must contain only finite values"
        )

    # 5. Centroid
    centroid = np.mean(xyz, axis=0)

    # 6. AABB boundaries
    min_bound = np.min(xyz, axis=0)
    max_bound = np.max(xyz, axis=0)

    # 7. AABB center and extent
    bbox_center = (min_bound + max_bound) / 2.0
    bbox_extent = max_bound - min_bound

    # 8. Coordinate-wise median
    median_center = np.median(xyz, axis=0)

    # 9. Return geometric statistics
    return {
        "centroid": centroid,
        "min_bound": min_bound,
        "max_bound": max_bound,
        "bbox_center": bbox_center,
        "bbox_extent": bbox_extent,
        "median_center": median_center,
    }
