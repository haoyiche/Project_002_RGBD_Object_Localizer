import numpy as np


def dbscan_numpy(
    points: np.ndarray,
    eps: float,
    min_points: int,
) -> np.ndarray:
    """
    NumPy DBSCAN reference implementation.

    Args:
        points: (N, 3), coordinates in meters
        eps: neighborhood radius in meters
        min_points: minimum neighborhood size,
                    including the point itself

    Returns:
        labels: (N,), int64
                -1 means noise
                0, 1, ... mean cluster IDs
    """

    # =========================================================
    # TODO 1: Validate Shape, finite values and parameters
    # =========================================================
    try:
        pts = np.asarray(points, dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"points must be numeric: {e}")

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")

    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"eps must be finite positive, got {eps}")

    if isinstance(min_points, bool):
        raise ValueError("min_points must be int, not bool")
    if not isinstance(min_points, (int, np.integer)):
        raise ValueError(f"min_points must be int, got {type(min_points)}")
    if min_points < 1:
        raise ValueError(f"min_points must be >= 1, got {min_points}")

    if not np.isfinite(pts).all():
        raise ValueError("points must contain only finite values")

    N = pts.shape[0]

    # =========================================================
    # TODO 2: Handle the empty point cloud
    # =========================================================
    if N == 0:
        return np.empty((0,), dtype=np.int64)

    # =========================================================
    # TODO 3: Calculate the pairwise distance matrix, shape (N, N)
    # =========================================================
    # diff[i, j] = pts[i] - pts[j]
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.hypot(
    np.hypot(diff[:, :, 0], diff[:, :, 1]),
    diff[:, :, 2],)

    # =========================================================
    # TODO 4: Construct eps-neighborhood for each point (include itself)
    # =========================================================
    # dist[i, i] = 0 <= eps 恒成立，所以邻居自动包含自身
    neighbor_mask = dist <= eps
    neighbors = [np.nonzero(neighbor_mask[i])[0] for i in range(N)]

    # 预先计算每个点是否 Core
    is_core = np.array(
        [neighbors[i].size >= min_points for i in range(N)],
        dtype=bool,
    )

    # =========================================================
    # TODO 5: Distinguish UNVISITED and NOISE states
    # =========================================================
    UNVISITED = -2
    NOISE     = -1

    labels = np.full(N, UNVISITED, dtype=np.int64)

    # =========================================================
    # TODO 6: Scan points and start clusters at core points
    # =========================================================
    cluster_id = 0

    for i in range(N):
        if labels[i] != UNVISITED:
            continue

        # 非核心点：暂时标记为 NOISE，后续可能被吸收为 Border
        if not is_core[i]:
            labels[i] = NOISE
            continue

        # 核心点：开启新簇
        labels[i] = cluster_id

        # =====================================================
        # TODO 7: Expand cluster through core points
        # Border points join a cluster but do not expand it
        # =====================================================
        queue = list(neighbors[i])
        in_queue = np.zeros(N, dtype=bool)
        in_queue[queue] = True

        while queue:
            j = queue.pop()

            # 如果 j 之前被标为 NOISE，现在被 Core 吸收为 Border
            if labels[j] == NOISE or labels[j] == UNVISITED:
                labels[j] = cluster_id

            # Border 点加入簇，但不继续扩展
            if not is_core[j]:
                continue

            # Core 点：把它的邻居加入队列
            for k in neighbors[j]:
                if not in_queue[k]:
                    in_queue[k] = True
                    queue.append(k)

        cluster_id += 1

    # =========================================================
    # TODO 8: Return labels as int64
    # =========================================================
    return labels.astype(np.int64)