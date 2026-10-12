import statistics
import time

import numpy as np

from src.experiments.exp030_radius_minimal import (
    radius_filter_minimal,
)
from src.geometry.pointcloud_filter import (
    radius_outlier_filter,
)


SEED = 20261012
REPEATS = 5


def run_numpy(points, radius, k):
    filtered, counts, keep, _ = radius_filter_minimal(
        points, radius, k
    )
    return filtered, counts, keep, np.flatnonzero(keep)


def run_kdtree(points, radius, k):
    result = radius_outlier_filter(points, radius, k)
    return (
        result["filtered_points"],
        result["neighbor_counts"],
        result["keep_mask"],
        result["kept_indices"],
    )


def verify_same_results(points, radius, k):
    reference = run_numpy(points, radius, k)
    optimized = run_kdtree(points, radius, k)

    for expected, actual in zip(reference, optimized):
        np.testing.assert_array_equal(actual, expected)

    return optimized[1], optimized[2]


def timed_call(func):
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000.0


def benchmark_case(name, points, radius, k):
    n = len(points)

    # Verify correctness BEFORE timing.
    counts, keep = verify_same_results(
        points, radius, k
    )

    numpy_func = lambda: run_numpy(points, radius, k)
    tree_func = lambda: run_kdtree(points, radius, k)

    # Warm-up outside measurements.
    numpy_func()
    tree_func()

    numpy_times = []
    tree_times = []

    # Alternate execution order to reduce ordering bias.
    for repeat in range(REPEATS):
        if repeat % 2 == 0:
            numpy_times.append(timed_call(numpy_func))
            tree_times.append(timed_call(tree_func))
        else:
            tree_times.append(timed_call(tree_func))
            numpy_times.append(timed_call(numpy_func))

    numpy_ms = statistics.median(numpy_times)
    tree_ms = statistics.median(tree_times)

    # Analytical estimate, not measured peak RSS.
    estimated_numpy_mb = (
        n * n * (3 * 8 + 8 + 1) / 1_000_000
    )

    ratio = numpy_ms / tree_ms if tree_ms > 0 else float("inf")

    print(f"\n--- {name} ---")
    print("N:", n)
    print("Radius:", radius)
    print("Min neighbors:", k)
    print("Mean neighbor count:", round(float(counts.mean()), 2))
    print("Kept points:", int(keep.sum()))
    print(f"NumPy median time:  {numpy_ms:.3f} ms")
    print(f"KD-Tree median time: {tree_ms:.3f} ms")
    print(f"NumPy / KD-Tree time ratio: {ratio:.2f}x")
    print(
        "NumPy main-array estimate:",
        f"{estimated_numpy_mb:.2f} MB"
    )
    print("Independent result consistency: PASS")


def main():
    print("=== EXP030 Radius Filter Benchmark ===")
    print("Seed:", SEED)
    print("Repeats:", REPEATS)
    print("Time unit: ms")
    print("Memory estimate: theoretical, not peak RSS")

    rng = np.random.default_rng(SEED)

    # Sparse-neighborhood scenarios.
    # Expand volume with N to keep average density
    # approximately constant.
    for n in [200, 300, 500, 1000]:
        side = (n / 1000.0) ** (1.0 / 3.0)

        points = rng.uniform(
            low=0.0,
            high=side,
            size=(n, 3),
        ).astype(np.float64)

        benchmark_case(
            name=f"Sparse N={n}",
            points=points,
            radius=0.1,
            k=2,
        )

    # Dense scenario: every point is within radius
    # of every other point.
    dense_n = 300

    dense_points = rng.uniform(
        low=0.0,
        high=1.0,
        size=(dense_n, 3),
    ).astype(np.float64)

    dense_radius = 2.0

    dense_counts, _ = verify_same_results(
        dense_points,
        dense_radius,
        2,
    )

    np.testing.assert_array_equal(
        dense_counts,
        np.full(dense_n, dense_n - 1),
    )

    benchmark_case(
        name="Dense N=300",
        points=dense_points,
        radius=dense_radius,
        k=2,
    )

    print("\nDense Neighbor Reference: PASS")
    print("EXP030 Performance Benchmark: PASS")


if __name__ == "__main__":
    main()
