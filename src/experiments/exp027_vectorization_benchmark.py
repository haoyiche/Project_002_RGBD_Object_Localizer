import csv
from pathlib import Path
from statistics import median
from time import perf_counter_ns

import numpy as np

from src.geometry.depth_to_pointcloud import depth_to_pointcloud
from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


PARAMS = {
    "fx": 600.0,
    "fy": 600.0,
    "cx": 319.5,
    "cy": 239.5,
}

CASES = [
    (64, 64, 7),
    (240, 320, 5),
    (480, 640, 3),
]

OUTPUT = Path("results/exp027/vectorization_benchmark.csv")


def make_depth_map(height, width, rng):
    depth = rng.uniform(
        low=0.4,
        high=3.0,
        size=(height, width),
    )

    # 10% invalid depths
    invalid = rng.random((height, width)) < 0.10
    depth[invalid] = 0.0

    # Explicit invalid cases
    depth[0, 0] = np.nan
    depth[0, 1] = np.inf
    depth[0, 2] = -1.0

    return depth


def timed_call(func, depth):
    start = perf_counter_ns()

    result = func(
        depth_m=depth,
        **PARAMS,
    )

    elapsed_ms = (perf_counter_ns() - start) / 1e6

    return elapsed_ms, result


def benchmark_case(height, width, repeats, rng):
    depth = make_depth_map(height, width, rng)

    # Check correctness before measuring performance
    points_v0 = depth_to_pointcloud(
        depth_m=depth,
        **PARAMS,
    )

    points_v1 = depth_to_pointcloud_vectorized(
        depth_m=depth,
        **PARAMS,
    )

    expected_valid = int(
        np.count_nonzero(
            np.isfinite(depth) & (depth > 0)
        )
    )

    assert points_v0.shape == (expected_valid, 3)
    assert points_v1.shape == points_v0.shape

    np.testing.assert_allclose(
        points_v1,
        points_v0,
        rtol=0,
        atol=1e-12,
    )

    # Warm-up
    depth_to_pointcloud(depth_m=depth, **PARAMS)
    depth_to_pointcloud_vectorized(depth_m=depth, **PARAMS)

    times_v0 = []
    times_v1 = []

    for i in range(repeats):
        # Alternate execution order
        if i % 2 == 0:
            t0, _ = timed_call(depth_to_pointcloud, depth)
            t1, _ = timed_call(
                depth_to_pointcloud_vectorized, depth
            )
        else:
            t1, _ = timed_call(
                depth_to_pointcloud_vectorized, depth
            )
            t0, _ = timed_call(depth_to_pointcloud, depth)

        times_v0.append(t0)
        times_v1.append(t1)

    v0_ms = median(times_v0)
    v1_ms = median(times_v1)

    speedup = v0_ms / v1_ms

    print()
    print(f"Depth map: {height} x {width}")
    print(f"Total pixels: {height * width}")
    print(f"Valid points: {expected_valid}")
    print(f"Repeats: {repeats}")
    print(f"V0 median: {v0_ms:.4f} ms")
    print(f"V1 median: {v1_ms:.4f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print("Correctness: PASS")

    return {
        "height": height,
        "width": width,
        "total_pixels": height * width,
        "valid_points": expected_valid,
        "repeats": repeats,
        "v0_median_ms": v0_ms,
        "v1_median_ms": v1_ms,
        "speedup": speedup,
    }


def main():
    print("=== EXP027 Vectorization Benchmark ===")

    rng = np.random.default_rng(42)
    rows = []

    for height, width, repeats in CASES:
        row = benchmark_case(
            height=height,
            width=width,
            repeats=repeats,
            rng=rng,
        )

        rows.append(row)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== BENCHMARK SUMMARY ===")

    for row in rows:
        print(
            f"{row['height']}x{row['width']} | "
            f"V0={row['v0_median_ms']:.3f} ms | "
            f"V1={row['v1_median_ms']:.3f} ms | "
            f"Speedup={row['speedup']:.2f}x"
        )

    print()
    print(f"Saved: {OUTPUT}")
    print("EXP027 vectorization benchmark COMPLETED.")


if __name__ == "__main__":
    main()
