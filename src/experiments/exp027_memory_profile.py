import gc
import tracemalloc
from statistics import median

import numpy as np

from src.geometry.depth_to_pointcloud import depth_to_pointcloud
from src.geometry.depth_to_pointcloud_vectorized import (
    depth_to_pointcloud_vectorized,
)


PARAMS = dict(
    fx=600.0,
    fy=600.0,
    cx=319.5,
    cy=239.5,
)

REPEATS = 3


def make_inputs():
    rng = np.random.default_rng(42)

    small = rng.uniform(
        0.4, 3.0, size=(240, 320)
    )

    invalid = rng.random(small.shape) < 0.10
    small[invalid] = 0.0

    small[0, 0] = np.nan
    small[0, 1] = np.inf
    small[0, 2] = -1.0

    # Preserve exactly the same valid-depth ratio.
    large = np.repeat(
        np.repeat(small, 2, axis=0),
        2,
        axis=1,
    )

    return [small, large]


def measure_once(func, depth):
    gc.collect()

    tracemalloc.start()

    try:
        points = func(
            depth_m=depth,
            **PARAMS,
        )

        current, peak = tracemalloc.get_traced_memory()
        output_shape = points.shape

    finally:
        tracemalloc.stop()

    del points

    return current, peak, output_shape


def profile_function(func, depth):
    peaks = []
    retained = []

    for _ in range(REPEATS):
        current, peak, shape = measure_once(
            func, depth
        )

        peaks.append(peak / (1024 ** 2))
        retained.append(current / (1024 ** 2))

    return {
        "shape": shape,
        "peak_mib": median(peaks),
        "retained_mib": median(retained),
    }


def main():
    print("=== EXP027 Memory Profiling ===")
    print("Metric: tracemalloc tracked allocations")

    for depth in make_inputs():
        height, width = depth.shape

        # Verify correctness outside profiling.
        points_v0 = depth_to_pointcloud(
            depth_m=depth,
            **PARAMS,
        )

        points_v1 = depth_to_pointcloud_vectorized(
            depth_m=depth,
            **PARAMS,
        )

        assert points_v0.shape == points_v1.shape

        np.testing.assert_allclose(
            points_v0,
            points_v1,
            rtol=0,
            atol=1e-12,
        )

        valid_count = len(points_v0)

        del points_v0, points_v1
        gc.collect()

        # Warm-up outside profiling.
        warm_v0 = depth_to_pointcloud(
            depth_m=depth,
            **PARAMS,
        )
        warm_v1 = depth_to_pointcloud_vectorized(
            depth_m=depth,
            **PARAMS,
        )

        del warm_v0, warm_v1
        gc.collect()

        v0 = profile_function(
            depth_to_pointcloud, depth
        )

        v1 = profile_function(
            depth_to_pointcloud_vectorized, depth
        )

        print()
        print(f"Resolution: {height}x{width}")
        print(f"Valid points: {valid_count}")

        print(
            f"V0 traced peak: "
            f"{v0['peak_mib']:.3f} MiB"
        )
        print(
            f"V1 traced peak: "
            f"{v1['peak_mib']:.3f} MiB"
        )

        print(
            f"V0 retained at return: "
            f"{v0['retained_mib']:.3f} MiB"
        )
        print(
            f"V1 retained at return: "
            f"{v1['retained_mib']:.3f} MiB"
        )

        ratio = (
            v0["peak_mib"] / v1["peak_mib"]
            if v1["peak_mib"] > 0
            else float("nan")
        )

        print(f"Peak ratio V0/V1: {ratio:.2f}x")
        print("Geometry consistency: PASS")

    print()
    print("EXP027 memory profiling COMPLETED.")


if __name__ == "__main__":
    main()
