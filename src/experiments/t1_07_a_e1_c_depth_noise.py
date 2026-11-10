"""T1-07-A E1-C: synthetic depth-dependent heteroscedastic noise."""

import json
from pathlib import Path

import numpy as np


def sigma_depth_mm(depths_m):
    z = np.asarray(depths_m, dtype=np.float64)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("INVALID_DEPTH_SHAPE")
    if not np.isfinite(z).all() or np.any(z <= 0):
        raise ValueError("INVALID_DEPTH_VALUES")

    # sigma(Z) = 2 * (Z / 1 meter)^2 millimeters
    return 2.0 * z**2


def centroid_std_mm(depths_m):
    sigmas = sigma_depth_mm(depths_m)
    return float(np.sqrt(np.sum(sigmas**2)) / len(sigmas))


def main():
    rng = np.random.default_rng(20261109)
    trials = 20_000

    cases = {
        "near_100": np.full(100, 1.0),
        "far_100": np.full(100, 2.0),
        "far_400": np.full(400, 2.0),
        "mixed_100": np.concatenate((
            np.full(50, 1.0),
            np.full(50, 2.0),
        )),
    }

    results = {}

    for name, depths_m in cases.items():
        sigmas = sigma_depth_mm(depths_m)
        predicted = centroid_std_mm(depths_m)

        # Each row is one independent simulated frame.
        noise_mm = rng.normal(
            size=(trials, len(depths_m))
        ) * sigmas[None, :]

        # Error relative to the original, fixed-depth centroid.
        centroid_errors_mm = noise_mm.mean(axis=1)

        observed = float(centroid_errors_mm.std(ddof=1))
        observed_mean = float(centroid_errors_mm.mean())

        assert abs(observed / predicted - 1) < 0.05
        assert abs(observed_mean) < 5 * predicted / np.sqrt(trials)

        results[name] = {
            "N": len(depths_m),
            "mean_depth_m": float(depths_m.mean()),
            "predicted_std_mm": predicted,
            "observed_std_mm": observed,
            "observed_mean_error_mm": observed_mean,
        }

        print(
            f"{name}: theory={predicted:.6f} mm, "
            f"observed={observed:.6f} mm"
        )

    assert np.isclose(results["near_100"]["predicted_std_mm"], 0.2)
    assert np.isclose(results["far_100"]["predicted_std_mm"], 0.8)
    assert np.isclose(results["far_400"]["predicted_std_mm"], 0.4)
    assert np.isclose(
        results["mixed_100"]["predicted_std_mm"],
        np.sqrt(3400) / 100,
    )

    # Naively substituting average depth underestimates this case.
    naive_mixed_std_mm = 2.0 * 1.5**2 / np.sqrt(100)
    assert naive_mixed_std_mm < results["mixed_100"]["predicted_std_mm"]

    output = Path("outputs/t1_07_a/e1_c_depth_noise.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "schema": "t1-07-a-e1-c-heteroscedastic-v1",
        "seed": 20261109,
        "trials": trials,
        "model": "sigma_mm=2*(Z_m/1m)^2",
        "results": results,
        "naive_mixed_std_mm": naive_mixed_std_mm,
        "physical_sensor_noise_measured": False,
        "physical_accuracy_claimed": False,
    }

    # Never overwrite a previously frozen experiment report.
    with output.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")

    print("OUTPUT:", output)
    print("T107_A_E1_C_HETEROSCEDASTIC_PASS")


if __name__ == "__main__":
    main()
