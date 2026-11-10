"""T1-07-A E1-B: fixed bias, frame-shared noise and IID noise."""

import json
from pathlib import Path

import numpy as np

SEED = 20261109
TRIALS = 20_000

BIAS_MM = 10.0
SHARED_STD_MM = 1.0
IID_STD_MM = 2.0

rng = np.random.default_rng(SEED)

# 每次试验模拟一帧。
# c 在同一帧的所有点之间共享；不同帧独立。
shared_noise_mm = rng.normal(
    loc=0.0,
    scale=SHARED_STD_MM,
    size=TRIALS,
)

results = []

for n in (100, 400):
    independent_noise_mm = rng.normal(
        loc=0.0,
        scale=IID_STD_MM,
        size=(TRIALS, n),
    )

    centroid_error_mm = (
        BIAS_MM
        + shared_noise_mm
        + independent_noise_mm.mean(axis=1)
    )

    predicted_mean_mm = BIAS_MM
    predicted_std_mm = np.sqrt(
        SHARED_STD_MM**2 + IID_STD_MM**2 / n
    )

    observed_mean_mm = float(
        centroid_error_mm.mean()
    )
    observed_std_mm = float(
        centroid_error_mm.std(ddof=1)
    )

    assert abs(
        observed_mean_mm - predicted_mean_mm
    ) < 0.06, "MEAN_BIAS_MISMATCH"

    assert abs(
        observed_std_mm / predicted_std_mm - 1
    ) < 0.05, "CORRELATED_STD_MISMATCH"

    assert observed_std_mm > 0.9, (
        "SHARED_NOISE_FLOOR_MISSING"
    )

    print(
        f"N={n}: "
        f"mean_theory={predicted_mean_mm:.6f} mm, "
        f"mean_observed={observed_mean_mm:.6f} mm, "
        f"std_theory={predicted_std_mm:.6f} mm, "
        f"std_observed={observed_std_mm:.6f} mm"
    )

    results.append({
        "N": n,
        "predicted_mean_mm": predicted_mean_mm,
        "observed_mean_mm": observed_mean_mm,
        "predicted_std_mm": float(predicted_std_mm),
        "observed_std_mm": observed_std_mm,
    })

ratio = (
    results[1]["observed_std_mm"]
    / results[0]["observed_std_mm"]
)

print("STD_RATIO_N400_OVER_N100:", ratio)

report = {
    "schema": "t1-07-a-e1-b-bias-shared-iid-v1",
    "seed": SEED,
    "trials": TRIALS,
    "bias_mm": BIAS_MM,
    "shared_std_mm": SHARED_STD_MM,
    "iid_std_mm": IID_STD_MM,
    "results": results,
    "std_ratio": float(ratio),
    "reference_kind": "SYNTHETIC_NOISE_SIMULATION",
    "physical_sensor_noise_measured": False,
    "physical_accuracy_claimed": False,
}

output = Path(
    "outputs/t1_07_a/e1_b_bias_shared_noise.json"
)
output.parent.mkdir(parents=True, exist_ok=True)

# 禁止覆盖旧实验结果
with output.open("x", encoding="utf-8") as f:
    json.dump(
        report, f,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    f.write("\n")

print("OUTPUT:", output)
print("T107_A_E1_B_SHARED_NOISE_PASS")
