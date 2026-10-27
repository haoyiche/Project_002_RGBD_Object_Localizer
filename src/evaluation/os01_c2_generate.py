"""OS-01-C2: generate 20 distinct, traceable synthetic RGB-D inputs.

Run from the repository root with:
    python -m src.evaluation.os01_c2_generate

This creates input data and a manifest, NOT measured physical ground truth
and NOT a batch evaluation result.
"""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


GROUPS = {
    "normal_geometry": 6,
    "mask_perturbation": 4,
    "depth_anomaly": 4,
    "cluster_gate_pressure": 4,
    "missing_reference": 2,
}


def case_specs():
    specs = []
    normal = [
        (0.88, 0, 0), (0.92, 0, 1), (0.96, 0, -1),
        (1.04, 1, 0), (1.08, -1, 0), (1.12, 1, 1),
    ]
    for z, dv, du in normal:
        specs.append(("normal_geometry", {"z": z, "dv": dv, "du": du}))
    for drop in [(2, 4), (2, 5), (3, 3), "empty"]:
        specs.append(("mask_perturbation", {"drop": drop}))
    for mode in ["extra_zero", "nan", "inf", "all_zero"]:
        specs.append(("depth_anomaly", {"mode": mode}))
    for z, mode in [
        (1.00, "filter_all"),
        (1.01, "all_noise"),
        (1.02, "gate_reject"),
        (1.03, "filter_four"),
    ]:
        specs.append(("cluster_gate_pressure", {"z": z, "mode": mode}))
    for z in [1.07, 1.09]:
        specs.append(("missing_reference", {"z": z}))
    return specs


def make_input(group, config):
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    depth_m = np.zeros((6, 8), dtype=np.float64)
    mask = np.zeros((6, 8), dtype=np.bool_)
    z = float(config.get("z", 1.0))
    dv, du = int(config.get("dv", 0)), int(config.get("du", 0))
    r0, c0 = 2 + dv, 3 + du
    rgb[r0:r0 + 2, c0:c0 + 3] = [200, 60, 40]
    depth_m[r0:r0 + 2, c0:c0 + 3] = z
    mask[r0:r0 + 2, c0:c0 + 3] = True
    depth_m[r0, c0] = 0.0

    if group == "mask_perturbation":
        drop = config["drop"]
        if drop == "empty":
            mask[:] = False
        else:
            mask[drop[0], drop[1]] = False
    elif group == "depth_anomaly":
        mode = config["mode"]
        if mode == "extra_zero":
            depth_m[2, 4] = 0.0
        elif mode == "nan":
            depth_m[2, 5] = np.nan
        elif mode == "inf":
            depth_m[3, 3] = np.inf
        elif mode == "all_zero":
            depth_m[mask] = 0.0

    # A nominal patch-center design coordinate, computed from the intended
    # 2x3 patch geometry, not obtained by running the localization pipeline.
    ref = [((4 + du) - 3.5) * z / 100.0,
           ((2.5 + dv) - 2.5) * z / 100.0,
           z]
    return rgb, depth_m, mask, ref


def pipeline_settings(group, config):
    z = float(config.get("z", 1.0))
    # Radius scales with depth for shifted normal examples.
    radius = 0.011 * z if group == "normal_geometry" else 0.011
    params = {
        "radius_m": round(radius, 8),
        "min_neighbors": 1,
        "dbscan_eps_m": round(radius, 8),
        "dbscan_min_points": 3,
    }
    gate_params = None
    if group == "cluster_gate_pressure":
        mode = config["mode"]
        if mode == "filter_all":
            params["min_neighbors"] = 4
        elif mode == "all_noise":
            params["dbscan_min_points"] = 6
        elif mode == "gate_reject":
            gate_params = {"min_points": 6, "max_points": 20,
                           "max_extent_m": [0.04, 0.04, 0.02],
                           "max_centroid_bbox_gap_m": 0.02}
        elif mode == "filter_four":
            params["min_neighbors"] = 2
    return params, gate_params


def content_hash(rgb, depth_m, mask):
    digest = sha256()
    for arr in (rgb, depth_m, mask):
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()


def generate_dataset(project_root=Path(".")):
    root = Path(project_root)
    data_dir = root / "data/raw/synthetic/os01_c"
    manifest_path = root / "data/manifests/os01_c_samples.json"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for index, (group, config) in enumerate(case_specs(), start=1):
        sample_id = f"OS01_C_{index:03d}"
        relative_path = f"data/raw/synthetic/os01_c/{sample_id}.npz"
        rgb, depth_m, mask, reference = make_input(group, config)
        file_path = root / relative_path
        np.savez_compressed(
            file_path, rgb=rgb, depth_m=depth_m, target_mask=mask,
            fx=100.0, fy=100.0, cx=3.5, cy=2.5,
            frame_id="camera_optical_frame",
        )
        params, gate_params = pipeline_settings(group, config)
        has_reference = group != "missing_reference"
        records.append({
            "sample_id": sample_id,
            "source_kind": "synthetic",
            "scenario_group": group,
            "generation_params": config,
            "input_path": relative_path,
            "input_sha256": sha256(file_path.read_bytes()).hexdigest(),
            "content_sha256": content_hash(rgb, depth_m, mask),
            "reference_kind": "synthetic_nominal_patch_center" if has_reference else "none",
            "reference_position_m": reference if has_reference else None,
            "reference_frame_id": "camera_optical_frame" if has_reference else None,
            "pipeline_params": params,
            "gate_params": gate_params,
        })

    manifest = {
        "schema_version": "os01_c2_v1",
        "description": "Deterministic synthetic inputs; no independent physical ground truth",
        "samples": records,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                             encoding="utf-8")
    validate_manifest(manifest_path, root)
    return manifest_path


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest_path, project_root=Path(".")):
    root = Path(project_root).resolve()
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == "os01_c2_v1", "schema version mismatch")
    samples = manifest["samples"]
    _require(len(samples) == 20, "expected exactly 20 samples")
    _require(Counter(x["scenario_group"] for x in samples) == GROUPS,
             "scenario-group counts differ from the plan")
    _require(len({x["sample_id"] for x in samples}) == 20, "duplicate sample IDs")
    _require(sum(x["reference_position_m"] is None for x in samples) == 2,
             "expected exactly two missing references")
    _require(len({x["content_sha256"] for x in samples}) == 20,
             "duplicate RGB-D input contents")
    for sample in samples:
        path = (root / sample["input_path"]).resolve()
        _require(path.is_relative_to(root), "sample path escapes project root")
        _require(path.is_file(), f"input does not exist: {path}")
        _require(sha256(path.read_bytes()).hexdigest() == sample["input_sha256"],
                 f"file hash mismatch: {sample['sample_id']}")
        with np.load(path, allow_pickle=False) as data:
            rgb, depth, mask = data["rgb"], data["depth_m"], data["target_mask"]
            _require(rgb.shape == (6, 8, 3) and rgb.dtype == np.uint8,
                     "invalid RGB shape or dtype")
            _require(depth.shape == (6, 8) and depth.dtype == np.float64,
                     "invalid depth shape or dtype")
            _require(mask.shape == (6, 8) and mask.dtype == np.bool_,
                     "invalid mask shape or dtype")
            _require(content_hash(rgb, depth, mask) == sample["content_sha256"],
                     f"array content mismatch: {sample['sample_id']}")
    return manifest


def main():
    manifest_path = generate_dataset()
    manifest = validate_manifest(manifest_path)
    print("Manifest:", manifest_path)
    print("Samples:", len(manifest["samples"]))
    print("Scenario groups:", dict(Counter(s["scenario_group"] for s in manifest["samples"])))
    print("Missing references:", sum(s["reference_position_m"] is None for s in manifest["samples"]))
    print("Unique RGB-D contents:", len({s["content_sha256"] for s in manifest["samples"]}))
    print("All sample files and SHA-256 digests verified")


if __name__ == "__main__":
    main()
