"""OS-01-C2 generation/manifest acceptance tests."""

import json

import pytest

from src.evaluation.os01_c2_generate import (
    GROUPS,
    generate_dataset,
    validate_manifest,
)


def test_generates_20_distinct_traceable_samples(tmp_path):
    manifest_path = generate_dataset(tmp_path)
    manifest = validate_manifest(manifest_path, tmp_path)
    samples = manifest["samples"]

    assert len(samples) == 20
    assert {g: sum(s["scenario_group"] == g for s in samples)
            for g in GROUPS} == GROUPS
    assert len({s["sample_id"] for s in samples}) == 20
    assert len({s["content_sha256"] for s in samples}) == 20
    assert sum(s["reference_position_m"] is None for s in samples) == 2
    assert all(s["source_kind"] == "synthetic" for s in samples)
    assert all(s["reference_kind"] != "physical_ground_truth"
               for s in samples)


def test_tampered_manifest_hash_is_rejected(tmp_path):
    manifest_path = generate_dataset(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["samples"][0]["input_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        validate_manifest(manifest_path, tmp_path)
