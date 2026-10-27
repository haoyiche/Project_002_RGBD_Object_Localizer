"""OS-01-C2: actual end-to-end 20-sample evaluation tests."""

import csv
import json

import pytest

from src.evaluation.os01_c2_generate import generate_dataset
from src.evaluation.os01_c2_batch_evaluate import run_batch


def test_batch_evaluates_all_20_without_hiding_exclusions(tmp_path):
    generate_dataset(tmp_path)
    rows, summary, csv_path, json_path = run_batch(project_root=tmp_path)

    assert len(rows) == 20
    assert len({r["sample_id"] for r in rows}) == 20
    assert summary["total_samples"] == 20
    assert summary["missing_reference_count"] == 2
    assert summary["geometry_pass_count"] == sum(r["geometry_passed"] for r in rows)
    assert summary["evaluable_count"] == sum(r["eligible"] for r in rows)
    assert summary["geometry_nonpass_count"] == 20 - summary["geometry_pass_count"]
    assert summary["excluded_count"] == 20 - summary["evaluable_count"]
    assert summary["geometry_pass_rate"] == pytest.approx(summary["geometry_pass_count"] / 20)
    assert summary["evaluable_rate"] == pytest.approx(summary["evaluable_count"] / 20)
    assert summary["physical_accuracy_claimed"] is False
    assert all(r["publishable"] is False for r in rows)
    assert all(r["error_mm"] is None for r in rows if not r["eligible"])
    assert all(r["error_mm"] is not None for r in rows if r["eligible"])
    assert summary["error_stats_mm"]["mean"] is not None
    assert 0 <= summary["error_stats_mm"]["median"] <= summary["error_stats_mm"]["max"]

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 20
    assert {r["sample_id"] for r in csv_rows} == {r["sample_id"] for r in rows}
    assert json.loads(json_path.read_text(encoding="utf-8")) == summary


def test_batch_distinguishes_filter_noise_gate_and_missing_reference(tmp_path):
    manifest_path = generate_dataset(tmp_path)
    rows, summary, _, _ = run_batch(project_root=tmp_path, manifest_path=manifest_path)
    by_mode = {
        spec["generation_params"]["mode"]: row
        for spec, row in zip(json.loads(manifest_path.read_text(encoding="utf-8"))["samples"], rows)
        if "mode" in spec["generation_params"]
    }
    assert by_mode["filter_all"]["pipeline_status"] == "NO_POINTS_AFTER_FILTER"
    assert by_mode["all_noise"]["pipeline_status"] == "NO_CLUSTERS"
    assert by_mode["gate_reject"]["pipeline_status"] == "GEOMETRY_GATE_REJECTED"
    assert all(r["exclusion_reason"] == "MISSING_REFERENCE"
               for r in rows if r["scenario_group"] == "missing_reference")
    assert summary["pipeline_status_counts"]["NO_CLUSTERS"] >= 1
