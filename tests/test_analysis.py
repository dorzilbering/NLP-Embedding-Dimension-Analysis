"""Analysis checks use actual immutable inputs or isolated damaged copies."""
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from analyze_results import ROOT, changes, generate, retention_summary, unique_keys, validate

RAW = ROOT / "results/raw"


def test_complete_matrix_and_baseline_arithmetic():
    rows, _, report = validate(RAW)
    assert report["status"] == "pass" and not report["issues"]
    assert report["represented_configurations"] == 84
    assert report["metric_rows"] == 210 and report["raw_file_count"] == 20
    for row in changes(rows):
        assert row["delta_score"] == pytest.approx(row["score"] - row["native_score"])
        assert row["relative_change_pct"] == pytest.approx(100 * (row["score"] / row["native_score"] - 1))
        if row["reduction"] == "native":
            assert row["delta_score"] == 0 and row["retained_native_pct"] == 100


@pytest.mark.parametrize("fault", ["missing_file", "missing_metric", "duplicate", "nan", "bool", "dimension", "identity", "revision", "calibration"])
def test_corrupt_results_fail_closed(tmp_path, fault):
    raw = tmp_path / "raw"
    shutil.copytree(RAW, raw)
    path = raw / "Phi4-mini/STSB/results.json"
    data = json.loads(path.read_text())
    if fault == "missing_file":
        path.unlink()
    else:
        if fault == "missing_metric": data["scores"].pop()
        elif fault == "duplicate": data["scores"].append(copy.deepcopy(data["scores"][0]))
        elif fault == "nan": data["scores"][0]["score"] = float("nan")
        elif fault == "bool": data["scores"][0]["score"] = True
        elif fault == "dimension": data["scores"][0]["dimension"] = 999
        elif fault == "identity": data["scores"][0]["model"] = "Qwen3-8B"
        elif fault == "revision": data["scores"][0]["dataset_revision"] = "a" * 40
        else: data["metadata"]["calibration_hash"] = "a" * 64
        path.write_text(json.dumps(data), encoding="utf-8")
    assert validate(raw)[2]["status"] == "fail"
    with pytest.raises(ValueError): generate(raw, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_duplicate_keys_rejected():
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        json.loads('{"score": 1, "score": 2}', object_pairs_hook=unique_keys)


def test_zero_and_signed_baseline_are_not_retention_ratios():
    for baseline in (0, -.5):
        rows = [{"model": "Phi4-mini", "task": "STSB", "metric": "cosine_spearman", "dimension": d,
                 "score": baseline if d == 3072 else .2} for d in (3072, 1536)]
        assert all(r["relative_change_pct"] is None and r["retained_native_pct"] is None for r in changes(rows))


def test_retention_is_smallest_tested_dimension_for_all_metrics():
    rows, _, _ = validate(RAW)
    relative = changes(rows)
    for summary in retention_summary(relative):
        selected = [r for r in relative if r["model"] == summary["model"] and r["task"] == summary["task"]
                    and (summary["metric"] == "all_task_metrics" or r["metric"] == summary["metric"])]
        eligible = [d for d in {r["dimension"] for r in selected}
                    if all(r["retained_native_pct"] >= 95 for r in selected if r["dimension"] == d)]
        assert summary["smallest_tested_dimension"] == min(eligible)


def test_generated_outputs_and_raw_integrity(tmp_path):
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in RAW.rglob("*.json")}
    generate(RAW, tmp_path)
    assert len(list((tmp_path / "figures").glob("*.png"))) == 8
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 8
    for path in (tmp_path / "figures").glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    assert json.loads((tmp_path / "tables/validation.json").read_text())["sha256"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}


def test_output_cannot_write_into_raw():
    with pytest.raises(ValueError, match="overlap"):
        generate(RAW, RAW)
