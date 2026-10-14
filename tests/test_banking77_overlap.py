"""Synthetic official-shape overlap regressions; no dataset/model downloads."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pilot.banking77 import build_bundle, official_overlap_audit, validate_official_bundle
from pilot.task_data import fingerprint, load_bundle, text_key, validate_bundle
from training.data import eligible_training_rows, split_training_rows


@pytest.fixture
def overlapping_bundle():
    class Partition(list):
        column_names = ["text", "label"]
        features = {"label": SimpleNamespace(names=[f"intent_{i}" for i in range(77)])}
    dataset = {split: Partition({"text": f"synthetic {split} row {i}", "label": i % 77}
                               for i in range(count))
               for split, count in (("train", 10003), ("test", 3080))}
    for i in range(7):
        dataset["test"][i]["text"] = "  " + dataset["train"][i]["text"].upper() + "  "
    # Eight excluded train rows, but seven distinct canonical texts.
    dataset["train"][100] = dict(dataset["train"][0])
    return build_bundle(dataset, "a" * 40, "synthetic")


def test_known_raw_overlap_roundtrip_and_exclusion_audit(overlapping_bundle, tmp_path):
    bundle = overlapping_bundle
    original = copy.deepcopy(bundle)
    path = tmp_path / "official.json"
    path.write_text(json.dumps(bundle))
    before = path.read_bytes()
    loaded = load_bundle(path, "Banking77")
    assert validate_official_bundle(loaded) is loaded
    eligible, audit = eligible_training_rows(loaded)
    assert len(loaded["train"]) == 10003 and len(loaded["evaluation"]) == 3080
    assert audit["overlap_unique_text_count"] == 7
    assert audit["excluded_train_row_count"] == 8
    assert audit["excluded_train_ids"] == [f"train:{i}" for i in range(7)] + ["train:100"]
    expected_hash = hashlib.sha256(text_key(bundle["train"][0]["text"]).encode("utf-8")).hexdigest()
    assert audit["excluded_train_rows"][0]["canonical_text_sha256"] == expected_hash
    assert audit["excluded_train_rows"][-1]["canonical_text_sha256"] == expected_hash
    assert len(audit["overlap_text_sha256"]) == 7
    assert audit["raw_split_hashes"] == bundle["source"]["split_hashes"]
    assert audit["eligible_train_row_count"] == len(eligible) == 9995
    assert audit["exclusion_applied"] is True
    assert audit["eligible_train_hash"] == fingerprint(eligible)
    assert {text_key(r["text"]) for r in eligible}.isdisjoint({text_key(r["text"]) for r in loaded["evaluation"]})
    assert eligible == [r for r in bundle["train"] if r["id"] not in set(audit["excluded_train_ids"])]
    train, validation, split = split_training_rows(eligible)
    assert len(train) + len(validation) == len(eligible)
    assert set(split["train_ids"] + split["validation_ids"]).isdisjoint(audit["excluded_train_ids"])
    assert bundle == loaded == original and path.read_bytes() == before


def test_exclusions_use_only_test_text_not_test_labels(overlapping_bundle):
    _, before = eligible_training_rows(overlapping_bundle)
    modified = copy.deepcopy(overlapping_bundle)
    for row in modified["evaluation"]:
        row["label"] = str((int(row["label"]) + 1) % 77)
    modified["source"]["split_hashes"]["test"] = fingerprint(modified["evaluation"])
    _, after = eligible_training_rows(modified)
    for field in ("excluded_train_ids", "excluded_train_rows", "overlap_text_sha256", "eligible_train_hash"):
        assert before[field] == after[field]
    assert after["test_labels_used_for_exclusions"] is False


@pytest.mark.parametrize("fault", ["unrelated", "subset", "hash", "row_order", "count", "revision"])
def test_exception_does_not_bypass_official_integrity_or_other_sources(overlapping_bundle, fault):
    bundle = copy.deepcopy(overlapping_bundle)
    if fault == "unrelated":
        bundle["source"]["dataset_id"] = bundle["fit_source"]["dataset_id"] = "unrelated/banking77"
    elif fault == "subset":
        bundle["source"]["selection"] = "subset"
    elif fault == "hash":
        bundle["source"]["split_hashes"]["test"] = "corrupted"
    elif fault == "row_order":
        bundle["train"].reverse()
    elif fault == "count":
        bundle["train"].pop()
    else:
        bundle["source"]["revision"] = bundle["fit_source"]["revision"] = "not-a-sha"
    with pytest.raises(ValueError):
        validate_bundle(bundle, "Banking77")


@pytest.mark.parametrize("count", [1, 6, 8])
def test_unexpected_overlap_count_is_rejected(overlapping_bundle, count):
    bundle = copy.deepcopy(overlapping_bundle)
    for i in range(8):
        bundle["evaluation"][i]["text"] = (bundle["train"][i]["text"] if i < count else f"nonoverlapping test {i}")
    bundle["source"]["split_hashes"]["test"] = fingerprint(bundle["evaluation"])
    with pytest.raises(ValueError, match="Unexpected official Banking77 overlap"):
        validate_official_bundle(bundle)


def test_legacy_pca_keeps_all_rows_and_audits_overlap(overlapping_bundle, monkeypatch):
    import pilot.task_runner as runner
    from pilot.task_config import PROTOCOLS, task_plan
    # Production planning still sees the raw official counts; no historical PCA changes.
    plan = task_plan("Banking77", dimensions=[3072], bundle=overlapping_bundle)
    assert plan["data"]["fit_count"] == 10003 and plan["data"]["evaluation_count"] == 3080
    seen = []
    def classification(train_x, train_labels, test_x, test_labels, seed):
        seen.append((len(train_x), len(test_x)))
        return {"accuracy": 0.0, "macro_f1": 0.0}, ["0"] * len(test_x)
    monkeypatch.setattr(runner, "classification", classification)
    def encode(texts):
        return np.random.default_rng(42).normal(size=(len(texts), 8))
    tiny_plan = {"task": "Banking77", "model": "synthetic", "native_dimension": 8,
                 "seed": 42, "dimensions": [8, 4], "pca_components": 4,
                 "pca_train_sentences": 16, "protocol": PROTOCOLS["Banking77"]}
    original = copy.deepcopy(overlapping_bundle)
    _, _, metadata, pca = runner.evaluate_bundle(overlapping_bundle, tiny_plan, encode)
    assert seen == [(10003, 3080), (10003, 3080)]
    assert metadata["official_overlap_audit"]["overlap_unique_text_count"] == 7
    assert metadata["overlap_handling"] == "legacy_pca_preserves_all_official_train_rows"
    assert metadata["official_overlap_audit"]["exclusion_applied"] is False
    assert len(metadata["fit_ids"]) == 10003 and pca is not None
    assert "Known official Banking77" in metadata["leakage_check"]
    assert overlapping_bundle == original
