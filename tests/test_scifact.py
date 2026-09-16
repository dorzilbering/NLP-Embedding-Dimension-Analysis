"""Synthetic SciFact-shaped fixtures; no network, real data or pretrained models."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pilot.scifact import build_bundle, validate_official_bundle
from pilot.task_config import task_plan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in SciFact tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")


@pytest.fixture
def tables():
    corpus = [{"_id": f"d{i}", "title": f"Title {i}", "text": f"Abstract {i}"} for i in range(5183)]
    queries = [{"_id": f"q{i}", "text": f"Synthetic claim {i}"} for i in range(1109)]
    train = [{"query-id": f"q{i}", "corpus-id": "d0", "score": 1.} for i in range(809)]
    train += [{"query-id": f"q{i}", "corpus-id": "d1", "score": 1.} for i in range(110)]
    test = [{"query-id": f"q{i}", "corpus-id": "d0", "score": 1.} for i in range(809, 1109)]
    test += [{"query-id": f"q{i}", "corpus-id": "d1", "score": 1.} for i in range(809, 848)]
    return corpus, queries, train, test


def claims():
    return {split: [{"id": i, "claim": f"AllenAI synthetic {split} claim {i}"} for i in range(count)]
            for split, count in (("train", 1261), ("validation", 450))}


def export(tables, reference=None):
    return build_bundle(*tables, "a" * 40, claims() if reference is None else reference, "b" * 40)


def test_official_structure_and_reference(tables):
    bundle = export(tables)
    assert len(bundle["corpus"]) == 5183 and len(bundle["queries"]) == 300
    assert sum(map(len, bundle["qrels"].values())) == 339
    assert bundle["corpus"][0] == {"id": "d0", "text": "Title 0\nAbstract 0"}
    assert len(bundle["reference"]) == 1711
    assert bundle["fit_source"]["dataset_id"] == "allenai/scifact"
    assert bundle["fit_source"]["revision"] == "b" * 40
    assert bundle["fit_source"]["splits"] == ["train", "validation"]
    plan = task_plan("SciFact", bundle=bundle)
    assert plan["executable"] and plan["pca_train_sentences"] == 1711
    assert task_plan("SciFact", train_count=10, bundle=bundle)["pca_train_sentences"] == 1711


@pytest.mark.parametrize("fault", ["count", "duplicate_doc", "duplicate_query", "unknown_doc", "unknown_query",
                                  "duplicate_judgment", "negative", "nan", "bool", "no_positive", "split_overlap",
                                  "bad_title"])
def test_invalid_source_rejected(tables, fault):
    corpus, queries, train, test = tables
    if fault == "count":
        corpus.pop()
    elif fault == "duplicate_doc":
        corpus[1]["_id"] = "d0"
    elif fault == "duplicate_query":
        queries[1]["_id"] = "q0"
    elif fault == "unknown_doc":
        test[0]["corpus-id"] = "missing"
    elif fault == "unknown_query":
        train[0]["query-id"] = "missing"
    elif fault == "duplicate_judgment":
        test[1] = dict(test[0])
    elif fault in ("negative", "nan", "bool"):
        test[0]["score"] = {"negative": -1, "nan": float("nan"), "bool": True}[fault]
    elif fault == "no_positive":
        test[299]["score"] = 0
    elif fault == "split_overlap":
        for row in test:
            if row["query-id"] == "q809":
                row["query-id"] = "q0"
    else:
        corpus[0]["title"] = None
    with pytest.raises(ValueError):
        export(tables)


def test_reference_independent_of_test_judgments(tables):
    first = export(tables)
    for row in tables[3]:
        row["score"] = 2.
        row["corpus-id"] = "d2" if row["corpus-id"] == "d0" else "d3"
    changed = export(tables)
    assert changed["reference"] == first["reference"]
    assert changed["fit_source"] == first["fit_source"]
    assert changed["qrels"] != first["qrels"]


def test_reference_training_deduplication_and_test_exclusion(tables):
    reference = claims()
    reference["validation"][0]["claim"] = reference["train"][0]["claim"].upper()
    reference["validation"][1]["claim"] = tables[1][809]["text"].upper()
    reference["train"][1]["claim"] = "Title 0\nAbstract 0"
    bundle = export(tables, reference)
    assert len(bundle["reference"]) == 1708
    reasons = {r["reason"] for r in bundle["fit_source"]["excluded_rows"]}
    assert reasons == {"duplicate_claim", "retrieval_test_query", "retrieval_corpus"}
    assert len(bundle["queries"]) == 300 and sum(map(len, bundle["qrels"].values())) == 339


@pytest.mark.parametrize("removed", [175, 300])
def test_insufficient_eligible_reference_blocks_pca(tables, removed):
    reference = claims()
    for i in range(removed):
        reference["validation"][i]["claim"] = tables[1][809 + i % 300]["text"]
    bundle = export(tables, reference)
    assert len(bundle["reference"]) == 1711 - removed
    with pytest.raises(ValueError, match=">1536"):
        task_plan("SciFact", bundle=bundle)
    assert task_plan("SciFact", dimensions=[3072], bundle=bundle)["executable"]


@pytest.mark.parametrize("fault", ["test", "count", "field"])
def test_invalid_allenai_partitions(tables, fault):
    reference = claims()
    if fault == "test":
        reference["test"] = []
    elif fault == "count":
        reference["train"].pop()
    else:
        reference["validation"][0]["claim"] = None
    with pytest.raises(ValueError):
        export(tables, reference)


@pytest.mark.parametrize("fault", ["revision", "split", "source", "fit_source", "reference_id", "hash"])
def test_invalid_bundle_rejected(tables, fault):
    bundle = export(tables)
    if fault == "revision":
        bundle["source"]["revision"] = "not-a-sha"
    elif fault == "split":
        bundle["source"]["evaluation_split"] = "validation"
    elif fault == "source":
        bundle["source"]["dataset_id"] = "wrong/source"
    elif fault == "fit_source":
        bundle["fit_source"]["configuration"] = "corpus"
    elif fault == "reference_id":
        bundle["reference"][0]["id"] = "q809"
    else:
        bundle["qrels"]["q809"]["d0"] = 2
    with pytest.raises(ValueError):
        validate_official_bundle(bundle)


def test_preparation_stdlib_dry_run(tmp_path):
    script = Path(__file__).resolve().parents[1] / "prepare_scifact.py"
    output = tmp_path / "unused.json"
    proc = subprocess.run([sys.executable, "-B", "-S", str(script), "--dry-run", "--output", str(output)],
                          text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["dataset"] == "mteb/scifact"
    assert not output.exists()


def test_mocked_preparation_and_runner_dry_run(tables, tmp_path, monkeypatch, capsys):
    import prepare_scifact
    import run_experiment
    calls = []
    configs = {"corpus": {"corpus": tables[0]}, "queries": {"queries": tables[1]},
               "default": {"train": tables[2], "test": tables[3]}}
    def info(identifier, revision):
        assert identifier in ("mteb/scifact", "allenai/scifact")
        return SimpleNamespace(sha=("a" if identifier == "mteb/scifact" else "b") * 40)
    def load(identifier, name, revision, **kwargs):
        if identifier == "allenai/scifact":
            assert revision == "b" * 40 and name == "claims"
            assert kwargs["streaming"] and kwargs["trust_remote_code"]
            assert kwargs["split"] in ("train", "validation")
            calls.append(kwargs["split"])
            return iter(claims()[kwargs["split"]])
        assert identifier == "mteb/scifact" and revision == "a" * 40
        calls.append(name)
        return configs[name]
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=info)))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        get_dataset_config_names=lambda *a, **kw: list(configs), load_dataset=load))
    output = tmp_path / "scifact.json"
    prepare_scifact.main(["--output", str(output)])
    assert set(calls) == set(configs) | {"train", "validation"}
    validate_official_bundle(json.loads(output.read_text()))
    capsys.readouterr()
    run_experiment.main(["--task", "SciFact", "--data", str(output), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["executable"]
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        prepare_scifact.main(["--output", str(output)])
    assert len(calls) == 5 and output.read_bytes() == before


def test_unexpected_remote_configuration_rejected_before_loading(tmp_path, monkeypatch):
    import prepare_scifact
    def forbidden(*args, **kwargs):
        raise AssertionError("No dataset load for unexpected configurations")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha="a" * 40))))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        get_dataset_config_names=lambda *a, **kw: ["unknown"], load_dataset=forbidden))
    with pytest.raises(ValueError, match="configurations"):
        prepare_scifact.main(["--output", str(tmp_path / "unused.json")])
