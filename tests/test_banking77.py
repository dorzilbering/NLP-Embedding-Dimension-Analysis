"""Official-shape synthetic exports only; no real dataset or model downloads."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from pilot.banking77 import build_bundle, validate_official_bundle, SPLIT_COUNTS
from pilot.task_config import task_plan


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network forbidden in Banking77 tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")


class Partition(list):
    column_names = ["text", "label"]
    features = {"label": SimpleNamespace(names=[f"intent_{i}" for i in range(77)])}


@pytest.fixture
def dataset():
    return {split: Partition({"text": f"synthetic {split} text {i}", "label": i % 77}
                             for i in range(count)) for split, count in SPLIT_COUNTS.items()}


def export(dataset):
    return build_bundle(dataset, "a" * 40, "fixture-config")


def test_full_export_and_plan(dataset):
    data = export(dataset)
    assert len(data["train"]) == 10003 and len(data["evaluation"]) == 3080
    assert data["train"][0]["text"] == dataset["train"][0]["text"]
    assert data["evaluation"][-1]["id"] == "test:3079"
    plan = task_plan("Banking77", bundle=data)
    assert plan["executable"] and plan["pca_components"] == 1536
    assert plan["protocol"]["evaluation_split"] == "test"
    assert plan["protocol"]["classifier"] == dict(C=1., solver="lbfgs", max_iter=1000, tol=1e-4, class_weight=None)


@pytest.mark.parametrize("fault", ["count", "schema", "label", "overlap", "classes", "splits", "conflict"])
def test_bad_source_rejected(dataset, fault):
    if fault == "count":
        dataset["train"].pop()
    elif fault == "schema":
        dataset["test"].column_names = ["text", "intent"]
    elif fault == "label":
        dataset["test"][0]["label"] = 77
    elif fault == "overlap":
        dataset["test"][0]["text"] = dataset["train"][0]["text"].upper()
    elif fault == "classes":
        dataset["test"].features = {"label": SimpleNamespace(names=["bad"] * 77)}
    elif fault == "splits":
        dataset["validation"] = dataset.pop("test")
    else:
        dataset["train"][1]["text"] = dataset["train"][0]["text"]
    with pytest.raises(ValueError):
        export(dataset)


@pytest.mark.parametrize("fault", ["revision", "source", "split", "subset", "hash", "row_order"])
def test_invalid_export_rejected(dataset, fault):
    data = export(dataset)
    if fault == "revision":
        data["source"]["revision"] = data["fit_source"]["revision"] = "not-a-sha"
    elif fault == "source":
        data["source"]["dataset_id"] = data["fit_source"]["dataset_id"] = "wrong/source"
    elif fault == "split":
        data["source"]["evaluation_split"] = "validation"
    elif fault == "subset":
        data["source"]["selection"] = "subset"
    elif fault == "hash":
        data["evaluation"][0]["text"] = "altered text"
    else:
        data["train"][0], data["train"][1] = data["train"][1], data["train"][0]
    with pytest.raises(ValueError):
        validate_official_bundle(data)


def test_duplicate_training_rows_preserved(dataset):
    dataset["train"][77] = dict(dataset["train"][0])
    data = export(dataset)
    assert len(data["train"]) == 10003
    assert data["source"]["train_unique_texts"] == 10002
    assert task_plan("Banking77", bundle=data)["executable"]


def test_pca_deduplicates_but_classifier_gets_all_training_rows(monkeypatch):
    import numpy as np
    import pilot.task_runner as runner
    from pilot.task_config import PROTOCOLS
    rng = np.random.default_rng(3)
    train = [dict(id=f"t{i}", text=f"t{i}", label=str(i % 2)) for i in range(12)]
    train.append(dict(id="duplicate", text="t0", label="0"))
    evaluation = [dict(id=f"e{i}", text=f"e{i}", label=str(i)) for i in range(2)]
    source = dict(dataset_id="synthetic", revision="fixture-v1", configuration="fixture", evaluation_split="test",
                  selection="subset", text_format="plain")
    data = dict(schema_version=1, task="Banking77", source=source, train=train, evaluation=evaluation,
                fit_source=dict(dataset_id="synthetic", revision="fixture-v1", configuration="fixture", split="train",
                                role="train", split_method="independent synthetic test"))
    mapping = {r["text"]: rng.normal(size=8) for r in train + evaluation}
    fitted, classifier_rows = [], []
    original = runner.fit_pca
    def fit(x, *args):
        fitted.append(x)
        return original(x, *args)
    def classify(x, labels, eval_x, eval_labels, seed):
        classifier_rows.append(len(x))
        return {"accuracy": 0., "macro_f1": 0.}, ["0", "0"]
    monkeypatch.setattr(runner, "fit_pca", fit)
    monkeypatch.setattr(runner, "classification", classify)
    plan = dict(task="Banking77", model="synthetic", native_dimension=8, seed=42, dimensions=[8, 4, 2],
                pca_components=4, pca_train_sentences=12, protocol=PROTOCOLS["Banking77"])
    runner.evaluate_bundle(data, plan, lambda texts: np.asarray([mapping[t] for t in texts]))
    assert len(fitted) == 1 and len(np.unique(fitted[0], axis=0)) == 12
    assert classifier_rows == [13, 13, 13]


def test_preparation_dry_run_stdlib_only(tmp_path):
    script = Path(__file__).resolve().parents[1] / "prepare_banking77.py"
    output = tmp_path / "unused.json"
    proc = subprocess.run([sys.executable, "-B", "-S", str(script), "--dry-run", "--output", str(output)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["official_split_counts"] == SPLIT_COUNTS
    assert not output.exists()


def test_preparation_mocked_download_and_immutable_revision(dataset, tmp_path, monkeypatch, capsys):
    import prepare_banking77
    calls = []
    def info(identifier, revision):
        calls.append(("resolve", identifier, revision))
        return SimpleNamespace(sha="a" * 40)
    def configs(identifier, revision):
        assert revision == "a" * 40
        return ["fixture-config"]
    def load(identifier, name, revision):
        calls.append(("load", identifier, name, revision))
        return dataset
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=info)))
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(get_dataset_config_names=configs, load_dataset=load))
    output = tmp_path / "bundle.json"
    prepare_banking77.main(["--output", str(output)])
    data = json.loads(output.read_text(encoding="utf-8"))
    validate_official_bundle(data)
    assert calls[-1] == ("load", "PolyAI/banking77", "fixture-config", "a" * 40)
    assert task_plan("Banking77", bundle=data)["executable"]
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        prepare_banking77.main(["--output", str(output)])
    assert output.read_bytes() == before and len(calls) == 2


def test_full_local_bundle_dry_run(dataset, tmp_path, capsys):
    import run_experiment
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(export(dataset)), encoding="utf-8")
    run_experiment.main(["--task", "Banking77", "--data", str(path), "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert plan["executable"] and plan["data"]["fit_count"] == 10003
    assert plan["data"]["evaluation_count"] == 3080
