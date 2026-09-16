"""Entirely synthetic task tests: no datasets, checkpoints or network access."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import pytest

from pilot.task_config import PROTOCOLS, task_plan
from pilot.task_data import load_bundle, validate_bundle
from pilot.evaluation import classification, clustering, retrieval
from pilot.task_runner import evaluate_bundle, save_task_outputs, sts_metrics


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in task tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")


def bundle(task="Banking77"):
    source = dict(dataset_id="synthetic-fixture", revision="fixture-sha256-v1", configuration="tiny",
                  evaluation_split="test", selection="subset", text_format="plain")
    fit_source = dict(dataset_id=source["dataset_id"], revision=source["revision"], configuration="tiny",
                      split="train", role="train", split_method="synthetic independent fixtures")
    train = [dict(id=f"t{i}", text=f"training text {i}", label=str(i % 2)) for i in range(20)]
    evaluation = [dict(id=f"e{i}", text=f"evaluation text {i}", label=str(i % 2)) for i in range(4)]
    data = dict(schema_version=1, task=task, source=source, fit_source=fit_source,
                train=train, evaluation=evaluation)
    if task == "Arxiv-Clustering":
        source.update(variant="S2S", subset_id="synthetic-subset")
    elif task == "SciFact":
        source["text_format"] = "title_newline_abstract"
        data.update(reference=data.pop("train"), queries=data.pop("evaluation"),
                    corpus=[dict(id=f"d{i}", text=f"title {i}\nabstract {i}") for i in range(5)],
                    qrels={f"e{i}": {f"d{i}": 1} for i in range(4)})
    return data


@pytest.mark.parametrize("task", ["SciFact", "Banking77", "Arxiv-Clustering"])
def test_bundle_and_native_plan(task):
    data = bundle(task)
    if task in ("Banking77", "SciFact"):
        with pytest.raises(ValueError, match="full official"):
            task_plan(task, dimensions=[3072], bundle=data)
        return
    plan = task_plan(task, dimensions=[3072], bundle=data, clusters=2 if task == "Arxiv-Clustering" else None)
    assert plan["executable"]
    assert plan["data"]["evaluation_count"] == 4


@pytest.mark.parametrize("task", ["SciFact", "Banking77", "Arxiv-Clustering"])
def test_missing_data_blocks_and_pca_count(task):
    assert not task_plan(task)["executable"]
    with pytest.raises(ValueError, match="full official" if task in ("Banking77", "SciFact") else "Not enough"):
        task_plan(task, bundle=bundle(task), clusters=2 if task == "Arxiv-Clustering" else None)


@pytest.mark.parametrize("task,target", [("Banking77", "evaluation"), ("Arxiv-Clustering", "evaluation"),
                                         ("SciFact", "queries"), ("SciFact", "corpus")])
def test_canonical_overlap_rejected(task, target):
    data = bundle(task)
    fit = data["reference"] if task == "SciFact" else data["train"]
    fit[0]["text"] = "  ＨＥＬＬＯ  World "
    data[target][0]["text"] = "hello world"
    with pytest.raises(ValueError, match="Leakage"):
        validate_bundle(data, task)


@pytest.mark.parametrize("split", ["test", "validation", "heldout", "evaluation"])
def test_fit_cannot_claim_eval_split(split):
    data = bundle()
    data["fit_source"]["split"] = split
    with pytest.raises(ValueError):
        validate_bundle(data, "Banking77")


@pytest.mark.parametrize("fault", ["duplicate_text", "duplicate_id", "unseen_label", "revision", "full", "same_id"])
def test_invalid_classification_bundle(fault):
    data = bundle()
    if fault == "duplicate_text":
        data["train"][1]["text"] = data["train"][0]["text"]
    elif fault == "duplicate_id":
        data["train"][1]["id"] = data["train"][0]["id"]
    elif fault == "unseen_label":
        data["evaluation"][0]["label"] = "unknown"
    elif fault == "revision":
        data["fit_source"]["revision"] = "different"
    elif fault == "full":
        data["source"]["selection"] = "full"
    else:
        data["evaluation"][0]["id"] = data["train"][0]["id"]
    with pytest.raises(ValueError):
        validate_bundle(data, "Banking77")


@pytest.mark.parametrize("qrels", [{}, {"bad": 1}, {"d0": -1}, {"d0": 0}, {"d0": float("nan")}, {"d0": True}])
def test_bad_qrels(qrels):
    data = bundle("SciFact")
    data["qrels"]["e0"] = qrels
    with pytest.raises(ValueError):
        validate_bundle(data, "SciFact")


def test_missing_qrels_and_variant():
    data = bundle("SciFact")
    del data["qrels"]["e0"]
    with pytest.raises(ValueError):
        validate_bundle(data, "SciFact")
    data = bundle("Arxiv-Clustering")
    del data["source"]["variant"]
    with pytest.raises(ValueError):
        validate_bundle(data, "Arxiv-Clustering")
    assert not task_plan("Arxiv-Clustering", dimensions=[3072], bundle=bundle("Arxiv-Clustering"))["executable"]


def test_duplicate_json_keys(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text('{"task":"SciFact", "task":"Banking77"}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        load_bundle(path, "Banking77")


def test_classification_train_only_and_deterministic(monkeypatch):
    from sklearn.linear_model import LogisticRegression
    original = LogisticRegression.fit
    observed = []
    def spy(self, x, y):
        observed.append((x.copy(), list(y), self.get_params()))
        return original(self, x, y)
    monkeypatch.setattr(LogisticRegression, "fit", spy)
    train = np.array([[1, .1], [1, -.1], [-1, .1], [-1, -.1]])
    eval_x = np.array([[1, 0], [-1, 0]])
    scores, predictions = classification(train, ["a", "a", "b", "b"], eval_x, ["a", "b"], 42)
    assert scores == {"accuracy": 1., "macro_f1": 1.}
    changed, pred2 = classification(train, ["a", "a", "b", "b"], eval_x, ["b", "a"], 42)
    assert predictions == pred2 and changed["accuracy"] == 0
    assert all(len(x) == 4 for x, _, _ in observed)
    assert all(p["C"] == 1. and p["solver"] == "lbfgs" for _, _, p in observed)


def test_clustering_fit_reference_only(monkeypatch):
    from sklearn.cluster import MiniBatchKMeans
    original = MiniBatchKMeans.fit
    fitted = []
    def spy(self, x, *args, **kwargs):
        fitted.append(x.copy())
        return original(self, x, *args, **kwargs)
    monkeypatch.setattr(MiniBatchKMeans, "fit", spy)
    train = np.array([[1, .01], [1, -.01], [-1, .01], [-1, -.01]])
    test = np.array([[1, 0], [-1, 0]])
    scores, pred = clustering(train, test, ["a", "b"], 2, 42)
    scores2, pred2 = clustering(train, test, ["renamed-a", "renamed-b"], 2, 42)
    assert scores == scores2 == {"v_measure": 1., "adjusted_rand": 1., "nmi": 1.}
    assert pred == pred2 and all(x.shape == (4, 2) for x in fitted)
    with pytest.raises(ValueError):
        clustering(train, test, ["a", "b"], 5, 42)


def test_retrieval_known_ranking_and_linear_ndcg():
    scores, rankings = retrieval([[1, 0]], ["q"], [[.8, .6], [1, 0], [0, 1]],
                                 ["relevant", "irrelevant", "third"], {"q": {"relevant": 2, "third": 1}})
    expected = (2 / np.log2(3) + 1 / np.log2(4)) / (2 + 1 / np.log2(3))
    assert scores["ndcg_at_10"] == pytest.approx(expected)
    assert scores["mrr_at_10"] == .5
    assert scores["recall_at_10"] == scores["recall_at_100"] == 1
    assert rankings["q"][0]["doc_id"] == "irrelevant"


def test_retrieval_ties_cutoffs_and_no_positive():
    ids = [f"d{i:03}" for i in range(105)]
    scores, rankings = retrieval([[1, 0]], ["q"], [[1, 0]] * 105, ids[::-1], {"q": {"d104": 1}})
    assert set(scores.values()) == {0.}
    assert len(rankings["q"]) == 100 and rankings["q"][0]["doc_id"] == "d000"
    with pytest.raises(ValueError):
        retrieval([[1, 0]], ["q"], [[1, 0]], ["d"], {"q": {"d": 0}})


@pytest.mark.parametrize("task", ["SciFact", "Banking77", "Arxiv-Clustering"])
def test_complete_synthetic_pipeline_train_only_pca_and_exports(task, tmp_path, monkeypatch):
    import pilot.task_runner as runner
    data = bundle(task)
    # Internal engine is model-agnostic; these toy widths never pass the production Phi CLI.
    plan = dict(task=task, model="synthetic", native_dimension=12, seed=42, dimensions=[12, 4, 2],
                pca_components=4, pca_train_sentences=10, clusters=2, protocol=PROTOCOLS[task])
    rng = np.random.default_rng(123)
    groups = [data[k] for k in ("reference", "queries", "corpus") if k in data] or [data["train"], data["evaluation"]]
    mapping = {r["text"]: rng.normal(size=12).astype(np.float32) for group in groups for r in group}
    calls, fitted = [], []
    def encode(texts):
        calls.append(texts)
        return np.asarray([mapping[t] for t in texts])
    original = runner.fit_pca
    def fit(x, *args):
        fitted.append(x.copy())
        return original(x, *args)
    monkeypatch.setattr(runner, "fit_pca", fit)
    rows, pred, metadata, pca = evaluate_bundle(data, plan, encode)
    assert len(calls) == len(groups) and len(fitted) == 1
    selected = range(20) if task == "SciFact" else np.random.default_rng(42).permutation(20)[:10]
    np.testing.assert_array_equal(fitted[0], np.asarray([mapping[groups[0][i]["text"]] for i in selected]))
    assert pca.components_.shape == (4, 12)
    assert len(rows) == 3 * len(PROTOCOLS[task]["metrics"])
    assert set(pred["by_dimension"]) == {"12", "4", "2"}
    assert not set(metadata["pca_fit_ids"]) & set(metadata["evaluation_ids"])
    output = tmp_path / "output"
    save_task_outputs(output, rows, pred, metadata, pca)
    assert {p.name for p in output.iterdir()} == {"metrics.csv", "results.json", "predictions.json", "pca.npz"}
    assert json.loads((output / "results.json").read_text())["scores"] == rows
    with pytest.raises(ValueError, match="not empty"):
        save_task_outputs(output, rows, pred, metadata, pca)


@pytest.mark.parametrize("invalid", [np.zeros((20, 11)), np.full((20, 12), np.nan)])
def test_engine_rejects_invalid_embeddings(invalid):
    plan = dict(task="Banking77", native_dimension=12)
    plan["seed"] = 42
    with pytest.raises(ValueError):
        evaluate_bundle(bundle(), plan, lambda texts: invalid)


@pytest.mark.parametrize("task", ["STSB", "SciFact", "Banking77", "Arxiv-Clustering"])
def test_stdlib_only_dry_run(task, tmp_path):
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, "-B", "-S", str(root / "run_experiment.py"), "--task", task,
                           "--dry-run", "--output", str(tmp_path / "unused")],
                          cwd=root, capture_output=True, text=True, env=os.environ.copy())
    assert proc.returncode == 0, proc.stderr
    plan = json.loads(proc.stdout)
    assert plan["dimensions"] == [3072, 1536, 768, 384]
    assert plan["executable"] == (task == "STSB")
    assert not (tmp_path / "unused").exists()


def test_blocked_execution_never_imports_model(monkeypatch):
    import run_experiment
    monkeypatch.setitem(sys.modules, "pilot.model", None)
    with pytest.raises(SystemExit):
        run_experiment.main(["--task", "SciFact"])


def test_sts_delegation_and_common_export(monkeypatch, tmp_path):
    import run_experiment
    import run_pilot
    output = tmp_path / "sts"
    called = []
    result = {"metadata": {"dataset_revision": "synthetic-revision"}, "scores": [
        dict(dataset="synthetic", split="validation", pairs=3, dimension=3072,
             reduction="native", metric="cosine_spearman", cosine_spearman=.5, seed=42)]}
    def fake():
        called.append(list(sys.argv))
        output.mkdir()
        (output / "results.json").write_text(json.dumps(result))
    monkeypatch.setattr(run_pilot, "main", fake)
    original = list(sys.argv)
    run_experiment.main(["--task", "STSB", "--output", str(output)])
    assert sys.argv == original
    assert called[0][1:6] == ["--dimensions", "3072", "1536", "768", "384"]
    assert json.loads((output / "results.json").read_text()) == result
    assert (output / "metrics.csv").exists()
    assert sts_metrics(result)[0]["score"] == .5


def test_new_model_stays_blocked():
    plan = task_plan("Banking77", model="Qwen3-8B", train_count=3000)
    assert not plan["executable"] and not plan["implemented"]


@pytest.mark.parametrize("revision", ["main", "latest", "unknown"])
def test_unresolved_revision_rejected(revision):
    data = bundle()
    data["source"]["revision"] = revision
    with pytest.raises(ValueError, match="immutable"):
        validate_bundle(data, "Banking77")


@pytest.mark.parametrize("task", ["SciFact", "Banking77", "Arxiv-Clustering"])
def test_dry_run_with_local_bundle(task, tmp_path, capsys):
    import run_experiment
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(bundle(task)), encoding="utf-8")
    argv = ["--task", task, "--data", str(path), "--dimensions", "3072", "--dry-run"]
    if task == "Arxiv-Clustering":
        argv.extend(["--clusters", "2"])
    if task in ("Banking77", "SciFact"):
        with pytest.raises(ValueError, match="full official"):
            run_experiment.main(argv)
        return
    run_experiment.main(argv)
    assert json.loads(capsys.readouterr().out)["executable"]
