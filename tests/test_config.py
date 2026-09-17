"""Offline configuration and legacy-pilot wiring tests; no pretrained model execution."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from pilot.config import MODEL_SPECS, assignment_matrix, make_plan, validate_dimensions
from pilot.core import cached_embeddings, select_data

ROOT = Path(__file__).resolve().parents[1]


def test_required_matrix():
    assert assignment_matrix()["required_evaluation_configurations"] == 84
    assert MODEL_SPECS["Phi4-mini"]["dimensions"] == (3072, 1536, 768, 384)
    assert all(spec["model_id"] for spec in MODEL_SPECS.values())


def test_legacy_and_full_phi_plans():
    legacy = make_plan()
    full = make_plan(dimensions=[3072, 1536, 768, 384])
    assert legacy["dimensions"] == [3072, 768, 384]
    assert legacy["pca_components"] == 768
    assert full["pca_components"] == 1536 and full["executable"]
    assert make_plan(dimensions=[3072])["pca_components"] == 0


@pytest.mark.parametrize("dimensions", [[], [0], [-1], [3073], [768, 768], [True], [768.0]])
def test_invalid_dimensions(dimensions):
    with pytest.raises(ValueError):
        validate_dimensions(dimensions, 3072)


@pytest.mark.parametrize("kwargs", [
    {"dimensions": [3072, 1536], "train_count": 1536},
    {"dimensions": [1024]}, {"loading_strategy": "int4"},
    {"batch_size": 0}, {"eval_count": 1}, {"max_length": 1024},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


def test_all_assignment_models_have_real_checkpoint_plans():
    plan = make_plan(model="Qwen3-8B", task="SciFact", train_count=3000)
    assert plan["model_id"] == "Qwen/Qwen3-8B"
    assert plan["pca_components"] == 2048


def test_dry_run_without_site_packages_or_gpu(tmp_path):
    result = subprocess.run([sys.executable, "-S", str(ROOT / "run_pilot.py"), "--dry-run",
                             "--dimensions", "3072", "1536", "768", "384"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["pca_components"] == 1536
    assert not list(tmp_path.iterdir())


def test_dynamic_leakage_threshold_and_selection():
    data = {"train": [{"sentence1": f"train{i}", "sentence2": f"other{i}"} for i in range(800)],
            "validation": [{"sentence1": "train0", "sentence2": "v", "score": 1},
                           {"sentence1": "x", "sentence2": "y", "score": 2}],
            "test": [{"sentence1": "other0", "sentence2": "z", "score": 3}]}
    with pytest.raises(ValueError):
        select_data(data, 1536, 2, 42, min_train_count=1537)
    train, _, _ = select_data(data, 1537, 2, 42, min_train_count=1537)
    assert len(train) == 1537 and "train0" not in train and "other0" not in train


def test_cache_native_width_is_explicit(tmp_path):
    def encode(texts): return np.ones((len(texts), 16), dtype=np.float32)
    a, hit = cached_embeddings(tmp_path, {}, ["a"], encode, native_dimension=16)
    assert a.shape == (1, 16) and not hit
    _, hit = cached_embeddings(tmp_path, {}, ["a"], encode, native_dimension=16)
    assert hit
    with pytest.raises(ValueError): cached_embeddings(tmp_path, {}, ["a"], encode, native_dimension=32)


@pytest.mark.parametrize("dimensions,expected_components", [(None, 768), ([3072, 1536, 768, 384], 1536), ([3072], 0)])
def test_runner_wiring_keeps_training_and_evaluation_separate(monkeypatch, tmp_path, dimensions, expected_components):
    import run_pilot
    import pilot.core as core
    import pilot.model as model
    calls = {}; train_texts, eval_texts = ["training only"], ["left", "right"]
    pairs = [{"sentence1": "left", "sentence2": "right", "score": 1}]
    split = SimpleNamespace(column_names=["sentence1", "sentence2", "score"])
    dataset = {name: split for name in ("train", "validation", "test")}
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: dataset))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha="model-sha"), dataset_info=lambda *a, **k: SimpleNamespace(sha="data-sha"))))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda n: None, cuda=SimpleNamespace(manual_seed_all=lambda n: None, synchronize=lambda: None, max_memory_allocated=lambda: 0), backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)))))
    monkeypatch.setattr(run_pilot.importlib.metadata, "version", lambda name: "test")
    monkeypatch.setattr(model, "check_gpu", lambda: {"gpu": "offline-test-double"})
    monkeypatch.setattr(model, "PhiEncoder", lambda *a: SimpleNamespace(dtype="test", validate_batching=lambda: {}, encode=lambda texts: None, tokenizer=lambda texts, **k: {"input_ids": [[1] for _ in texts]}))
    def selection(*a, **k):
        assert k["min_train_count"] == expected_components + 1
        return train_texts, pairs, [0]
    monkeypatch.setattr(core, "select_data", selection)
    train_x, eval_x = np.ones((1, 3072)), np.ones((2, 3072)) * 2
    monkeypatch.setattr(core, "cached_embeddings", lambda directory, identity, texts, encode: ((train_x if texts == train_texts else eval_x), True))
    def fitting(array, max_dim, seed):
        assert array is train_x and max_dim == expected_components; calls["fit"] = True
        return SimpleNamespace(mean_=np.zeros(1), components_=np.zeros((1, 1)), explained_variance_ratio_=np.ones(1))
    monkeypatch.setattr(core, "fit_pca", fitting)
    def projection(array, pca, dimension):
        assert array is eval_x and dimension <= expected_components; calls.setdefault("projected", []).append(dimension); return np.ones((2, dimension))
    monkeypatch.setattr(core, "project", projection); monkeypatch.setattr(core, "score_pairs", lambda *a: (0.0, np.zeros(1)))
    def saving(output, rows, metadata): output.mkdir(); calls["rows"], calls["metadata"] = rows, metadata
    monkeypatch.setattr(run_pilot, "save_outputs", saving)
    argv = ["run_pilot.py", "--output", str(tmp_path / "result")]
    if dimensions is not None: argv += ["--dimensions", *map(str, dimensions)]
    monkeypatch.setattr(sys, "argv", argv); run_pilot.main()
    expected = [3072, 768, 384] if dimensions is None else dimensions
    assert [r["dimension"] for r in calls["rows"]] == expected
    assert calls["metadata"]["pca"]["components"] == expected_components
    assert bool(calls.get("fit")) == bool(expected_components)
    assert calls.get("projected", []) == [d for d in expected if d < 3072]
    assert (tmp_path / "result" / "pca.npz").exists() == bool(expected_components)


def test_four_dimension_export_and_plot_ticks(monkeypatch, tmp_path):
    import csv, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from run_pilot import save_outputs
    original = plt.subplots; axes = []
    def subplots(*args, **kwargs): fig, ax = original(*args, **kwargs); axes.append(ax); return fig, ax
    monkeypatch.setattr(plt, "subplots", subplots)
    rows = [{"dimension": d, "cosine_spearman": 0.0} for d in (3072, 1536, 768, 384)]
    save_outputs(tmp_path, rows, {"unit_test_only": True})
    assert list(axes[0].get_xticks()) == [384, 768, 1536, 3072]
    with (tmp_path / "scores.csv").open() as handle: saved = list(csv.DictReader(handle))
    assert [int(row["dimension"]) for row in saved] == [3072, 1536, 768, 384]


def test_legacy_native_cache_key_is_preserved(tmp_path):
    from pilot.core import digest
    identity, texts = {"revision": "original"}, ["a"]
    array = np.ones((1, 3072), dtype=np.float32); cached_embeddings(tmp_path, identity, texts, lambda _: array)
    assert (tmp_path / (digest({"identity": identity, "texts": texts}) + ".npz")).exists()
