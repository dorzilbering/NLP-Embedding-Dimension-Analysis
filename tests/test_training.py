"""Tiny random CPU architectures only. Never download pretrained checkpoints."""
import copy
import json
import socket
import subprocess
import sys

import pytest
import torch
from transformers import AutoModel, LlamaConfig, Phi3Config

from training.checkpoint import load_checkpoint, save_checkpoint
from training.config import DIMENSIONS, MODEL_IDS, ExperimentConfig
from training.embeddings import TrainableTokenEmbedding
from training.model import Banking77Model, last_valid_pool, load_model
from training.trainer import evaluate, frozen_hash, make_optimizer, seed_everything, train_epoch
from training.data import split_training_rows
from training.selection import MacroF1Selector, fit_with_validation


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network is forbidden in training unit tests")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "1")


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_backbone(family):
    cls = Phi3Config if family == "phi3" else LlamaConfig
    kwargs = dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                  max_position_embeddings=128, pad_token_id=0, attention_dropout=0.0)
    if family == "phi3":
        kwargs["original_max_position_embeddings"] = 128
    config = cls(**kwargs)
    config._attn_implementation = "eager"
    return AutoModel.from_config(config)


def batch():
    ids = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    return {"input_ids": ids, "attention_mask": ids.ne(0).long(), "labels": torch.tensor([1, 2])}


def spec(**kwargs):
    return ExperimentConfig("Phi4-mini", "a" * 40, 3072, **kwargs)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_required_lookup_dimensions(dimension):
    module = TrainableTokenEmbedding(8, dimension, 3072)
    assert module.lookup.weight.shape == (8, dimension)
    assert module.embedding_dim == dimension
    assert module(torch.tensor([[1, 2]])).shape == (1, 2, 3072)
    if dimension == 3072:
        assert isinstance(module.projection, torch.nn.Identity)
    else:
        assert module.projection.weight.shape == (3072, dimension)


def test_native_copy_and_svd_initialization():
    seed_everything(42)
    original = torch.nn.Embedding(24, 8, padding_idx=0)
    native = TrainableTokenEmbedding.from_pretrained(original, 8, chunk_size=5)
    torch.testing.assert_close(native.lookup.weight, original.weight, rtol=0, atol=0)
    reduced = TrainableTokenEmbedding.from_pretrained(original, 4, chunk_size=5)
    reconstructed = reduced.lookup.weight @ reduced.projection.weight.T
    u, s, vh = torch.linalg.svd(original.weight.detach().double(), full_matrices=False)
    optimal = (u[:, :4] * s[:4]) @ vh[:4]
    torch.testing.assert_close(reconstructed.double(), optimal, rtol=1e-5, atol=1e-6)
    expected = (original.weight.detach() - reconstructed).norm() / original.weight.detach().norm()
    assert reduced.initialization["relative_error"] == pytest.approx(expected.item(), rel=1e-5)
    with pytest.raises(ValueError):
        TrainableTokenEmbedding.from_pretrained(original, 9)


@pytest.mark.parametrize("family", ["phi3", "llama"])
@pytest.mark.parametrize("dimension", [32, 16, 8])
def test_internal_replacement_gradient_update_freeze_and_logits(family, dimension):
    seed_everything(42)
    backbone = tiny_backbone(family)
    original = backbone.get_input_embeddings()
    model = Banking77Model(backbone, dimension)
    assert backbone.get_input_embeddings() is model.embeddings
    assert model.embeddings.lookup.weight.shape == (64, dimension)
    assert model.embeddings is not original
    frozen_before = frozen_hash(model)
    before = model.embeddings.lookup.weight.detach().clone()
    unused_before = before[40].clone()
    output = model(**batch())
    assert output["logits"].shape == (2, 77)
    assert output["loss"].ndim == 0
    optimizer = make_optimizer(model, spec())
    output["loss"].backward()
    grad = model.embeddings.lookup.weight.grad
    assert torch.isfinite(grad).all() and grad[1:8].norm() > 0
    assert grad[0].abs().sum() == 0
    if dimension < 32:
        assert model.embeddings.projection.weight.grad.norm() > 0
    for name, p in model.backbone.named_parameters():
        if not name.startswith("embed_tokens."):
            assert not p.requires_grad and p.grad is None
    optimizer.step()
    assert not torch.equal(before, model.embeddings.lookup.weight)
    torch.testing.assert_close(model.embeddings.lookup.weight[40], unused_before, rtol=0, atol=0)
    assert frozen_hash(model) == frozen_before
    # HF's direct input_ids path really sees the replaced internal component.
    assert backbone(input_ids=batch()["input_ids"], attention_mask=batch()["attention_mask"],
                    use_cache=False).last_hidden_state.shape == (2, 4, 32)


@pytest.mark.parametrize("family", ["phi3", "llama"])
@pytest.mark.parametrize("dimension", [32, 16, 8])
def test_fresh_model_checkpoint_roundtrip(family, dimension, tmp_path):
    seed_everything(43)
    backbone = tiny_backbone(family)
    fresh_backbone = copy.deepcopy(backbone)
    model = Banking77Model(backbone, dimension)
    config = spec(epochs=1)
    metrics = train_epoch(model, [batch()], make_optimizer(model, config), "cpu")
    assert metrics["update_evidence"]["sample_delta_norm"] > 0
    model.eval()
    expected = model(**batch())["logits"].detach()
    path = tmp_path / "adaptation.pt"
    save_checkpoint(path, model, config, {"fixture": "synthetic"})
    fresh = Banking77Model(fresh_backbone, dimension).eval()
    assert load_checkpoint(path, fresh, config) == {"fixture": "synthetic"}
    torch.testing.assert_close(fresh(**batch())["logits"], expected, rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model, config)
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_checkpoint(path, fresh, spec(seed=44, epochs=1))


@pytest.mark.parametrize("family", ["phi3", "llama"])
def test_gradient_checkpointing_keeps_embedding_gradient(family):
    model = Banking77Model(tiny_backbone(family), 8, gradient_checkpointing=True)
    model(**batch())["loss"].backward()
    assert model.embeddings.lookup.weight.grad.norm() > 0
    assert model.backbone.is_gradient_checkpointing
    assert all(not module.training for module in model.backbone.modules()
               if isinstance(module, torch.nn.Dropout))


def test_accumulation_matches_combined_examples_including_short_window():
    seed_everything(7)
    model = Banking77Model(tiny_backbone("llama"), 8)
    other = copy.deepcopy(model)
    data = batch()
    singles = [{key: value[i:i + 1] for key, value in data.items()} for i in range(2)]
    # Three single-example microbatches yield a full window then a short window.
    history = train_epoch(model, singles + singles[:1], make_optimizer(model, spec()), "cpu", 2)
    train_epoch(other, [data, singles[0]], make_optimizer(other, spec()), "cpu", 1)
    assert history["examples"] == 3 and history["optimizer_steps"] == 2
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-6)
    scores, predictions = evaluate(model, [data], "cpu")
    assert len(predictions) == 2 and scores["examples"] == 2
    assert 0 <= scores["macro_f1"] <= 1


def test_padding_pooling_and_training_batch_invariance():
    model = Banking77Model(tiny_backbone("llama"), 8).eval()
    expected = model(**batch())["logits"]
    left_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    left = model(left_ids, left_ids.ne(0).long())["logits"]
    torch.testing.assert_close(expected, left, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="all-padding"):
        last_valid_pool(torch.zeros(1, 2, 3), torch.zeros(1, 2))


@pytest.mark.parametrize("family", ["phi3", "llama"])
def test_bf16_backbone_keeps_float32_trainables_and_gradient(family):
    backbone = tiny_backbone(family).to(dtype=torch.bfloat16)
    model = Banking77Model(backbone, 8)
    assert model.embeddings.lookup.weight.dtype == torch.float32
    assert model.embeddings.projection.weight.dtype == torch.float32
    assert model.classifier.weight.dtype == torch.float32
    model(**batch())["loss"].backward()
    assert model.embeddings.lookup.weight.grad.norm() > 0
    assert backbone(input_ids=batch()["input_ids"], attention_mask=batch()["attention_mask"],
                    use_cache=False).last_hidden_state.dtype == torch.bfloat16


@pytest.mark.parametrize("model", MODEL_IDS)
@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_six_configs(model, dimension):
    config = ExperimentConfig(model, "b" * 40, dimension)
    assert config.model_id == MODEL_IDS[model]
    assert ExperimentConfig(**config.to_dict()) == config


@pytest.mark.parametrize("kwargs", [{"revision": "main"}, {"dimension": 384}, {"epochs": 0},
                                     {"seed": -1}, {"embedding_lr": float("nan")},
                                     {"validation_fraction": 0}, {"validation_fraction": 1},
                                     {"validation_seed": -1}])
def test_invalid_config(kwargs):
    values = spec().to_dict()
    values.update(kwargs)
    with pytest.raises(ValueError):
        ExperimentConfig(**values)


def test_full_checkpoint_cpu_loading_is_rejected():
    with pytest.raises(RuntimeError, match="CPU is for tiny"):
        load_model(spec(), device="cpu")


@pytest.mark.parametrize("overlap", [False, True])
def test_dry_run_official_validation_without_scientific_imports(tmp_path, overlap):
    data = synthetic_official_bundle(overlap)
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))
    output = tmp_path / "absent"
    command = [sys.executable, "-B", "-S", "-m", "training.run", "--model", "Phi4-mini",
               "--revision", "a" * 40, "--dimension", "768", "--data", str(path),
               "--output", str(output), "--dry-run"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["token_embedding_dimension"] == 768
    assert plan["validation_split"]["source_split"] == "train"
    assert plan["training_exclusions"]["excluded_train_row_count"] == (8 if overlap else 0)
    assert plan["training_exclusions"]["overlap_unique_text_count"] == (7 if overlap else 0)
    assert not set(plan["validation_split"]["train_ids"]) & set(plan["validation_split"]["validation_ids"])
    assert not output.exists()
    data["source"]["split_hashes"]["test"] = "corrupted"
    path.write_text(json.dumps(data))
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and "hash mismatch" in result.stderr


def synthetic_official_bundle(overlap=False):
    from pilot.banking77 import build_bundle
    from types import SimpleNamespace
    class Partition(list):
        column_names = ["text", "label"]
        features = {"label": SimpleNamespace(names=[f"intent_{i}" for i in range(77)])}
    dataset = {split: Partition({"text": f"synthetic {split} row {i}", "label": i % 77}
                               for i in range(count))
               for split, count in (("train", 10003), ("test", 3080))}
    if overlap:
        for i in range(7):
            dataset["test"][i]["text"] = "  " + dataset["train"][i]["text"].upper() + "  "
        dataset["train"][100] = dict(dataset["train"][0])
    return build_bundle(dataset, "c" * 40, "synthetic")


def grouped_train_rows():
    rows = []
    for label in range(77):
        for index in range(10):
            # First two rows have the same canonical text but remain distinct rows.
            text = f"class {label} example {max(1, index)}"
            if index == 0:
                text = "  " + text.upper() + "  "
            rows.append({"id": f"train:{len(rows)}", "text": text, "label": str(label)})
    return rows


def test_validation_split_deterministic_stratified_grouped_and_complete():
    from pilot.task_data import text_key, fingerprint
    rows = grouped_train_rows()
    original = copy.deepcopy(rows)
    train, validation, metadata = split_training_rows(rows, fraction=0.2, seed=42)
    assert (train, validation, metadata) == split_training_rows(rows, fraction=0.2, seed=42)
    assert rows == original
    train_ids = {r["id"] for r in train}
    validation_ids = {r["id"] for r in validation}
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == {r["id"] for r in rows}
    assert len(train) + len(validation) == len(rows)
    assert {text_key(r["text"]) for r in train}.isdisjoint({text_key(r["text"]) for r in validation})
    for label in map(str, range(77)):
        assert sum(r["label"] == label for r in train) == 8
        assert sum(r["label"] == label for r in validation) == 2
    assert metadata["train_hash"] == fingerprint(train)
    assert metadata["validation_hash"] == fingerprint(validation)
    # Membership is independent of row order, and both partitions preserve their input order.
    reversed_train, reversed_val, _ = split_training_rows(list(reversed(rows)), 0.2, 42)
    assert reversed_train == list(reversed(train)) and reversed_val == list(reversed(validation))
    _, different, _ = split_training_rows(rows, 0.2, 43)
    assert {r["id"] for r in different} != validation_ids


def test_split_seed_is_separate_from_training_seed_and_test_contents():
    rows = grouped_train_rows()
    configs = [spec(seed=seed) for seed in (42, 43, 44)]
    splits = [split_training_rows(rows, c.validation_fraction, c.validation_seed)[2] for c in configs]
    assert splits[0] == splits[1] == splits[2]
    class ForbiddenTest:
        def __iter__(self):
            raise AssertionError("Official test data accessed by validation splitting")
    bundle = {"train": rows, "evaluation": ForbiddenTest()}
    assert split_training_rows(bundle["train"])[2] == splits[0]


@pytest.mark.parametrize("fault", ["test_id", "duplicate_id", "conflict", "one_group", "missing_class"])
def test_validation_split_rejects_unisolatable_or_invalid_inputs(fault):
    rows = grouped_train_rows()
    if fault == "test_id":
        rows[0]["id"] = "test:0"
    elif fault == "duplicate_id":
        rows[0]["id"] = rows[1]["id"]
    elif fault == "conflict":
        rows[0]["label"] = "1"
    elif fault == "one_group":
        for row in rows[:10]:
            row["text"] = "one text for class zero"
    else:
        rows = rows[10:]
    with pytest.raises(ValueError):
        split_training_rows(rows)


def test_split_reports_approximate_counts_when_groups_prevent_exact_target():
    rows = grouped_train_rows()
    for index, row in enumerate(rows[:10]):
        row["text"] = "class zero big group" if index < 7 else "class zero small group"
    train, validation, metadata = split_training_rows(rows, 0.2, 42)
    assert metadata["class_counts"]["0"] == {"train": 7, "validation": 3, "target_validation": 2}


def test_macro_f1_selection_ignores_accuracy_loss_and_keeps_earliest_tie():
    selector = MacroF1Selector()
    assert selector.consider(1, {"macro_f1": 0.4, "accuracy": 0.9, "loss": 0.1})
    assert selector.consider(2, {"macro_f1": 0.6, "accuracy": 0.1, "loss": 9.0})
    assert not selector.consider(3, {"macro_f1": 0.6, "accuracy": 1.0, "loss": 0.0})
    assert not selector.consider(4, {"macro_f1": 0.5, "accuracy": 1.0, "loss": 0.0})
    assert selector.best_epoch == 2 and selector.best_score == 0.6
    with pytest.raises(ValueError, match="increasing"):
        selector.consider(4, {"macro_f1": 0.7})


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_selection_rejects_invalid_macro_f1(score):
    selector = MacroF1Selector()
    with pytest.raises(ValueError, match="finite"):
        selector.consider(1, {"macro_f1": score})
    assert selector.best_epoch is None


@pytest.mark.parametrize("family", ["phi3", "llama"])
def test_selection_restores_best_weights_and_never_sees_test(family, tmp_path, monkeypatch):
    import training.selection as selection_module
    seed_everything(42)
    model = Banking77Model(tiny_backbone(family), 8)
    config = spec(epochs=3, accumulation_steps=1)
    train_data = [batch()]
    validation_batch = batch()
    validation_batch["input_ids"] = torch.tensor([[11, 12, 13, 0], [14, 15, 16, 17]])
    validation_data = [validation_batch]
    frozen_before = frozen_hash(model)
    val_rows_before = model.embeddings.lookup.weight.detach()[11:18].clone()
    states, eval_calls = [], []
    actual_train, actual_evaluate = selection_module.train_epoch, selection_module.evaluate
    def record_training(*args, **kwargs):
        assert args[1] is train_data
        result = actual_train(*args, **kwargs)
        states.append(copy.deepcopy(model.adaptation_state()))
        return result
    def scripted_validation(*args, **kwargs):
        assert args[1] is validation_data
        assert not torch.equal(validation_batch["input_ids"], batch()["input_ids"])
        before = copy.deepcopy(model.adaptation_state())
        result, predictions = actual_evaluate(*args, **kwargs)
        for component in before:
            for name, value in before[component].items():
                torch.testing.assert_close(model.adaptation_state()[component][name], value, rtol=0, atol=0)
        result["macro_f1"] = [0.5, 0.8, 0.6][len(eval_calls)]
        eval_calls.append(result)
        return result, predictions
    monkeypatch.setattr(selection_module, "train_epoch", record_training)
    monkeypatch.setattr(selection_module, "evaluate", scripted_validation)
    history, selected = fit_with_validation(model, train_data, validation_data,
                                           make_optimizer(model, config), config, "cpu", tmp_path,
                                           {"train_hash": "synthetic-train", "validation_hash": "synthetic-val"})
    assert len(history) == len(eval_calls) == 3
    assert selected["best_epoch"] == 2 and selected["best_score"] == 0.8
    assert selected["checkpoint"] == "epoch-002.pt"
    assert not (tmp_path / "epoch-003.pt").exists()
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["epoch-001.pt", "epoch-002.pt"]
    assert selected["test_access_during_selection"] is False
    for component in states[1]:
        for name, value in states[1][component].items():
            torch.testing.assert_close(model.adaptation_state()[component][name], value, rtol=0, atol=0)
    assert not torch.equal(states[1]["classifier"]["weight"], states[2]["classifier"]["weight"])
    torch.testing.assert_close(model.embeddings.lookup.weight[11:18], val_rows_before, rtol=0, atol=0)
    assert frozen_hash(model) == frozen_before


def test_validation_evaluation_creates_no_parameter_gradients():
    model = Banking77Model(tiny_backbone("llama"), 8)
    model.zero_grad(set_to_none=True)
    evaluate(model, [batch()], "cpu")
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("overlap", [False, True])
def test_runner_scores_test_only_after_finalized_selection(tmp_path, monkeypatch, overlap):
    """Exercise CLI orchestration with injected tiny CPU weights and no training."""
    import training.model as model_module
    import training.selection as selection_module
    import training.trainer as trainer_module
    from training.run import main
    model = Banking77Model(tiny_backbone("phi3"), 8)
    class Tokenizer:
        pad_token_id, eos_token_id, padding_side = 0, 2, "right"
        def __len__(self):
            return 64
        def __call__(self, texts, **kwargs):
            tokens = torch.tensor([[1, 2]] * len(texts))
            return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}
    monkeypatch.setattr(model_module, "load_model", lambda config: (model, Tokenizer()))
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == "cuda":
            args = ("cpu",) + args[1:]
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic-cpu-fixture")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    output = tmp_path / "run"
    selected = {"best_epoch": 2, "best_score": 0.8, "metric": "validation_macro_f1"}
    events = []
    def fake_fit(model, train, validation, optimizer, config, device, output, provenance, scheduler):
        assert {row["id"] for row in train.dataset}.isdisjoint({row["id"] for row in validation.dataset})
        assert len(train.dataset) + len(validation.dataset) == (9995 if overlap else 10003)
        assert all(row["id"].startswith("train:") for row in train.dataset + validation.dataset)
        excluded_ids = set(provenance["training_exclusions"]["excluded_train_ids"])
        assert excluded_ids.isdisjoint({row["id"] for row in train.dataset + validation.dataset})
        assert "test" not in provenance
        with torch.no_grad():
            model.embeddings.lookup.weight[1].add_(0.1)
        model.eval()
        events.append("selection-finalized")
        return [], selected
    def fake_test(model, loader, device):
        assert events == ["selection-finalized"]
        assert json.loads((output / "selection.json").read_text()) == selected
        assert (output / "final.pt").exists()
        assert len(loader.dataset) == 3080
        assert all(row["id"].startswith("test:") for row in loader.dataset)
        events.append("test-scored")
        return {"macro_f1": 0.5, "accuracy": 0.5}, [0] * 3080
    monkeypatch.setattr(selection_module, "fit_with_validation", fake_fit)
    monkeypatch.setattr(trainer_module, "evaluate", fake_test)
    data = tmp_path / "data.json"
    raw = synthetic_official_bundle(overlap)
    data.write_text(json.dumps(raw))
    original_bytes = data.read_bytes()
    main(["--model", "Phi4-mini", "--revision", "a" * 40, "--dimension", "768",
          "--data", str(data), "--output", str(output)])
    assert events == ["selection-finalized", "test-scored"]
    assert json.loads((output / "results.json").read_text())["selection"] == selected
    assert data.read_bytes() == original_bytes
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["training_exclusions"]["excluded_train_row_count"] == (8 if overlap else 0)
    assert manifest["test_ids"] == [row["id"] for row in raw["evaluation"]]
