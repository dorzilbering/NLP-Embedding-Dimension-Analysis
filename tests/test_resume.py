"""Deterministic interruption/recovery on tiny random CPU models only."""
import copy
import json
import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_training import tiny_backbone, spec, synthetic_official_bundle
from training.model import Banking77Model
from training.trainer import make_optimizer, seed_everything
from training.selection import fit_with_validation
from training.checkpoint import (atomic_write, capture_rng, latest_resume_checkpoint,
                                 restore_rng, run_lock, save_checkpoint)


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for first, second in zip(a, b):
            assert_equal(first, second)
    else:
        assert a == b


def setup_run(base, config):
    model = copy.deepcopy(base)
    def stochastic_loss(module, args, output):
        # Exercise Python, NumPy and torch RNG recovery, although production
        # frozen backbones deliberately disable dropout.
        if module.training:
            output["loss"] = output["loss"] * (0.5 + random.random() + float(np.random.rand()) + torch.rand(()))
    model.register_forward_hook(stochastic_loss)
    rows = [{"input_ids": torch.tensor([i + 1, i + 2, i + 3]),
             "attention_mask": torch.ones(3, dtype=torch.long), "labels": torch.tensor(i % 3)}
            for i in range(11)]
    train = DataLoader(rows, batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(99))
    validation = DataLoader(rows[:3], batch_size=2, generator=torch.Generator().manual_seed(42))
    optimizer = make_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    return model, train, validation, optimizer, scheduler


@pytest.mark.parametrize("crash", ["step1", "step4", "mid_window", "corrupt_latest"])
def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch, crash):
    import training.selection as selection
    seed_everything(42)
    base = Banking77Model(tiny_backbone("llama"), 8)
    config = spec(epochs=3, batch_size=2, accumulation_steps=2, checkpoint_every_steps=1)
    identity = {"run_id": "synthetic-owned-run", "data": {"bundle_hash": "fixture", "train_hash": "train"}}
    provenance = {"train_hash": "train", "validation_hash": "validation"}
    uninterrupted, interrupted = tmp_path / "full", tmp_path / "resumed"
    uninterrupted.mkdir()
    interrupted.mkdir()
    full = setup_run(base, config)
    seed_everything(42)
    expected_history, expected_selection = fit_with_validation(*full[:4], config, "cpu", uninterrupted,
                                                              provenance, full[4], identity=identity)
    expected_rng = capture_rng()
    broken = setup_run(base, config)
    real_save = selection.save_resume_checkpoint
    target = 4 if crash in ("step4", "corrupt_latest") else 1
    def fail_after_save(*args, **kwargs):
        result = real_save(*args, **kwargs)
        if args[6]["global_step"] == target and kwargs.get("phase", 0) == 0:
            raise RuntimeError("Simulated disconnect")
        return result
    if crash == "mid_window":
        calls = []
        def fail_mid_window(module, args, output):
            if module.training:
                calls.append(1)
                if len(calls) == 4:
                    raise RuntimeError("Simulated disconnect")
        broken[0].register_forward_hook(fail_mid_window)
    else:
        monkeypatch.setattr(selection, "save_resume_checkpoint", fail_after_save)
    seed_everything(42)
    with pytest.raises(RuntimeError, match="Simulated disconnect"):
        fit_with_validation(*broken[:4], config, "cpu", interrupted, provenance, broken[4], identity=identity)
    monkeypatch.setattr(selection, "save_resume_checkpoint", real_save)
    if crash == "corrupt_latest":
        newest = sorted(interrupted.glob("resume-*.pt"))[-1]
        newest.write_bytes(b"corrupt serialization")
        # Also imitate a crash before a temporary file/sidecar is committed.
        (interrupted / ".resume-incomplete.tmp").write_bytes(b"partial")
        (interrupted / "resume-999999999999-0-incomplete.pt").write_bytes(b"partial without checksum")
    payload = latest_resume_checkpoint(interrupted, config, identity)
    assert payload["progress"]["global_step"] == (3 if crash == "corrupt_latest" else target)
    original_checkpoints = {p.name: p.read_bytes() for p in interrupted.glob("*.pt")}
    seed_everything(999)  # A restarted process has unrelated RNG before restoration.
    resumed = setup_run(base, config)
    history, chosen = fit_with_validation(*resumed[:4], config, "cpu", interrupted, provenance, resumed[4],
                                         identity=identity, resume_payload=payload)
    assert_equal(history, expected_history)
    for key in ("best_epoch", "best_score", "validation", "metric", "tie_break"):
        assert chosen[key] == expected_selection[key]
    assert_equal(full[0].adaptation_state(), resumed[0].adaptation_state())
    assert_equal(full[3].state_dict(), resumed[3].state_dict())
    assert_equal(full[4].state_dict(), resumed[4].state_dict())
    assert_equal(expected_rng, capture_rng())
    final = latest_resume_checkpoint(interrupted, config, identity)
    assert final["progress"]["epoch"] == 4 and final["progress"]["global_step"] == 9
    assert final["scheduler"]["last_epoch"] == 9
    assert len([json.loads(line) for line in (interrupted / "training_metrics.jsonl").read_text().splitlines()]) == 3
    for name, content in original_checkpoints.items():
        assert (interrupted / name).read_bytes() == content


def test_resume_rejects_configuration_and_dataset_changes(tmp_path):
    seed_everything(42)
    base = Banking77Model(tiny_backbone("phi3"), 8)
    config = spec(epochs=1, checkpoint_every_steps=1)
    identity = {"run_id": "one", "bundle_hash": "original"}
    run = setup_run(base, config)
    fit_with_validation(*run[:4], config, "cpu", tmp_path, {}, run[4], identity=identity)
    with pytest.raises(ValueError, match="configuration mismatch"):
        latest_resume_checkpoint(tmp_path, spec(epochs=2, checkpoint_every_steps=1), identity)
    with pytest.raises(ValueError, match="dataset hashes mismatch"):
        latest_resume_checkpoint(tmp_path, config, {**identity, "bundle_hash": "changed"})
    with pytest.raises(ValueError, match="dataset hashes mismatch"):
        latest_resume_checkpoint(tmp_path, config, {**identity, "run_id": "other-run"})


def test_atomic_failure_never_publishes_partial_checkpoint_or_clobbers_file(tmp_path):
    path = tmp_path / "checkpoint.pt"
    def crash(handle):
        handle.write(b"partial")
        raise RuntimeError("write failed")
    with pytest.raises(RuntimeError, match="write failed"):
        atomic_write(path, crash)
    assert not path.exists() and not list(tmp_path.glob("*.tmp"))
    atomic_write(path, lambda handle: handle.write(b"original"))
    with pytest.raises(FileExistsError):
        atomic_write(path, lambda handle: handle.write(b"different run"))
    assert path.read_bytes() == b"original"


def test_rng_roundtrip_and_process_lock(tmp_path):
    seed_everything(42)
    state = capture_rng()
    expected = (random.random(), float(np.random.rand()), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), float(np.random.rand()), torch.rand(3))
    assert_equal(expected, actual)
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):
        pass


def test_old_adaptation_checkpoint_cannot_resume(tmp_path):
    model = Banking77Model(tiny_backbone("llama"), 8)
    save_checkpoint(tmp_path / "final.pt", model, spec())
    with pytest.raises(ValueError, match="No valid resumable"):
        latest_resume_checkpoint(tmp_path, spec(), {})


def test_cli_completed_resume_checks_data_and_does_not_load_model(tmp_path, monkeypatch):
    from training.run import main
    import training.model as models
    # Reuse the deterministic dry-run plan to build a completed synthetic run.
    from pilot.task_data import fingerprint
    from training.data import eligible_training_rows, split_training_rows
    bundle = synthetic_official_bundle()
    config = spec()
    eligible, audit = eligible_training_rows(bundle)
    _, _, split = split_training_rows(eligible)
    identity = {"run_id": "completed-fixture", "data": {
        "bundle_hash": fingerprint(bundle), "train_hash": split["train_hash"],
        "validation_hash": split["validation_hash"], "eligible_train_hash": audit["eligible_train_hash"],
        "protocol": "banking77-trainable-token-embedding-excluded-overlap-validation-v3",
        "data_order": "epoch_seeded_shuffle_v1"}}
    output = tmp_path / "run"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps({"config": config.to_dict(), "resume_identity": identity}))
    result = {"run_id": identity["run_id"], "scores": {"macro_f1": 0.5}}
    (output / "results.json").write_text(json.dumps(result))
    before = (output / "results.json").read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Completed resume attempted model loading/test scoring")
    monkeypatch.setattr(models, "load_model", forbidden)
    data = tmp_path / "data.json"
    data.write_text(json.dumps(bundle))
    command = ["--model", "Phi4-mini", "--revision", "a" * 40, "--dimension", "3072",
               "--data", str(data), "--output", str(output), "--resume"]
    main(command)
    assert (output / "results.json").read_bytes() == before
    with pytest.raises(ValueError, match="configuration mismatch"):
        main(command + ["--seed", "43"])
    bundle["train"][10]["text"] += " altered"
    bundle["source"]["split_hashes"]["train"] = fingerprint(bundle["train"])
    data.write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="dataset hashes mismatch"):
        main(command)


@pytest.mark.parametrize("stage", ["training", "finalization"])
def test_cli_resume_recovers_existing_run_without_overwriting_checkpoints(tmp_path, monkeypatch, stage):
    import training.model as models
    import training.selection as selection_module
    import training.trainer as trainer
    import training.checkpoint as checkpoints
    from training.run import main
    seed_everything(42)
    base = Banking77Model(tiny_backbone("phi3"), 8)
    class Tokenizer:
        pad_token_id, eos_token_id, padding_side = 0, 2, "right"
        def __len__(self):
            return 64
        def __call__(self, texts, **kwargs):
            ids = torch.tensor([[1, 2]] * len(texts))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    monkeypatch.setattr(models, "load_model", lambda config: (copy.deepcopy(base), Tokenizer()))
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0] == "cuda":
            args = ("cpu",) + args[1:]
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic-cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    actual_fit = selection_module.fit_with_validation
    calls = []
    def simulate_finished_training(model, train, validation, optimizer, config, device, output, provenance,
                                   scheduler, *, identity, resume_payload):
        if resume_payload is None:
            with torch.no_grad():
                model.embeddings.lookup.weight[1].add_(0.1)
            metrics = {"macro_f1": 0.8, "accuracy": 0.5, "loss": 1.0, "examples": len(validation.dataset)}
            checkpoints.save_checkpoint(output / "epoch-001.pt", model, config,
                                        {"epoch": 1, "validation": metrics, "provenance": provenance})
            progress = {"epoch": 2, "global_step": 1, "partial": None,
                        "history": [{"epoch": 1, "training": {"optimizer_steps": 1},
                                     "validation": metrics, "selected": True}],
                        "selector": {"best_epoch": 1, "best_score": 0.8, "last_epoch": 1},
                        "best_path": "epoch-001.pt"}
            checkpoints.save_resume_checkpoint(output, model, optimizer, scheduler, config, identity, progress, phase=1)
            if stage == "training":
                raise RuntimeError("Simulated disconnect")
            resume_payload = checkpoints.latest_resume_checkpoint(output, config, identity)
        return actual_fit(model, train, validation, optimizer, config, "cpu", output, provenance, scheduler,
                          identity=identity, resume_payload=resume_payload)
    monkeypatch.setattr(selection_module, "fit_with_validation", simulate_finished_training)
    original_save = checkpoints.save_checkpoint
    crashed = []
    def crash_after_final_save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        if stage == "finalization" and path.name == "final.pt" and not crashed:
            crashed.append(True)
            raise RuntimeError("Simulated disconnect")
    monkeypatch.setattr(checkpoints, "save_checkpoint", crash_after_final_save)
    def score_test(model, loader, device):
        assert all(row["id"].startswith("test:") for row in loader.dataset)
        calls.append("test")
        return {"accuracy": 0.5, "macro_f1": 0.5}, [0] * len(loader.dataset)
    monkeypatch.setattr(trainer, "evaluate", score_test)
    data, output = tmp_path / "data.json", tmp_path / "run"
    data.write_text(json.dumps(synthetic_official_bundle()))
    args = ["--model", "Phi4-mini", "--revision", "a" * 40, "--dimension", "768", "--epochs", "1",
            "--data", str(data), "--output", str(output)]
    with pytest.raises(RuntimeError, match="Simulated disconnect"):
        main(args)
    assert calls == []
    original_checkpoints = {path.name: path.read_bytes() for path in output.glob("*.pt")}
    main(args + ["--resume"])
    assert calls == ["test"]
    completed = json.loads((output / "results.json").read_text())
    assert completed["selection"]["best_epoch"] == 1
    assert completed["selection"]["test_access_during_selection"] is False
    main(args + ["--resume"])
    assert calls == ["test"]  # Completed runs never rescore official test.
    for name, content in original_checkpoints.items():
        assert (output / name).read_bytes() == content
