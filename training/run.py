"""Separate Banking77 training CLI. One model/dimension/seed per invocation."""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

from pilot.banking77 import validate_official_bundle
from pilot.task_data import fingerprint, load_bundle
from .config import DIMENSIONS, MODEL_IDS, ExperimentConfig
from .data import split_training_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--revision", required=True, help="Immutable checkpoint commit SHA.")
    parser.add_argument("--dimension", type=int, choices=DIMENSIONS, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--embedding-lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--validation-seed", type=int, default=42,
                        help="Keep this fixed across models, dimensions and training seeds.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = ExperimentConfig(**{name: getattr(args, name) for name in ExperimentConfig.__dataclass_fields__})
    bundle = validate_official_bundle(load_bundle(args.data, "Banking77"))
    train_rows, validation_rows, split = split_training_rows(
        bundle["train"], config.validation_fraction, config.validation_seed)
    plan = {"config": config.to_dict(), "model_id": config.model_id, "task": "Banking77",
            "token_embedding_dimension": config.dimension, "transformer_hidden_dimension": 3072,
            "sentence_vector_dimension": 3072, "classes": 77, "bundle_hash": fingerprint(bundle),
            "source": bundle["source"], "fit_source": bundle["fit_source"],
            "protocol": "banking77-trainable-token-embedding-validation-macro-f1-v2",
            "selection": "max validation macro-F1; exact ties keep earliest epoch; no full-train refit",
            "validation_split": split,
            "initialization": "pretrained_copy_or_uncentered_token_weight_truncated_svd",
            "representation": "last_valid_token_l2_normalized_plain_text_no_chat_no_added_eos"}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.output.exists():
        raise ValueError("Output exists; choose a fresh run directory.")
    # Must precede importing torch / CUDA initialization for deterministic GEMMs.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from torch.utils.data import DataLoader
    from .model import load_model
    from .checkpoint import save_checkpoint, load_checkpoint
    from .trainer import seed_everything, make_optimizer, evaluate, frozen_hash, parameter_hash
    from .selection import fit_with_validation
    seed_everything(config.seed)
    started = time.perf_counter()
    model, tokenizer = load_model(config)
    device = "cuda"

    def collate(rows):
        tokens = tokenizer([row["text"] for row in rows], padding=True, truncation=True,
                           max_length=config.max_length, return_tensors="pt")
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"],
                "labels": torch.tensor([int(row["label"]) for row in rows], dtype=torch.long)}

    generator = torch.Generator().manual_seed(config.seed)
    train = DataLoader(train_rows, batch_size=config.batch_size, shuffle=True,
                       generator=generator, collate_fn=collate, num_workers=0)
    validation = DataLoader(validation_rows, batch_size=config.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0)
    optimizer = make_optimizer(model, config)
    total_steps = config.epochs * math.ceil(len(train) / config.accumulation_steps)
    # Start with a positive LR so the first-step update assertion is meaningful.
    warmup = int(total_steps * config.warmup_fraction)
    from torch.optim.lr_scheduler import LambdaLR
    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))
    scheduler = LambdaLR(optimizer, lr_factor)
    baseline_frozen = frozen_hash(model)
    initial_embedding_hash = parameter_hash(model.embeddings.named_parameters())
    plan.update(trainable_parameters=model.trainable_manifest(), initialization=model.embeddings.initialization,
                backbone_config=model.backbone.config.to_dict(),
                tokenizer={"class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
                           "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id,
                           "padding_side": tokenizer.padding_side},
                frozen_backbone_initial_hash=baseline_frozen, initial_embedding_hash=initial_embedding_hash,
                packages={name: importlib.metadata.version(name) for name in
                          ("torch", "transformers", "accelerate", "safetensors", "huggingface-hub")},
                python=platform.python_version(), gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda, backbone_dtype=str(model.backbone_dtype), trainable_dtype="float32",
                effective_batch_size=config.batch_size * config.accumulation_steps,
                cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                deterministic_algorithms=True, tf32=False, weight_decay=0.0,
                train_ids=[row["id"] for row in train_rows],
                validation_ids=[row["id"] for row in validation_rows],
                test_ids=[row["id"] for row in bundle["evaluation"]])
    args.output.mkdir(parents=True, exist_ok=False)
    save_checkpoint(args.output / "initialization.pt", model, config, {"bundle_hash": plan["bundle_hash"]})
    (args.output / "manifest.json").write_text(json.dumps(plan, indent=2, allow_nan=False))
    history, selection = fit_with_validation(
        model, train, validation, optimizer, config, device, args.output,
        {"dataset_revision": bundle["source"]["revision"], "train_hash": split["train_hash"],
         "validation_hash": split["validation_hash"]}, scheduler)
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2, allow_nan=False))
    # Optimizer state belongs to the final training epoch, not the restored best
    # checkpoint. It is never reused; release it before inference/checkpoint checks.
    del optimizer, scheduler
    final_frozen = frozen_hash(model)
    final_embedding_hash = parameter_hash(model.embeddings.named_parameters())
    if baseline_frozen != final_frozen or initial_embedding_hash == final_embedding_hash:
        raise RuntimeError("Frozen integrity or trained embedding hash check failed.")
    save_checkpoint(args.output / "final.pt", model, config,
                    {"bundle_hash": plan["bundle_hash"], "epochs_completed": config.epochs,
                     "selection": selection})
    # Reload into the existing wrapper; the exact pinned frozen backbone is
    # unchanged. Compare logits on validation, before any test-model access.
    verification_batch = {k: v.to(device) for k, v in next(iter(validation)).items()}
    with torch.no_grad():
        before = model(**verification_batch)["logits"].clone()
    load_checkpoint(args.output / "final.pt", model, config)
    with torch.no_grad():
        after = model(**verification_batch)["logits"]
    torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
    # Selection and reload verification are now final. Official test is scored
    # once, using the restored best checkpoint; it never enters the fitting loop.
    test = DataLoader(bundle["evaluation"], batch_size=config.batch_size, shuffle=False,
                      collate_fn=collate, num_workers=0)
    scores, predictions = evaluate(model, test, device)
    torch.cuda.synchronize()
    results = {"scores": scores, "history": history, "selection": selection,
               "frozen_backbone_unchanged": True,
               "final_frozen_hash": final_frozen, "final_embedding_hash": final_embedding_hash,
               "checkpoint_reload_verified": True, "seconds": time.perf_counter() - started,
               "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(),
               "predictions": [{"id": row["id"], "label": int(row["label"]), "prediction": pred}
                               for row, pred in zip(bundle["evaluation"], predictions)]}
    (args.output / "results.json").write_text(json.dumps(results, indent=2, allow_nan=False))
    print(json.dumps({"output": str(args.output.resolve()), "scores": scores}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(f"Training stopped: {error}") from error
