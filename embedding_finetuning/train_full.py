"""Banking77 full fine-tuning with trainable internal token embeddings.
Runs one model/dimension. The transformer is NEVER frozen.
Validation chooses the best checkpoint; official test is evaluated once after training.
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class ReducedEmbedding(nn.Module):
    def __init__(self, source, width):
        super().__init__()
        vocab, hidden = source.weight.shape
        self.table = nn.Embedding(vocab, width, device=source.weight.device, dtype=source.weight.dtype)
        self.projection = nn.Linear(width, hidden, bias=False, device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.table.weight.copy_(source.weight[:, :width])
            self.projection.weight.zero_()
            self.projection.weight[:width].copy_(torch.eye(width, device=source.weight.device, dtype=source.weight.dtype))

    def forward(self, ids):
        return self.projection(self.table(ids))


class BankingDataset(Dataset):
    def __init__(self, filename):
        with open(filename, newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file))
        assert all("text" in r and "label" in r for r in self.rows)
        assert all(0 <= int(r["label"]) < 77 for r in self.rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dimension", type=int, required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-steps", type=int, default=0, help="0 = full epochs; positive = limited diagnostic run")
    p.add_argument("--skip-test", action="store_true", help="Use for calibration, not final experiment")
    p.add_argument("--no-checkpoint", action="store_true", help="Do not save huge weights in diagnostic runs")
    p.add_argument("--save-every", type=int, default=250, help="Optimizer updates between resumable checkpoints; 0 disables periodic saves")
    p.add_argument("--resume", action="store_true", help="Resume from latest_training.pt in output-dir")
    args = p.parse_args()
    assert args.epochs > 0 and args.batch_size > 0 and args.grad_accum > 0
    assert args.lr > 0 and torch.cuda.is_available()
    assert not (args.resume and args.no_checkpoint), "Resume requires checkpointing"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    train_set = BankingDataset(data_dir / "train.csv")
    val_set = BankingDataset(data_dir / "validation.csv")
    test_set = BankingDataset(data_dir / "test.csv")
    assert (len(train_set), len(val_set), len(test_set)) == (9002, 1001, 3080)
    assert set(int(r["label"]) for r in train_set.rows) == set(range(77))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def collate(rows):
        encoded = tokenizer(
            [r["text"] for r in rows], return_tensors="pt",
            padding=True, truncation=True, max_length=args.max_length
        )
        labels = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
        return encoded, labels

    def train_loader_for_epoch(epoch):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        return DataLoader(train_set, batch_size=args.batch_size, shuffle=True, generator=generator, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    backbone = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to("cuda")
    backbone.config.use_cache = False
    native = backbone.get_input_embeddings().weight.shape[1]
    assert 0 < args.dimension <= native
    if args.dimension < native:
        backbone.set_input_embeddings(ReducedEmbedding(backbone.get_input_embeddings(), args.dimension))
    backbone.requires_grad_(True)
    backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    head = nn.Linear(native, 77, device="cuda", dtype=torch.float32)

    # SGD avoids the very large optimizer state of AdamW for a fully trainable 3B+ model.
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=args.lr
    )
    embedding = backbone.get_input_embeddings()
    embedding_param = next(embedding.parameters())
    embedding_ids = {id(x) for x in embedding.parameters()}
    transformer_param = next(
        x for x in backbone.parameters() if id(x) not in embedding_ids and x.ndim >= 2
    )

    def forward(encoded):
        encoded = encoded.to("cuda")
        hidden = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False
        ).last_hidden_state
        positions = encoded["attention_mask"].sum(1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device="cuda"), positions]
        return head(pooled.float())

    def evaluate(loader):
        backbone.eval()
        head.eval()
        predictions, actual = [], []
        total_loss = 0.0
        with torch.inference_mode():
            for encoded, labels in loader:
                logits = forward(encoded)
                total_loss += F.cross_entropy(logits, labels.to("cuda"), reduction="sum").item()
                predictions.extend(logits.argmax(-1).cpu().tolist())
                actual.extend(labels.tolist())
        return {
            "loss": total_loss / len(actual),
            "accuracy": accuracy_score(actual, predictions),
            "macro_f1": f1_score(actual, predictions, labels=list(range(77)), average="macro", zero_division=0),
        }

    config = vars(args).copy()
    config.update({
        "native_hidden_size": native,
        "transformer_frozen": False,
        "embedding_trainable": True,
        "reduced_embedding_initialization": "first columns plus zero-padded identity projection",
        "optimizer": "SGD",
        "checkpoint_selection": "highest validation macro F1",
        "test_policy": "once after best checkpoint is restored",
    })
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = []
    best_f1 = -1.0
    global_steps = 0
    checkpoint = output / "best_model.pt"
    latest = output / "latest_training.pt"
    start_epoch = 1
    start_batch = 0

    if args.resume:
        if not latest.exists():
            raise FileNotFoundError(f"No resume checkpoint: {latest}")
        state = torch.load(latest, map_location="cpu", weights_only=False)
        saved_config = state["run_config"]
        for field in ("model", "dimension", "batch_size", "grad_accum", "max_length", "lr", "seed"):
            assert saved_config[field] == getattr(args, field), f"Resume configuration mismatch: {field}"
        backbone.load_state_dict(state["backbone"])
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        best_f1 = state["best_f1"]
        global_steps = state["global_steps"]
        start_epoch = state["next_epoch"]
        start_batch = state["next_batch"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed at epoch={start_epoch}, batch={start_batch}, updates={global_steps}", flush=True)

    def save_resume(next_epoch, next_batch):
        if args.no_checkpoint:
            return
        state = {
            "run_config": vars(args).copy(),
            "backbone": backbone.state_dict(),
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "global_steps": global_steps,
            "next_epoch": next_epoch,
            "next_batch": next_batch,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        }
        temporary = output / "latest_training.tmp.pt"
        torch.save(state, temporary)
        os.replace(temporary, latest)
        print(f"Resume checkpoint saved at update {global_steps}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = train_loader_for_epoch(epoch)
        backbone.train()
        head.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        batches = 0
        for batch_index, (encoded, labels) in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            logits = forward(encoded)
            loss = F.cross_entropy(logits, labels.to("cuda"))
            (loss / args.grad_accum).backward()
            total_loss += loss.item()
            batches += 1
            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % args.grad_accum == 0 or last_batch:
                assert embedding_param.grad is not None and embedding_param.grad.float().norm().item() > 0, "Missing embedding gradient"
                assert transformer_param.grad is not None and transformer_param.grad.float().norm().item() > 0, "Missing transformer gradient"
                torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(head.parameters()), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_steps += 1
                if global_steps % 100 == 0:
                    print(f"epoch={epoch} update={global_steps} batch_loss={loss.item():.4f}", flush=True)
                if args.save_every and global_steps % args.save_every == 0:
                    save_resume(epoch if not last_batch else epoch + 1, batch_index + 1 if not last_batch else 0)
                if args.max_train_steps and global_steps >= args.max_train_steps:
                    break

        validation = evaluate(val_loader)
        entry = {"epoch": epoch, "updates": global_steps, "train_loss": total_loss / max(batches, 1), "validation": validation}
        history.append(entry)
        (output / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(json.dumps(entry), flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            if not args.no_checkpoint:
                torch.save({
                    "backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                }, checkpoint)
                print(f"Saved best checkpoint: {checkpoint}", flush=True)
        save_resume(epoch + 1, 0)
        if args.max_train_steps and global_steps >= args.max_train_steps:
            break

    result = {
        "model": args.model,
        "dimension": args.dimension,
        "native_hidden_size": native,
        "epochs_completed": len(history),
        "optimizer_updates": global_steps,
        "best_validation_macro_f1": best_f1,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "status": "diagnostic_not_final" if args.max_train_steps else "full_training",
    }
    if not args.skip_test:
        assert not args.no_checkpoint, "Final test requires a saved checkpoint"
        assert not args.max_train_steps, "Do not evaluate official test during diagnostic training"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        backbone.load_state_dict(saved["backbone"])
        head.load_state_dict(saved["head"])
        result["best_epoch"] = saved["epoch"]
        result["test"] = evaluate(test_loader)
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
