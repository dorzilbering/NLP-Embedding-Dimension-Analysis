"""Banking77 internal embedding fine-tuning; one run per invocation.
Run with --model microsoft/Phi-4-mini-instruct --dimension 768 --max-steps 20
Experimental design: frozen transformer, trainable token embedding (+ trainable
projection for reduced dimensions), trainable 77-class classification head.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class ReducedEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, dimension: int):
        super().__init__()
        vocab, hidden = source.weight.shape
        assert 0 < dimension < hidden
        self.table = nn.Embedding(vocab, dimension, device=source.weight.device, dtype=source.weight.dtype)
        self.projection = nn.Linear(dimension, hidden, bias=False, device=source.weight.device, dtype=source.weight.dtype)
        with torch.no_grad():
            self.table.weight.copy_(source.weight[:, :dimension])
            self.projection.weight.zero_()
            self.projection.weight[:dimension].copy_(torch.eye(dimension, device=source.weight.device, dtype=source.weight.dtype))

    def forward(self, input_ids):
        return self.projection(self.table(input_ids))


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--dimension", type=int, default=768)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train = load_rows(Path(args.data_dir) / "train.csv")
    validation = load_rows(Path(args.data_dir) / "validation.csv")
    assert len(train) == 9002 and len(validation) == 1001
    labels = sorted({int(row["label"]) for row in train})
    assert labels == list(range(77))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    backbone = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        low_cpu_mem_usage=True
    ).to("cuda")
    backbone.config.use_cache = False
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    original = backbone.get_input_embeddings()
    native_dim = original.weight.shape[1]
    if args.dimension == native_dim:
        internal_embedding = original
    else:
        internal_embedding = ReducedEmbedding(original, args.dimension)
        backbone.set_input_embeddings(internal_embedding)
    for parameter in internal_embedding.parameters():
        parameter.requires_grad_(True)

    head = nn.Linear(native_dim, 77, device="cuda", dtype=torch.bfloat16)
    optimizer = torch.optim.SGD(
        list(internal_embedding.parameters()) + list(head.parameters()),
        lr=args.lr
    )

    def forward(batch):
        encoded = tokenizer(
            [row["text"] for row in batch], return_tensors="pt",
            padding=True, truncation=True, max_length=args.max_length
        ).to("cuda")
        targets = torch.tensor([int(row["label"]) for row in batch], device="cuda")
        outputs = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"], use_cache=False
        )
        positions = encoded["attention_mask"].sum(1) - 1
        pooled = outputs.last_hidden_state[
            torch.arange(len(batch), device="cuda"), positions
        ]
        return head(pooled).float(), targets

    backbone.eval()  # Frozen transformer, no dropout
    head.train()
    order = np.random.default_rng(args.seed).permutation(len(train))
    logs = []
    for step in range(args.max_steps):
        start = (step * args.batch_size) % len(order)
        idx = order[start:start + args.batch_size]
        if len(idx) < args.batch_size:
            order = np.random.default_rng(args.seed + step).permutation(len(train))
            idx = order[:args.batch_size]
        batch = [train[int(i)] for i in idx]
        optimizer.zero_grad(set_to_none=True)
        logits, targets = forward(batch)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        grad = next(internal_embedding.parameters()).grad
        grad_norm = grad.float().norm().item() if grad is not None else 0.0
        assert grad_norm > 0, "Internal embedding has no nonzero gradient"
        optimizer.step()
        logs.append({"step": step + 1, "loss": float(loss.item()), "embedding_grad_norm": grad_norm})
        if (step + 1) % 5 == 0:
            print(f"step={step + 1} loss={loss.item():.4f} embedding_grad={grad_norm:.4f}", flush=True)

    # Validation only; official test remains untouched.
    head.eval()
    predictions, truths = [], []
    with torch.no_grad():
        for start in range(0, len(validation), args.batch_size):
            logits, targets = forward(validation[start:start + args.batch_size])
            predictions.extend(logits.argmax(-1).cpu().tolist())
            truths.extend(targets.cpu().tolist())
    from sklearn.metrics import accuracy_score, f1_score
    result = {
        "model": args.model, "native_dimension": native_dim,
        "embedding_dimension": args.dimension,
        "train_steps": args.max_steps, "train_examples": len(train),
        "validation_examples": len(validation),
        "validation_accuracy": accuracy_score(truths, predictions),
        "validation_macro_f1": f1_score(truths, predictions, average="macro", labels=list(range(77)), zero_division=0),
        "seed": args.seed, "batch_size": args.batch_size,
        "max_length": args.max_length, "optimizer": "SGD",
        "lr": args.lr, "transformer_frozen": True,
        "reduced_initialization": "first-dimensions truncation + identity-pad projection",
        "status": "pilot_not_final"
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "train_log.json").write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
