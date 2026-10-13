"""Full-backbone + internal embedding training smoke test (not a final experiment).
Every backbone parameter and the 77-class head receives gradients.
SGD avoids Adam optimizer-state memory; batch=1, length=32 for A100 40GB.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


class ReducedEmbedding(nn.Module):
    def __init__(self, original, width):
        super().__init__()
        vocab, hidden = original.weight.shape
        self.table = nn.Embedding(vocab, width, device=original.weight.device, dtype=original.weight.dtype)
        self.projection = nn.Linear(width, hidden, bias=False, device=original.weight.device, dtype=original.weight.dtype)
        with torch.no_grad():
            self.table.weight.copy_(original.weight[:, :width])
            self.projection.weight.zero_()
            self.projection.weight[:width].copy_(torch.eye(width, device=original.weight.device, dtype=original.weight.dtype))

    def forward(self, input_ids):
        return self.projection(self.table(input_ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--dimension", type=int, default=768)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-length", type=int, default=32)
    args = parser.parse_args()
    torch.manual_seed(42)
    assert torch.cuda.is_available()

    with open(Path(args.data_dir) / "train.csv", newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained(
        args.model, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True
    ).to("cuda")
    backbone.config.use_cache = False
    native = backbone.get_input_embeddings().weight.shape[1]
    assert args.dimension <= native
    if args.dimension < native:
        embedding = ReducedEmbedding(backbone.get_input_embeddings(), args.dimension)
        backbone.set_input_embeddings(embedding)
    else:
        embedding = backbone.get_input_embeddings()

    # No backbone freezing: all model parameters trainable.
    backbone.requires_grad_(True)
    backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    head = nn.Linear(native, 77, device="cuda", dtype=torch.bfloat16)
    backbone.train()
    head.train()

    encoded = tokenizer(
        [row["text"]], return_tensors="pt", padding=True,
        truncation=True, max_length=args.max_length
    ).to("cuda")
    target = torch.tensor([int(row["label"])], device="cuda")

    embedding_param = next(embedding.parameters())
    # Representative transformer parameter: must not be an embedding parameter.
    embedding_ids = {id(p) for p in embedding.parameters()}
    transformer_param = next(
        p for name, p in backbone.named_parameters()
        if id(p) not in embedding_ids and p.ndim >= 2
    )
    token_ids = encoded["input_ids"].unique()\n    before_embedding = embedding.table.weight.detach()[token_ids].float().clone() if isinstance(embedding, ReducedEmbedding) else embedding.weight.detach()[token_ids].float().clone()
    before_transformer = transformer_param.detach().flatten()[:128].float().clone()

    # SGD intentionally has no per-parameter optimizer state, unlike AdamW.
    optimizer = torch.optim.SGD(
        list(backbone.parameters()) + list(head.parameters()), lr=0.01
    )
    optimizer.zero_grad(set_to_none=True)
    out = backbone(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        use_cache=False
    )
    position = encoded["attention_mask"].sum(1) - 1
    pooled = out.last_hidden_state[
        torch.arange(len(target), device="cuda"), position
    ]
    logits = head(pooled).float()
    loss = F.cross_entropy(logits, target)
    loss.backward()

    emb_grad = embedding_param.grad
    transformer_grad = transformer_param.grad
    emb_norm = float(emb_grad.float().norm()) if emb_grad is not None else 0.0
    transformer_norm = float(transformer_grad.float().norm()) if transformer_grad is not None else 0.0
    assert emb_norm > 0, "No internal embedding gradient"
    assert transformer_norm > 0, "No transformer gradient"
    optimizer.step()
    result = {
        "model": args.model, "embedding_dimension": args.dimension,
        "transformer_frozen": False, "loss": float(loss.item()),
        "embedding_gradient_norm": emb_norm,
        "transformer_gradient_norm": transformer_norm,
        "embedding_weight_change_used_tokens": float(((embedding.table.weight if isinstance(embedding, ReducedEmbedding) else embedding.weight).detach()[token_ids].float() - before_embedding).abs().max()),
        "transformer_weight_change_sample": float((transformer_param.detach().flatten()[:128].float() - before_transformer).abs().max()),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "status": "single_step_smoke_test_not_final"
    }
    assert result["embedding_weight_change_used_tokens"] > 0, "Embedding weights of used tokens did not change"\n    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
