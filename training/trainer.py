"""CPU-testable supervised training mechanics and parameter update evidence."""
import hashlib
from itertools import islice
import random
import numpy as np

import torch
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def make_optimizer(model, config):
    parameters = list(model.embeddings.parameters()) + list(model.classifier.parameters())
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    if actual != {id(p) for p in parameters}:
        raise RuntimeError("Unexpected trainable backbone parameters.")
    return torch.optim.AdamW([
        {"params": model.embeddings.parameters(), "lr": config.embedding_lr},
        {"params": model.classifier.parameters(), "lr": config.classifier_lr},
    ], weight_decay=0.0)


def parameter_hash(parameters):
    """Stream exact bytes, including BF16, without a full backbone CPU copy."""
    digest = hashlib.sha256()
    for name, parameter in parameters:
        digest.update(name.encode())
        digest.update(str((tuple(parameter.shape), parameter.dtype)).encode())
        flat = parameter.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_000_000):
            block = flat[start:start + 1_000_000].cpu().contiguous()
            digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_hash(model):
    return parameter_hash((name, p) for name, p in model.named_parameters() if not p.requires_grad)


def train_epoch(model, batches, optimizer, device, accumulation_steps=1,
                max_grad_norm=1.0, scheduler=None, *, resume_state=None, on_step=None):
    """Scale by actual window examples, including short final batches/windows."""
    model.train()
    total_loss, correct, count, steps = 0.0, 0, 0, 0
    evidence = None
    consumed_batches = 0
    if resume_state is not None:
        total_loss, correct, count, steps = (resume_state[key] for key in
                                           ("total_loss", "correct", "count", "steps"))
        evidence = resume_state["evidence"]
        consumed_batches = resume_state["consumed_batches"]
    # Reconstruct deterministic epoch order and skip completed accumulation
    # windows. Iterator construction/collation must not advance resumed RNG.
    from .checkpoint import capture_rng, restore_rng
    rng = capture_rng() if resume_state is not None else None
    iterator = iter(batches)
    for _ in range(consumed_batches):
        try:
            next(iterator)
        except StopIteration as error:
            raise ValueError("Resume batch cursor exceeds epoch length.") from error
    if rng is not None:
        restore_rng(rng)
    while window := list(islice(iterator, accumulation_steps)):
        window_count = sum(batch["labels"].numel() for batch in window)
        optimizer.zero_grad(set_to_none=True)
        observed = []
        for batch in window:
            batch = {key: value.to(device) for key, value in batch.items()}
            output = model(**batch)
            n = batch["labels"].numel()
            loss = output["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            (loss * (n / window_count)).backward()
            total_loss += loss.detach().item() * n
            correct += (output["logits"].detach().argmax(-1) == batch["labels"]).sum().item()
            count += n
            observed.append(batch["input_ids"][batch["attention_mask"].bool()])
        if evidence is None:
            ids = torch.cat(observed).unique()
            padding = model.embeddings.padding_idx
            if padding is not None:
                ids = ids[ids != padding]
            ids = ids[:128]
            gradient = model.embeddings.lookup.weight.grad
            if not len(ids) or gradient is None:
                raise RuntimeError("No embedding gradient for observed tokens.")
            grad_norm = gradient[ids].float().norm().item()
            if not torch.isfinite(gradient[ids]).all() or grad_norm == 0:
                raise RuntimeError("Embedding gradients are zero or non-finite.")
            before = model.embeddings.lookup.weight.detach()[ids].cpu().clone()
            projection_before = {name: p.detach().cpu().clone()
                                 for name, p in model.embeddings.projection.named_parameters()}
            projection_gradient = sum(p.grad.float().norm().item()
                                      for p in model.embeddings.projection.parameters() if p.grad is not None)
            if projection_before and projection_gradient == 0:
                raise RuntimeError("No learned input projection gradient.")
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                      max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        steps += 1
        consumed_batches += len(window)
        if evidence is None:
            delta = model.embeddings.lookup.weight.detach()[ids].cpu() - before
            if not delta.abs().any():
                raise RuntimeError("Optimizer did not update internal token weights.")
            projection_delta = sum((p.detach().cpu() - projection_before[name]).norm().item()
                                   for name, p in model.embeddings.projection.named_parameters())
            if projection_before and projection_delta == 0:
                raise RuntimeError("Optimizer did not update input projection weights.")
            evidence = {"observed_token_ids": ids.cpu().tolist(), "embedding_gradient_norm": grad_norm,
                        "changed_sample_rows": int(delta.abs().any(1).sum()),
                        "sample_delta_norm": delta.norm().item(), "sample_max_abs_delta": delta.abs().max().item(),
                        "projection_gradient_norm": projection_gradient,
                        "projection_delta_norm": projection_delta}
        if on_step is not None:
            on_step({"total_loss": total_loss, "correct": correct, "count": count,
                     "steps": steps, "evidence": evidence, "consumed_batches": consumed_batches})
    if not count:
        raise ValueError("Empty training loader.")
    return {"loss": total_loss / count, "accuracy": correct / count,
            "examples": count, "optimizer_steps": steps, "update_evidence": evidence}


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    total_loss, labels, predictions = 0.0, [], []
    for batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        if not torch.isfinite(output["logits"]).all():
            raise RuntimeError("Non-finite evaluation logits.")
        total_loss += F.cross_entropy(output["logits"], batch["labels"], reduction="sum").item()
        labels.extend(batch["labels"].cpu().tolist())
        predictions.extend(output["logits"].argmax(-1).cpu().tolist())
    if not labels:
        raise ValueError("Empty evaluation loader.")
    f1s = []
    for label in range(77):
        tp = sum(y == label and p == label for y, p in zip(labels, predictions))
        fp = sum(y != label and p == label for y, p in zip(labels, predictions))
        fn = sum(y == label and p != label for y, p in zip(labels, predictions))
        f1s.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return {"loss": total_loss / len(labels),
            "accuracy": sum(y == p for y, p in zip(labels, predictions)) / len(labels),
            "macro_f1": sum(f1s) / 77, "examples": len(labels)}, predictions
