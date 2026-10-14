"""Adaptation-only checkpoints; frozen weights come from the pinned HF revision."""
from pathlib import Path
import torch


def save_checkpoint(path, model, config, metadata=None):
    path = Path(path)
    payload = {"schema_version": 1, "config": config.to_dict(),
               "adaptation": model.adaptation_state(),
               "initialization": model.embeddings.initialization,
               "metadata": metadata or {}}
    # Never silently replace another run's checkpoint.
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path, model, config):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("config") != config.to_dict():
        raise ValueError("Checkpoint schema or experiment configuration mismatch.")
    model.load_adaptation_state(payload["adaptation"])
    model.embeddings.initialization = payload["initialization"]
    return payload["metadata"]
