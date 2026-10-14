"""Atomic adaptation artifacts and resumable training snapshots with integrity checks."""
from pathlib import Path
import contextlib
import fcntl
import json
import hashlib
import os
import pickle
import random
import tempfile
import time
import uuid
import numpy as np
import torch


def atomic_write(path, writer, replace=False):
    """Flush a same-directory temporary file, then atomically rename it.

    Callers serialize run access with run_lock. Immutable checkpoints never
    replace an existing destination; replace=True is only for owned run journals.
    """
    path = Path(path)
    if not replace and path.exists():
        raise FileExistsError(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        if not replace and path.exists():
            raise FileExistsError(path)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value, replace=False):
    data = json.dumps(value, indent=2, allow_nan=False).encode()
    atomic_write(path, lambda handle: handle.write(data), replace)


@contextlib.contextmanager
def run_lock(output):
    """POSIX advisory lock is released by the OS when the process dies."""
    with (Path(output) / ".run.lock").open("a+b") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process is using this run directory.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), *state[2:]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs from checkpoint.")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, config, metadata=None):
    path = Path(path)
    payload = {"schema_version": 1, "config": config.to_dict(),
               "adaptation": model.adaptation_state(),
               "initialization": model.embeddings.initialization,
               "metadata": metadata or {}}
    # Never silently replace another run's checkpoint.
    atomic_write(path, lambda handle: torch.save(payload, handle))
    atomic_write(Path(str(path) + ".sha256"), lambda handle: handle.write(file_digest(path).encode()))


def load_checkpoint(path, model, config):
    checksum = Path(str(path) + ".sha256")
    if checksum.exists() and checksum.read_text().strip() != file_digest(path):
        raise ValueError("Adaptation checkpoint checksum mismatch.")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("config") != config.to_dict():
        raise ValueError("Checkpoint schema or experiment configuration mismatch.")
    model.load_adaptation_state(payload["adaptation"])
    model.embeddings.initialization = payload["initialization"]
    return payload["metadata"]


def save_resume_checkpoint(output, model, optimizer, scheduler, config, identity, progress, phase=0):
    payload = {"schema_version": 2, "kind": "training_resume", "config": config.to_dict(),
               "identity": identity, "adaptation": model.adaptation_state(),
               "initialization": model.embeddings.initialization, "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "rng": capture_rng(), "progress": progress}
    path = Path(output) / (f"resume-{progress['global_step']:012d}-{phase}-{time.time_ns()}-{uuid.uuid4().hex}.pt")
    atomic_write(path, lambda handle: torch.save(payload, handle))
    checksum = file_digest(path).encode()
    atomic_write(Path(str(path) + ".sha256"), lambda handle: handle.write(checksum))
    return path


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def latest_resume_checkpoint(output, config, identity):
    """Ignore torn/unreadable files, but never skip a valid foreign checkpoint."""
    for path in sorted(Path(output).glob("resume-*.pt"), reverse=True):
        try:
            expected = Path(str(path) + ".sha256").read_text().strip()
            if expected != file_digest(path):
                continue
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != "training_resume":
            raise ValueError("Unexpected checkpoint in run directory.")
        if payload.get("schema_version") != 2 or payload.get("config") != config.to_dict():
            raise ValueError("Resume experiment configuration mismatch.")
        if payload.get("identity") != identity:
            raise ValueError("Resume run identity/dataset hashes mismatch.")
        required = {"adaptation", "optimizer", "scheduler", "rng", "progress", "initialization"}
        if not required <= payload.keys():
            continue
        progress = payload["progress"]
        progress_keys = {"epoch", "global_step", "partial", "history", "selector", "best_path"}
        if not isinstance(progress, dict) or not progress_keys <= progress.keys():
            continue
        if (type(progress["epoch"]) is not int or not 1 <= progress["epoch"] <= config.epochs + 1 or
                type(progress["global_step"]) is not int or progress["global_step"] < 0 or
                not isinstance(progress["history"], list) or not isinstance(progress["selector"], dict)):
            continue
        best = progress["best_path"]
        if best:
            if Path(best).name != best:
                raise ValueError("Unsafe best checkpoint path in resume state.")
            try:
                if Path(str(Path(output) / best) + ".sha256").read_text().strip() != file_digest(Path(output) / best):
                    continue
                chosen = torch.load(Path(output) / best, map_location="cpu", weights_only=True)
                if chosen.get("config") != config.to_dict():
                    raise ValueError("Best checkpoint configuration mismatch.")
                if chosen["metadata"]["epoch"] != progress["selector"]["best_epoch"]:
                    raise ValueError("Best checkpoint epoch mismatch.")
            except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, KeyError):
                continue
        return payload
    raise ValueError("No valid resumable checkpoint found; adaptation-only checkpoints cannot resume training.")


def restore_training_checkpoint(payload, model, optimizer, scheduler):
    if (payload["scheduler"] is None) != (scheduler is None):
        raise ValueError("Resume scheduler mismatch.")
    model.load_adaptation_state(payload["adaptation"])
    model.embeddings.initialization = payload["initialization"]
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return payload["progress"]
