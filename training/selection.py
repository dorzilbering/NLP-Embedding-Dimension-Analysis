"""Training and checkpoint selection have no official test-loader interface."""
import json
import math
import time
from pathlib import Path
import uuid

from .checkpoint import (save_checkpoint, load_checkpoint, save_resume_checkpoint,
                         restore_training_checkpoint, atomic_write)
from .trainer import train_epoch, evaluate


class MacroF1Selector:
    """Strictly higher validation macro-F1 wins; exact ties keep earlier epoch."""

    def __init__(self):
        self.best_epoch = None
        self.best_score = None
        self.last_epoch = 0

    def consider(self, epoch, validation_metrics):
        score = validation_metrics["macro_f1"]
        if type(epoch) is not int or epoch <= self.last_epoch:
            raise ValueError("Selection epochs must be strictly increasing positive integers.")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Validation macro-F1 must be finite and in [0, 1].")
        self.last_epoch = epoch
        if self.best_score is None or score > self.best_score:
            self.best_epoch, self.best_score = epoch, score
            return True
        return False


def fit_with_validation(model, train_batches, validation_batches, optimizer, config,
                        device, output, provenance, scheduler=None, *, identity=None, resume_payload=None):
    """Fit only subtrain, score only validation, restore chosen weights before return.

    Save a new versioned adaptation checkpoint for every improvement; no existing
    file is overwritten. The caller may construct a test loader only after this
    function has finalized selection. No full-train refit is performed.
    """
    output = Path(output)
    started_at = time.monotonic()
    steps_per_epoch = math.ceil(len(train_batches) / config.accumulation_steps)
    total_steps = steps_per_epoch * config.epochs
    def report(message):
        print(message, flush=True)
    selector, history, best_path = MacroF1Selector(), [], None
    epoch, global_step, partial = 1, 0, None
    if resume_payload is not None:
        if resume_payload.get("config") != config.to_dict() or resume_payload.get("identity") != identity:
            raise ValueError("Resume experiment configuration or run identity/dataset hashes mismatch.")
        progress = restore_training_checkpoint(resume_payload, model, optimizer, scheduler)
        epoch, global_step, partial = progress["epoch"], progress["global_step"], progress["partial"]
        history = progress["history"]
        selector.__dict__.update(progress["selector"])
        best_path = output / progress["best_path"] if progress["best_path"] else None

    def progress_state(current_epoch, partial_state):
        generator = getattr(train_batches, "generator", None)
        return {"epoch": current_epoch, "global_step": global_step, "partial": partial_state,
                "history": history, "selector": dict(selector.__dict__),
                "best_path": best_path.name if best_path else None,
                "data_loader_generator": generator.get_state() if generator is not None else None}

    def journal():
        data = "".join(json.dumps(record, allow_nan=False) + "\n" for record in history).encode()
        atomic_write(output / "training_metrics.jsonl", lambda handle: handle.write(data), replace=True)

    if identity is not None and resume_payload is None:
        save_resume_checkpoint(output, model, optimizer, scheduler, config, identity, progress_state(1, None))
    journal()
    session_start_step = global_step
    report(f"Training started/resumed: epoch {epoch}/{config.epochs}, step {global_step}/{total_steps}")
    for epoch in range(epoch, config.epochs + 1):
        # Dedicated generator and epoch seed recreate the same permutation after
        # a crash without using model RNG or consuming previous epochs again.
        generator = getattr(train_batches, "generator", None)
        if generator is not None:
            generator.manual_seed(config.seed + epoch)
        prior_steps = partial["steps"] if partial else 0
        epoch_start_step = global_step - prior_steps
        report(f"Epoch {epoch}/{config.epochs} started (global step {global_step}/{total_steps})")
        def on_step(state):
            nonlocal global_step
            global_step = epoch_start_step + state["steps"]
            if global_step % config.checkpoint_every_steps == 0:
                save_resume_checkpoint(output, model, optimizer, scheduler, config, identity,
                                       progress_state(epoch, state))
                elapsed = time.monotonic() - started_at
                completed_this_session = max(1, global_step - session_start_step)
                remaining = max(0, total_steps - global_step) * elapsed / completed_this_session
                report(f"Epoch {epoch}/{config.epochs} | step {global_step}/{total_steps} "
                       f"({global_step / total_steps:.1%}) | train loss {state['total_loss'] / max(1, state['count']):.4f} "
                       f"| elapsed {elapsed / 60:.1f}m | ETA {remaining / 60:.1f}m "
                       f"| checkpoint saved")
        training = train_epoch(model, train_batches, optimizer, device, config.accumulation_steps,
                               config.max_grad_norm, scheduler,
                               **({"resume_state": partial, "on_step": on_step} if identity is not None else {}))
        if identity is None:
            global_step = epoch_start_step + training["optimizer_steps"]
        partial = None
        validation, _ = evaluate(model, validation_batches, device)
        improved = selector.consider(epoch, validation)
        record = {"epoch": epoch, "training": training, "validation": validation, "selected": improved}
        history.append(record)
        if improved:
            best_path = output / f"epoch-{epoch:03d}.pt"
            if best_path.exists():
                best_path = output / f"epoch-{epoch:03d}-{uuid.uuid4().hex}.pt"
            save_checkpoint(best_path, model, config,
                            {"provenance": provenance, "epoch": epoch,
                             "selection_metric": "validation_macro_f1", "validation": validation})
        if identity is not None:
            save_resume_checkpoint(output, model, optimizer, scheduler, config, identity,
                                   progress_state(epoch + 1, None), phase=1)
        journal()
        report(f"Epoch {epoch}/{config.epochs} complete | train loss {training['loss']:.4f} "
               f"| validation macro-F1 {validation['macro_f1']:.4f} "
               f"| validation accuracy {validation['accuracy']:.4f} "
               f"| best epoch {selector.best_epoch}")
    restored = load_checkpoint(best_path, model, config)
    if restored.get("provenance") != provenance or restored.get("epoch") != selector.best_epoch:
        raise ValueError("Selected checkpoint provenance/epoch mismatch.")
    model.eval()
    selection = {"metric": "validation_macro_f1", "tie_break": "earliest_epoch",
                 "best_epoch": selector.best_epoch, "best_score": selector.best_score,
                 "checkpoint": best_path.name, "validation": restored["validation"],
                 "test_access_during_selection": False, "full_train_refit": False}
    return history, selection
