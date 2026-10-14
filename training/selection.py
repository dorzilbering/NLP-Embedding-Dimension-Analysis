"""Training and checkpoint selection have no official test-loader interface."""
import json
import math
from pathlib import Path

from .checkpoint import save_checkpoint, load_checkpoint
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
                        device, output, provenance, scheduler=None):
    """Fit only subtrain, score only validation, restore chosen weights before return.

    Save a new versioned adaptation checkpoint for every improvement; no existing
    file is overwritten. The caller may construct a test loader only after this
    function has finalized selection. No full-train refit is performed.
    """
    output = Path(output)
    selector, history, best_path = MacroF1Selector(), [], None
    for epoch in range(1, config.epochs + 1):
        training = train_epoch(model, train_batches, optimizer, device, config.accumulation_steps,
                               config.max_grad_norm, scheduler)
        validation, _ = evaluate(model, validation_batches, device)
        improved = selector.consider(epoch, validation)
        record = {"epoch": epoch, "training": training, "validation": validation, "selected": improved}
        history.append(record)
        if improved:
            best_path = output / f"epoch-{epoch:03d}.pt"
            save_checkpoint(best_path, model, config,
                            {"provenance": provenance, "epoch": epoch,
                             "selection_metric": "validation_macro_f1", "validation": validation})
        with (output / "training_metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
    restored = load_checkpoint(best_path, model, config)
    model.eval()
    selection = {"metric": "validation_macro_f1", "tie_break": "earliest_epoch",
                 "best_epoch": selector.best_epoch, "best_score": selector.best_score,
                 "checkpoint": best_path.name, "validation": restored["validation"],
                 "test_access_during_selection": False, "full_train_refit": False}
    return history, selection
