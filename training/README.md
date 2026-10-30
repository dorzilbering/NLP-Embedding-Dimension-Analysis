# Banking77 trainable internal token embeddings — Phase 1

This package is independent of `pilot/`, `run_pilot.py`, and `run_experiment.py`.
It trains the internal token lookup, its optional input projection, and a linear
77-class classifier using supervised cross-entropy. All transformer parameters
are frozen, but autograd traverses the transformer back to the token weights.
There is no sentence-vector PCA or frozen representation cache in this path.

## Architecture and initialization

Supported checkpoints are `microsoft/Phi-4-mini-instruct` and the **base**
`meta-llama/Llama-3.2-3B`, not its Instruct variant. Both require an explicit
immutable 40-character checkpoint SHA. Tokenizer and backbone use that same SHA,
`trust_remote_code=False`, and the existing repository Transformers pin.
Llama access requires Hugging Face authorization/license acceptance.

Each model has three independent configurations: internal token lookup widths
3072, 1536, and 768. For reduced widths the lookup is followed by a trainable,
bias-free projection to 3072. Transformer hidden width and sentence-vector width
remain **3072 in all six configurations**. This experiment measures internal
token embedding width/rank, not a smaller transformer or smaller exported vectors.

At native width the pretrained lookup is copied exactly. Reduced configurations
use uncentered truncated SVD of the **pretrained vocabulary weight matrix** W.
Stream chunks into G = W.T @ W in CPU float64, compute its leading eigenvectors Q,
then initialize lookup E = W @ Q and projection P = Q. Effective token weights
are E @ P.T = W @ Q @ Q.T. No task examples or labels enter initialization.
Record the relative Frobenius reconstruction error. Eigenvector signs are fixed;
degenerate eigenspaces and BLAS differences can still alter factors across hosts.
Save `initialization.pt` to preserve the exact factors. This CPU initialization
has substantial compute/RAM costs for the real vocabulary; it is not benchmarked.

Only E, P (when present), and classifier weight/bias are in AdamW. They remain
float32; the frozen backbone runs BF16. No weight decay is used, so unused token
rows do not change solely due to regularization. An actual padding row is excluded
from embedding gradients. A tokenizer EOS-as-padding fallback is masked using
the explicit attention mask, not by comparing token IDs to the padding ID.

The model installs the new module through `set_input_embeddings`. The classifier
wrapper calls this installed module and supplies differentiably cast
`inputs_embeds` to the backbone. It pools the last attention-mask-valid final
hidden state and L2-normalizes it. Text is verbatim, with tokenizer-default special
tokens, no chat template and no manually appended EOS. Frozen backbone dropout is
disabled. Optional non-reentrant HF gradient checkpointing keeps its required
training flag enabled, with dropout modules disabled explicitly.

## Running later, after GPU verification

Reuse the existing official Banking77 JSON as-is, including the Google Drive copy.
Raw validation requires full 10,003/3,080 splits, 77 labels, immutable revision,
hashes, original IDs/order and duplicate-label checks. The full official export
explicitly permits the reported seven canonical train/test matches; other overlap
counts fail. Unrelated datasets retain strict leakage rejection. If preparing a new
export, the pinned official dataset loader uses `trust_remote_code=True` for both
configuration discovery and loading; model loading still uses `False`.

```bash
python -m training.run --model Phi4-mini \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --dimension 768 --data data/banking77_official.json \
  --output results/trainable_phi_768_seed42 --dry-run
```

The dry run uses only the standard library, validates the local dataset and prints
the plan; it neither downloads models nor creates output directories. Removing
`--dry-run` opts into real training and requires BF16-capable CUDA hardware.
For Llama, use `--model Llama3.2-3B` with a resolved SHA for the base checkpoint.
Use a fresh output directory per configuration and seed. Existing directories
are rejected for new runs; `--resume` explicitly reuses the verified original run.
Checkpoint files are immutable and never overwritten.

Defaults: training seed 42, five maximum epochs, microbatch 1, accumulation 32, maximum length
256, lookup/projection LR 1e-4, classifier LR 1e-3, 10% linear warmup, linear decay,
gradient norm clipping 1.0, AdamW weight decay 0. These are unvalidated starting
settings. Accumulation scales by actual examples, including final short windows.
Deterministic PyTorch algorithms, explicit cuBLAS workspace configuration and
disabled TF32 are used; unsupported deterministic kernels fail rather than silently
relaxing reproducibility. There is no automatic quantization or offload fallback.

Before validation splitting, exclude every train row whose `text_key` matches any
test text. This deterministic in-memory view never edits the raw JSON or test set.
Policy `banking77-preserve-raw-exclude-overlapping-train-v1` records excluded IDs,
counts, per-row and unique canonical-text SHA256 hashes, original split hashes and
eligible-training hash under `training_exclusions`. Repeated matching train rows
are all excluded, so the row count can exceed seven. No test label is consulted
to decide exclusions. Abort if exclusion removes a class.

Validation uses only the remaining eligible official training rows. Defaults are
`--validation-fraction 0.1 --validation-seed 42`. Keep this validation seed fixed
across both models, all dimensions, and all training seeds. Group duplicate
canonical texts together, stratify by the 77 labels, and select a group subset
per class nearest its rounded target validation count. A deterministic subset-sum
algorithm achieves the target exactly when group sizes permit it; count ties
prefer the smaller holdout. Seeded SHA256 ordering resolves equivalent subset
choices. Record achieved per-class counts, exact IDs and partition hashes.
Every eligible train row remains in exactly one partition, in original order;
excluded rows remain solely in the original raw bundle.
Both partitions must contain every class. Unsplittable classes fail explicitly.

Train only the resulting subtrain partition. Evaluate validation after each epoch
and select strictly higher **macro-F1**, keeping the earliest epoch on exact ties;
accuracy and loss do not break ties. Save each improving checkpoint under a new
versioned name, and restore the selected weights after the maximum epoch budget.
There is no early stopping, hyperparameter search, or full-training refit.
`final.pt` contains the selected checkpoint's adaptation weights, which need not
be the last epoch's weights. Validation evaluation uses no gradients or updates.

The selection function accepts only train and validation loaders. The runner
constructs the official test loader only after selection, weight restoration and
reload verification finish; test is then scored once. Reading the source bundle
for integrity and text-based decontamination precedes model selection. Test
labels, logits and scores never guide exclusions, optimization or checkpoint selection.
Test input texts **do** determine eligibility: this is a test-text-informed protocol,
not a test-blind one. Document this dependency and changed training population.
Legacy PCA keeps all official train rows and may retain overlap contamination;
historical scores/outputs are preserved and are not a controlled baseline for the
filtered supervised experiment. Canonical matching does not detect paraphrases,
and the seven-count guard is not independent authentication of the actual texts.
Do not tune settings from test scores.
For the complete matrix run both models at 3072/1536/768, independently initialized.
Repeat seeds 42/43/44 if reporting variability, with equal training exposure.

## Outputs and evidence

- `manifest.json`: configuration, base revision, dataset provenance/hashes/IDs,
  dimensions, initialization error, parameter manifest, environment and precision.
  Includes the independent validation seed, achieved class counts, and exact
  training/validation membership and hashes.
  Includes raw-data-preserving exclusions and canonical-text hashes.
- `initialization.pt`, improving `epoch-NNN.pt`, and selected `final.pt`:
  adaptation-only weights and configuration;
  frozen transformer weights are reconstructed from the exact base SHA. Reload
  with `load_model(config)` then `load_checkpoint(path, model, config)`.
- `training_metrics.jsonl`: nested training and validation loss/accuracy, validation
  macro-F1, whether the epoch improved selection, example count, optimizer steps,
  observed-token gradients and actual embedding/projection deltas from the first
  optimizer step of each epoch. Evidence covers up to 128 observed non-padding
  token IDs; it is not a claim that every vocabulary row changed.
- `results.json`: final test loss/accuracy/macro-F1, row-aligned predictions,
  selected epoch/validation score/checkpoint and the complete validation history,
  exact frozen-weight integrity checks, embedding hashes, reload-logit check,
  elapsed time and peak allocated CUDA memory.
- `selection.json`: finalized validation selection, written before any test scoring,
  so selected epoch/score remain reviewable even if test evaluation fails.
- `resume-STEP-PHASE-TIMESTAMP-ID.pt` plus `.sha256`: complete training snapshots,
  including adaptation, optimizer/scheduler, epoch/global step, completed batch
  cursor, partial epoch metrics, selection history, experiment/run/data identity,
  Python/NumPy/torch/CUDA RNG and loader-generator state. Phase 1 follows completed
  validation; phase 0 is initialization or an optimizer-boundary snapshot.

`initialization.pt`, `epoch-*.pt` and `final*.pt` remain adaptation-only artifacts.
Training recovery uses the new `resume-*.pt` snapshots, not those inference files.
The initialization checkpoint is essential evidence: factorization alone changes
weights, whereas gradients and post-optimizer deltas establish actual training.

## Crash recovery: exact commands

Start a new Phi run (these settings must stay identical on resume):

```bash
python -m training.run --model Phi4-mini \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --dimension 768 --data data/banking77_official.json \
  --output results/trainable_phi_768_seed42 \
  --seed 42 --epochs 5 --batch-size 1 --accumulation-steps 32 \
  --max-length 256 --validation-seed 42 --validation-fraction 0.1 \
  --checkpoint-every-steps 50
```

After an interruption, keep that output directory and raw dataset intact and run:

```bash
python -m training.run --model Phi4-mini \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --dimension 768 --data data/banking77_official.json \
  --output results/trainable_phi_768_seed42 \
  --seed 42 --epochs 5 --batch-size 1 --accumulation-steps 32 \
  --max-length 256 --validation-seed 42 --validation-fraction 0.1 \
  --checkpoint-every-steps 50 --resume
```

For any model/dimension, including Llama with its recorded immutable revision,
the exact original configuration can also be read from the manifest, avoiding
manual transcription (set `run_dir` and `dataset_path` to your existing files):

```python
import json
import subprocess
import sys
from pathlib import Path

run_dir = Path("results/trainable_phi_768_seed42")
dataset_path = Path("data/banking77_official.json")
config = json.loads((run_dir / "manifest.json").read_text())["config"]
command = [sys.executable, "-m", "training.run", "--data", str(dataset_path),
           "--output", str(run_dir), "--resume"]
for name, value in config.items():
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            command.append(flag)
    else:
        command.extend([flag, str(value)])
subprocess.run(command, check=True)
```

Snapshots are saved before the first optimizer step, every 50 optimizer steps by
default (configurable for a new run), and after every validation epoch. Saving
occurs only after a complete accumulation window. On a disconnect, roll back to
the latest valid snapshot and replay unsaved windows, including a partly computed
window. Epoch-seeded shuffling recreates the sample order with `num_workers=0`;
completed batches are skipped without advancing restored model RNG. Validation
selection resumes from its saved history and best checkpoint. Official test data
still enters model evaluation only after selection is finalized.

`--resume` checks the exact configuration, raw bundle hash, eligible/subtrain/
validation hashes, protocol, run ID, dependency versions, frozen backbone and
precision before continuing. It will not extend the epoch budget or change batch
size, seed, checkpoint interval, data, or other experiment settings. A missing or
corrupted newest checkpoint/sidecar is ignored in favor of an older valid snapshot;
a valid foreign checkpoint is an error, not a silent fallback. Old runs without
resume snapshots are not resumable. A completed run returns its saved scores
without retraining or scoring test again. If scoring was interrupted before its
atomic results commit, recovery may repeat final scoring with the already-fixed
selected model; test scores never feed back into selection.

All writes use same-directory temporary files, flush/fsync, and atomic rename.
SHA256 sidecars detect incomplete or corrupted snapshots. A POSIX process lock
prevents simultaneous writers and is automatically released after process death.
Checkpoint versions and the original manifest/initialization are preserved.
Only owned journals (`training_metrics.jsonl`, `selection.json`) are atomically
rebuilt on recovery; epoch history comes from the checkpoint to avoid duplicates.
If `final.pt` already exists after an interrupted finalization, write a new
`final-ID.pt`; `results.json.final_checkpoint` identifies the completed artifact.

Use persistent storage that supports atomic rename, file/directory fsync and POSIX
locking. Colab's temporary `/content` storage does not survive runtime deletion.
Google Drive/FUSE persistence and sync semantics must be smoke-tested; a local
atomic rename does not guarantee that a remote Drive upload has finished. Copies
of a run must include manifest, resume snapshots/sidecars, and referenced best
checkpoints/sidecars. Do not copy partially synced files into a different run.
Snapshots include very large Adam state and all versions are retained: budget
disk space, host RAM and checkpoint I/O for native-width vocabulary tables before
an A100 run. No automatic pruning or cross-environment bitwise guarantee is claimed.

## CPU validation and A100 follow-up

`python -m pytest tests/test_training.py` uses tiny random Phi3 and Llama models,
never downloaded checkpoints. It checks all requested dimensions, SVD correctness,
internal replacement, cross-entropy logits, gradient flow, real updates, frozen
weights, gradient checkpointing, accumulation, padding, and independent-model
checkpoint reload. The full existing suite also checks preservation of the pilots.
Additional tests cover deterministic stratification, grouped duplicate isolation,
training/test separation, preserved rows, class coverage, invalid splits,
macro-F1-only selection and ties, restoration of an earlier best checkpoint,
validation without parameter updates, and untouched validation-only lookup rows.
`python -m pytest tests/test_resume.py` covers optimizer/scheduler/RNG equality
after mid-window and mid-epoch crashes, corrupt-checkpoint fallback, configuration
and dataset mismatch rejection, atomic-write failure, process locking, and CLI
recovery after training or finalization interruptions without clobbering artifacts.

Still verify on an A100 before full experiments: exact pinned checkpoint configs
and tokenizer behavior, authorized Llama access, BF16/SDPA gradient flow and padding
invariance, deterministic CUDA kernels, real-vocabulary factorization RAM/time,
optimizer plus activation VRAM, checkpoint disk budget, reload against downloaded
base weights, and official dataset authenticity. Native tables contain hundreds
of millions of trainable parameters; prior inference feasibility on a T4 does not
establish training feasibility. No real checkpoint, dataset preparation, or GPU
training is claimed by Phase 1 unit tests.
