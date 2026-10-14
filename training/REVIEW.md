# Correctness review before GPU experiments

Reviewed against the lecturer's internal-token-embedding requirements.

| Requirement | Implementation and evidence |
| --- | --- |
| Two original-project models | Explicit Phi-4-mini-instruct and base Llama-3.2-3B IDs; require immutable revisions and checked HF architecture/3072 hidden width. Real checkpoints still need GPU/access verification. |
| Three dimensions per model | Config accepts exactly 3072/1536/768; the installed lookup is V x d and reduced configurations have a learned d -> 3072 input projection. |
| Modify and train internal embeddings | `set_input_embeddings` installs the replacement. The wrapper uses that exact module; cross-entropy gradients traverse the frozen transformer to lookup/projection weights. Tests verify gradients and optimizer changes, including native width. |
| Frozen backbone | Freeze before replacement; optimizer checks its exact parameter set. Tests and runtime hashes verify frozen weights. Gradient checkpointing remains differentiable. |
| Pretrained initialization | Native copy; reduced uncentered truncated SVD of token weights, without task examples or sentence representations. Temporary initialization does not advance classifier RNG. |
| Banking77 classification | 77-way linear classifier on normalized last-valid-token hidden state, supervised cross-entropy, accuracy and macro-F1. Reuses official source/split validation. |
| Validation isolation | Exclude test-text-matching train rows first, preserving the full raw export. Derive stratified grouped partitions from eligible train rows; retain all classes and eligible rows. Separate seed and IDs/hashes support shared splits. |
| Model selection | Validation macro-F1 alone; earliest exact tie. Versioned improving checkpoints are restored before any test-model evaluation. Selection API accepts no test loader. |
| Reproducibility and checkpoints | Immutable model/data provenance, seeded shuffle, deterministic algorithms, fixed optimization settings, initialization artifact, adaptation checkpoints and environment manifest. Reload checks and CPU tests verify logits. |
| Preserve original experiments | Shared validation now explicitly accepts known full-official Banking77 overlap; future PCA metadata audits it. Historical PCA fitting semantics and existing artifacts remain unchanged. |

The dimension here is the **internal token lookup dimension**. Transformer and
sentence-vector widths remain 3072; the input projection does not shrink the
backbone. This scope must remain explicit in the academic report. The pretrained
checkpoint's original table is replaced by trainable factor weights, rather than
retained as a frozen lookup followed by a sentence-vector reducer.

The validation extension corrects the prior fixed-final-epoch protocol. Training
now excludes held-out validation rows, and `final.pt` exports the selected epoch
rather than the last epoch. No full-train refit is performed. The official test
bundle is inspected for integrity and test-text-informed training exclusions;
its labels/scores do not decide exclusions or checkpoint selection. This is not
a test-blind protocol and its reduced training population must be reported.
All raw rows and test rows are unchanged. Initialization depends
only on pretrained token weights, so validation text also does not fit the SVD.

Remaining limitations: adaptation checkpoints do not include optimizer/RNG state
for resume; deterministic results across different BLAS/CUDA releases are not
guaranteed; per-step update evidence samples observed rows rather than claiming
all vocabulary rows change. Improving checkpoint versions and native-size token
tables require significant disk, host RAM, optimizer VRAM and activation VRAM.
Defaults are starting settings, not validated learning-rate recommendations.

Before full training, verify actual pinned models/tokenizers, authorized Llama
access, official dataset authenticity, BF16/SDPA and deterministic CUDA kernels,
gradient checkpointing, padding invariance, real-vocabulary factorization cost,
A100 memory at all three widths, and checkpoint reload against the frozen base.
Tiny random CPU architecture tests establish mechanics, not real-model performance.
