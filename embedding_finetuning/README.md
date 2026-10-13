# Banking77 internal embedding fine-tuning — supplementary experiment

**Status: design stage; no fine-tuning results have been produced yet.**

This supplement responds to the instructor's request: **two models, Banking77, native embedding dimension plus two reduced dimensions per model, with an actual internal embedding change and training**. The existing frozen-encoder/PCA evaluations remain unchanged and must not be presented as this supplement.

## Planned matrix (six independent trained configurations)

| Model | Native | Reduced 1 | Reduced 2 |
|---|---:|---:|---:|
| Llama3.2-3B | 3072 | 1536 | 768 |
| Phi4-mini | 3072 | 1536 | 768 |

These are **hidden/token embedding widths**, not vocabulary sizes or post-hoc PCA output dimensions.

## Proposed architecture — verify before full training

For each backbone, replace the token-embedding lookup with a trainable internal module:

- **Native baseline:** embedding lookup of width H (the original width), trainable.
- **Reduced runs:** embedding lookup of width d, followed **inside the model's input path** by a trainable linear projection d → H before the transformer layers. This keeps the transformer's required hidden width H while changing the actual learned embedding table width to d.
- Initialize the reduced embedding and projection with a documented reproducible policy; use the same initialization policy across comparable runs.
- Feed the adapted token embeddings via `inputs_embeds` to the backbone and train a 77-way classification head. The reduced embedding and projection **must** receive gradients and change weights during optimizer steps.
- To keep Colab resource use manageable, initially freeze transformer blocks and train the internal embedding adapter and classification head. **This is a design proposal, not a confirmed interpretation of the instructor's expected training scope.** If the instructor expects broader model fine-tuning, change the protocol before full runs.

The classification head is not a substitute for training the embedding module. Do **not** fit PCA to frozen output vectors and call that fine-tuning.

## Before the full six runs

1. Verify exact checkpoint compatibility, token embedding access, `inputs_embeds` behavior, padding/pooling and model licenses in the Colab environment.
2. Run a tiny smoke test for all three widths on **one** model: assert lookup output dimension d, projection output H, finite loss, nonzero embedding gradients, and actual embedding weight updates.
3. Confirm trainable parameter lists and optimizer coverage, including native baseline; verify frozen parameters remain frozen if using a frozen backbone.
4. Check peak GPU memory and runtime; do not promise that both 3B+ models will fit the free Colab tier.
5. Pin software and model revisions; use official Banking77 train/test (10,003/3,080), with a stratified validation split **only from train** for selection. Keep the official test untouched until final evaluation.
6. Fix seed, epochs/steps, max sequence length, learning-rate schedule, batch/effective batch size, optimizer, early stopping/selection and compute budget before comparing dimensions. Train all six configurations under comparable rules.
7. Save logs, trainable-module checkpoints, config, revision, dataset provenance, accuracy and macro-F1 for every run; record failures and reruns honestly.
8. Produce a six-row result table and plots. Explain that d→H projection preserves the transformer width, so the experiment varies the internal token embedding **bottleneck** rather than every transformer layer width.

## Reproducibility notes

Do not mix these training results with the 84 **evaluation-only** PCA configurations. Colab GPU availability and model access are external prerequisites. The architecture and training protocol must be validated before calling the experiment assignment-compliant.
