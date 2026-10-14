# Banking77: Trainable Internal Embedding Dimension — Experiment Contract

Status: design gate; implementation and results pending.

## Lecturer's requested extension (authoritative)
- Two of the original models.
- Banking77 task.
- Original embedding dimension and two reduced dimensions per model (six configurations).
- **Modify the internal Embedding component and train it**; post-hoc dimensionality reduction of output representations does not qualify.

## Scope and preservation
- Keep the existing frozen-encoder/PCA experiments unchanged and explicitly separate from this extension.
- Reuse official Banking77 train/test provenance and integrity checks from `pilot/banking77.py`.
- No reported scores or training claims until experiments actually execute.

## Architecture acceptance gate (must pass before training)
1. Identify the exact model checkpoint, revision, internal input embedding module, its weight shape, hidden width, and whether input/output embeddings are tied.
2. Establish the exact meaning of 'embedding dimension' for the selected architecture. The changed dimension must be that of a **trainable internal token embedding**, not pooled sentence vectors.
3. For reduced variants, consider `Embedding(vocab_size, d) -> trainable Linear(d, hidden_size)` installed **inside** the model input pathway. The original-dimension control must use an appropriately matched trainable embedding setup; document differences in parameter count and initialization.
4. If an output LM head shares input embedding weights, resolve tying correctly (e.g., classification backbone without an LM head); do not silently break weight sharing.
5. Confirm gradient flows through the modified embedding weights and optimizer includes them. Record nonzero gradient norm and pre/post optimizer-step parameter delta.
6. Define initialization carefully (e.g., factorization of pretrained *token embedding weights*), and distinguish this from PCA on produced text representations. Document initialization for all configurations.
7. Use a classification head for 77 intents, train on official train data only, and reserve an internal validation split from train for model selection. Keep official test untouched until final evaluation.
8. Set identical training budgets, sequence length, random seeds, early stopping rules, and metric definitions across dimensions; report memory/runtime and parameter counts.
9. Run a tiny smoke test on one model/dimension first: shape assertions, finite loss, backward pass, optimizer step, checkpoint reload, and learning signal.
10. Review the architecture interpretation against the lecturer's wording before launching all six full runs. If uncertainty remains about whether the projection-based design qualifies, request clarification.

## Suggested models (provisional)
- Phi-4-mini-instruct: native hidden width 3072; candidate embedding widths 3072 / 1536 / 768.
- Llama-3.2-3B: native hidden width 3072; candidate embedding widths 3072 / 1536 / 768.
Verify model-specific architecture, access and checkpoint revisions before committing to this pair.

## Output requirements
- Reproducible configs, exact dependency versions, hardware metadata, seeds, split IDs, checkpoints or hashes, training curves.
- Six result rows with test accuracy and optional macro-F1; transparent parameter counts and limitations.
- Separate report section describing architecture changes, optimization, fairness controls and comparison to prior post-hoc PCA experiment.
- Unit tests that fail if the embedding is frozen, outside the model, wrong width, or not updated by training.

## Execution milestones
1. Inspect architecture/checkpoints and decide design.
2. Implement modular training path on this branch with Codex or reviewed code.
3. Pass unit and GPU smoke tests.
4. Execute six runs on Colab A100, verify artifacts.
5. Update report and merge only after review.
