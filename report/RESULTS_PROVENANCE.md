# Results provenance and publication status

- `experiment_summary.csv` is a **manual transcription of the six final scores reported by the experiment owner**; it is not a direct copy of the original `results.json` files.
- The six training runs were performed externally in Google Colab. Original run manifests, checkpoints, epoch histories and predictions are **not currently committed** to this branch.
- `report/generate_figures.py` expects an experiment summary with `model,dimension,test_macro_f1,test_accuracy` and an epoch history with `model,dimension,epoch,val_macro_f1,val_accuracy,train_loss,val_loss`.
- The summary CSV alone is **insufficient to regenerate all 11 figures**. Do not synthesize missing per-epoch values.
- To make the published results independently auditable, export the original `results.json`/run manifests and `training_history.csv` from the Colab/Drive runs, review them for private paths or tokens, then add suitably small sanitized outputs.
- Phi-4 Mini SVD initialization relative Frobenius reconstruction errors reported separately: 1536: 0.25113159850670635; 768: 0.3511465454611678. These are initialization matrix-reconstruction errors, **not** classification errors.
- Raw Banking77 records, Hugging Face model weights, optimizer snapshots and large artifacts are intentionally excluded.
