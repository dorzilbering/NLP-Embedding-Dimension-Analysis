# Banking77 — Trainable Token Embedding Dimension Analysis

Research code for studying how the **internal trainable token embedding width** affects 77-class intent classification on the official [Banking77](https://huggingface.co/datasets/PolyAI/banking77) dataset.

**Current submission experiment:** two pretrained models × three internal embedding widths = **six configurations**. The frozen transformer retains a hidden width of 3072 throughout; this is **not** a smaller backbone, and it is **not** PCA applied to exported sentence vectors.

## Experimental design

| Setting | Value |
| --- | --- |
| Models | `microsoft/Phi-4-mini-instruct`; `meta-llama/Llama-3.2-3B` (base) |
| Internal token embedding widths | 3072 (native), 1536, 768 |
| Trainable parameters | Replacement token lookup; input projection when reduced; 77-way classification head |
| Frozen parameters | Pretrained transformer backbone |
| Initialization | Exact pretrained token-table copy (3072); truncated uncentered SVD of pretrained token weights (1536/768) |
| Data | Banking77 official train 10,003 and test 3,080; 77 classes |
| Leakage handling | Seven canonical train/test text overlaps excluded **from training**; raw official dataset preserved |
| Validation | Grouped stratified split of eligible training examples (~10%); select checkpoint by validation macro-F1 |
| Training | 5 epochs, seed 42, effective batch 32, max sequence length 256; cross-entropy; AdamW |
| Learning rates | Lookup/projection 1e-4; classifier 1e-3 |
| Evaluation | Official test split, accuracy and macro-F1 |

The reduced embedding is projected back to 3072 before entering the frozen transformer. Dimensional reduction and initialization/projection architecture change together, so the experiments **do not isolate dimension as the sole causal factor**. Total GPU memory and inference speed savings were **not established** by these classification scores.

## Reported results

The table below transcribes the six final test scores supplied by the experiment owner. It is also available in [`report/experiment_summary.csv`](report/experiment_summary.csv). These are **reported outcomes**, not scores independently reproduced by a fresh GitHub CI run.

| Model | Embedding width | Test macro-F1 | Test accuracy |
| --- | ---: | ---: | ---: |
| Llama 3.2 3B | 3072 | 0.7488 | 0.7659 |
| Llama 3.2 3B | 1536 | **0.8448** | **0.8468** |
| Llama 3.2 3B | 768 | 0.7962 | 0.8029 |
| Phi-4 Mini | 3072 | 0.3646 | 0.3890 |
| Phi-4 Mini | 1536 | 0.7053 | 0.7211 |
| Phi-4 Mini | 768 | **0.7171** | **0.7308** |

## Repository map

- [`training/`](training/): training pipeline, SVD initialization, trainable lookup/projection, selection, checkpoints and resume.
- [`training/README.md`](training/README.md): execution commands, required dataset format, artifacts and recovery procedure.
- [`training/REVIEW.md`](training/REVIEW.md): correctness review and limitations.
- [`tests/`](tests/): unit tests including training, checkpoint recovery and train/test overlap checks.
- [`report/generate_figures.py`](report/generate_figures.py): regenerate 11 plots as PNG/PDF from exported summary and per-epoch CSV files.
- [`report/experiment_summary.csv`](report/experiment_summary.csv): six reported test results.
- [`pilot/`](pilot/), `run_pilot.py`, `run_experiment.py`: **earlier frozen-representation/PCA experiments**, retained for provenance; **not the method used for this Banking77 trainable-embedding report**.

## Reproducing the training

Use a CUDA GPU with BF16 support, appropriate VRAM/disk capacity and access to both Hugging Face model checkpoints. The official Banking77 JSON export is required and intentionally not committed. Install the dependencies in [`requirements-colab.txt`](requirements-colab.txt) in an appropriate CUDA-enabled environment (the file does not install PyTorch).

Example **validation-only dry run** after preparing `data/banking77_official.json`:

```bash
python -m training.run --model Phi4-mini \
  --revision cfbefacb99257ffa30c83adab238a50856ac3083 \
  --dimension 768 --data data/banking77_official.json \
  --output results/trainable_phi_768_seed42 --dry-run
```

Remove `--dry-run` to train. The command above specifies the Phi checkpoint SHA; for Llama, use the **base** model and its verified immutable SHA. Run all six model/width combinations in separate output directories. See [the detailed training guide](training/README.md) for resume and artifact instructions.

Run offline unit tests:

```bash
python -m pytest -q
```

## Reproducing the figures

The plot generator requires **both** `experiment_summary.csv` and `training_history.csv` in the same input directory. The per-epoch history CSV and original checkpoints are not included in this GitHub branch; they must be exported from the actual training runs, not fabricated from the final scores.

```bash
python report/generate_figures.py --data /path/to/exported/figure_data
```

This repository intentionally excludes large checkpoints, raw datasets, and run outputs via `.gitignore`. See [`report/RESULTS_PROVENANCE.md`](report/RESULTS_PROVENANCE.md) for what is and is not published.

## Scope and limitations

Only one training seed and five epochs were reported. The baseline uses a direct pretrained copy, whereas reduced widths use SVD factors and a learned projection. Checkpoint selection uses validation macro-F1; the official test split is for final evaluation. The earlier PCA experiment documented elsewhere in this repository is a **separate study** and should not be mixed with these six trainable-embedding results.
