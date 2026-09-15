# Embedding Dimensionality and NLP Performance

This project studies how reducing frozen language-model text embeddings affects
NLP performance. The current pilot uses **Phi-4-mini and the English STS Benchmark**.

**Status:** implementation and lightweight tests are complete. The real GPU
experiment has **not been executed**; no experimental results are included.
GPU experiments will run remotely. No LLM fine-tuning is performed.

## Pilot methodology

| Setting | Value |
|---|---|
| Model | `microsoft/Phi-4-mini-instruct` |
| Dataset | `mteb/stsbenchmark-sts` (English) |
| Dimensions | **3072 native, 768 PCA, 384 PCA** |
| PCA calibration | 2,000 unique training sentences |
| Evaluation | 300 validation pairs; seed 42 |
| Metric | Cosine Spearman correlation with human similarity scores |
| Maximum input length | 256 tokens |

1. Deduplicate training sentences and exclude normalized sentence matches from
   **all validation and test data** before selecting PCA calibration texts.
   Test labels are not used; test sentences supply overlap exclusions only.
2. Freeze the backbone and extract its final normalized hidden state at the
   **last position with attention mask 1**. Use plain text, tokenizer-default
   special tokens, no chat template, and no added EOS. The pinned tokenizer
   disables automatic BOS/EOS. Use BF16 where supported, otherwise FP16.
3. Cache native pooled float32 embeddings for reuse across dimensions. Cache
   identity includes model revision, text order, extraction settings, and environment.
4. L2-normalize training embeddings and fit one seeded 768-component PCA with
   centering and **no whitening**, using training data only. For 384 dimensions,
   use its first 384 principal components. Apply the frozen transform to validation
   vectors and L2-normalize again. The 3072 baseline uses native L2-normalized vectors.
5. Compute sentence-pair cosine similarities and Spearman correlation, save the
   three scores, and generate the dimension-versus-score plot.

This uses MTEB's English STS data and cosine-Spearman metric directly, without the
full MTEB runner. The validation subset is not a full benchmark/leaderboard result.
PCA centering differs from the native baseline, so this pilot does not isolate
centering from dimensionality effects. PCA reduces vector storage and downstream
cost, not model memory or encoding cost.

## Hardware

Use a remote **NVIDIA CUDA GPU with 16 GB VRAM or more**. Recommend 32 GB system RAM
and at least 15 GB free disk. The runner requires at least 10 GiB free GPU memory
and refuses CPU model execution before downloads. This guard is a lower bound,
not a guarantee; reduce `--batch-size` to 1 if memory is insufficient.

## Installation

Use Python 3.11 or 3.12. From the repository root on the remote GPU machine:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-dev.txt
python -m pip check
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
The CUDA wheel requires a compatible NVIDIA driver; consult the
[PyTorch installation matrix](https://pytorch.org/get-started/previous-versions/#v271)
for other supported builds. For CPU-only tests, substitute the wheel index
`https://download.pytorch.org/whl/cpu`. This does not enable CPU execution of the pilot.

## Lightweight tests

```bash
python -m pytest -q
```

Nine offline tests cover shapes, finite values, PCA, leakage, scoring, caches,
output generation, padding, and the GPU guard. They use temporary fixtures and a
tiny randomly initialized architecture, without pretrained weights or downloads.
The real dataset loader was also previously verified. Pretrained GPU encoding and
FP16/BF16 behavior remain unverified.

## Run the pilot (remote GPU only)

```bash
python run_pilot.py --check-hardware
python run_pilot.py --train-sentences 2000 --validation-pairs 300 --batch-size 4 --max-length 256 --seed 42 --output results/pilot
```

The second command downloads weights and data on its first run. Use a new
`--output` directory for subsequent runs; matching native caches are reused.

## Outputs and reproducibility

After a successful GPU run:

```text
results/pilot/
  scores.csv                    # Three dimension/score rows
  results.json                  # Settings, revisions, checks, timing, hardware
  predictions.json              # Per-pair similarities and validation indices
  pca.npz                       # Training-fitted mean and components
  dimension_vs_spearman.png      # Measured scores only
cache/native/*.npz              # Reusable native embeddings
```

Default immutable revisions:

- Model: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Dataset: `96943a16ea6a35129e253c659081cb59daf81b30`

Overrides use `--model-revision` and `--dataset-revision`. Results record resolved
revisions, calibration hashes, validation indices, package versions, hardware, and
truncation counts. Direct dependencies are pinned. After a successful run, save
all installed versions with:

```bash
python -m pip freeze > results/pilot/environment.txt
```

Scores use correlation scale [-1, 1]. Numerical identity across GPU platforms is
not guaranteed. Results, caches, weights, and environments are ignored by Git.
Review and explicitly select compact real results for the later academic
submission; never publish model weights or credentials.

## Source files

- `run_pilot.py`: pilot CLI, orchestration, results, and plotting.
- `pilot/model.py`: frozen Phi loading, pooling, batching checks, and GPU guard.
- `pilot/core.py`: data selection, validation, caching, PCA, and scoring.
- `tests/test_pilot.py`: offline correctness tests.
- `requirements.txt`, `requirements-dev.txt`: runtime and test dependencies.

## References

- [Phi-4-mini model card](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [English STS dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [MTEB semantic similarity tasks](https://docs.mteb.org/overview/available_tasks/semantic-similarity/)
- [scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
