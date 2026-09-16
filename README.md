# Embedding Dimension Analysis

Reproducible implementation for evaluating five frozen LLMs on STSB, Banking77, SciFact, and ArxivClusteringS2S at the embedding dimensions required by the assignment.

## Experiment matrix

| Model | Checkpoint | Dimensions |
|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | 4096, 2048, 1024, 512, 256 |
| Gemma3-4B | `google/gemma-3-4b-it` | 2560, 1280, 640, 320 |
| Llama3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 3072, 1536, 768, 384 |
| Phi4-mini | `microsoft/Phi-4-mini-instruct` | 3072, 1536, 768, 384 |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 4096, 2048, 1024, 512 |

This is 84 model/task/dimension configurations.

## Reduction protocol

Models remain frozen. Native representations are the final hidden state at the last attention-mask-valid token. Reduced representations use PCA. One training-only calibration manifest is shared across tasks; a separate PCA basis is fitted for each model at its largest reduced dimension and smaller dimensions use prefixes of that basis. No benchmark test examples fit PCA.

`prepare_calibration.py` builds the common manifest from official Banking77 training text plus STSB training sentences, deduplicated by normalized text, and records immutable source revisions and a manifest hash. The same text IDs are used for all five models.

The previously executed Phi4-mini/STSB 300-pair run is retained as a **pilot**, not treated as completion of the final benchmark matrix.

## Task protocols

- **STSB:** cosine similarity / Spearman correlation.
- **Banking77:** official train/test, logistic regression, accuracy and macro-F1. Official duplicate texts are preserved; train/test normalized-text overlap is documented rather than silently deleting benchmark rows.
- **SciFact:** official MTEB test queries/corpus/qrels, exact cosine ranking, nDCG@10, Recall@10, Recall@100 and MRR@10. SciFact does not fit PCA.
- **ArxivClusteringS2S:** evaluate each official clustering set with MiniBatchKMeans and macro-average across sets. PCA comes only from the shared external training calibration.

## T4 loading

Target hardware is a Colab Tesla T4. Phi4-mini, Gemma3-4B and Llama3.2-3B use automatic 16-bit loading. Qwen3-8B and Mistral-7B use bitsandbytes 4-bit weight loading as a VRAM execution constraint; the extracted embeddings and PCA computations remain float32.

## Prepare data

```bash
python prepare_banking77.py --output data/banking77_official.json
python prepare_scifact.py --output data/scifact_official.json
python prepare_arxiv.py --output data/arxiv_official.json
python prepare_calibration.py --output data/pca_calibration.json
```

Keep these exported bundles with the results because they contain dataset provenance.

## Run

Configuration-only inspection:

```bash
python run_experiment.py --model Phi4-mini --task Banking77 --data data/banking77_official.json --calibration data/pca_calibration.json --dry-run
```

Real run:

```bash
python run_experiment.py --model Phi4-mini --task Banking77 --data data/banking77_official.json --calibration data/pca_calibration.json --output results/Phi4-mini/Banking77
```

The selected model defaults to all dimensions required by the assignment. Outputs include `metrics.csv`, `results.json`, `predictions.json`, and `pca.npz` when reduction is requested. Native embeddings are cached separately from output dimension.

## Validate

```bash
python -m pytest -q
python run_experiment.py --model Phi4-mini --task Banking77 --dry-run
```

Tests are synthetic/offline. A dry-run validates configuration only; it does not prove GPU compatibility. Run one real T4 smoke test before starting the full sweep.

## Structure

- `pilot/config.py`: five-model matrix and dimensions.
- `pilot/model.py`: generic frozen Hugging Face encoder and T4 loading policy.
- `pilot/core.py`: normalization, PCA, projection and embedding cache.
- `pilot/evaluation.py`: downstream metrics.
- `pilot/task_data.py`: bundle validation/provenance.
- `pilot/task_runner.py`: shared-calibration PCA and task evaluation.
- `pilot/arxiv.py`: Arxiv per-set evaluation.
- `prepare_*.py`: data/calibration exporters.
- `run_experiment.py`: assignment entry point.
- `run_pilot.py`: preserved historical Phi/STSB pilot.

## Reporting rule

The final report must distinguish executed results from planned configurations. Never invent scores for failed/unexecuted runs. Record model/checkpoint, dataset revisions, calibration manifest hash, dimensions, seed, loading/quantization strategy, package versions, GPU, truncation and runtime. The shared calibration PCA is a documented methodological choice to make reductions comparable while avoiding test fitting; it is not presented as an explicit requirement of the assignment prompt.
