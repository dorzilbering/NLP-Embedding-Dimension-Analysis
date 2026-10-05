# Embedding Dimension Analysis

Final evaluation of **five frozen pretrained language models**, **four NLP tasks**,
and **84 model/task/dimension configurations**. The source results are the 20 JSON
files under `results/raw/`, containing 210 metric observations. The final matrix
was validated against source commit `88c9536f04dda2194c6cf20d5388588b4298171e`
on `assignment-completion`.

No language model was fine-tuned and no internal token embedding layer was retrained.
The 84 configurations are evaluations, not 84 trained language models. Banking77
trains downstream logistic-regression classifiers; Arxiv fits clustering centroids.

## Models and dimensions

The first dimension in each row is the pretrained model's native hidden width.

| Model | Pretrained checkpoint | Evaluated dimensions |
|---|---|---|
| Qwen3-8B | `Qwen/Qwen3-8B` | 4096, 2048, 1024, 512, 256 |
| Gemma3-4B | `google/gemma-3-4b-it` | 2560, 1280, 640, 320 |
| Llama3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` | 3072, 1536, 768, 384 |
| Phi4-mini | `microsoft/Phi-4-mini-instruct` | 3072, 1536, 768, 384 |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.3` | 4096, 2048, 1024, 512 |

There are 21 model/dimension combinations, each evaluated on four tasks: 84 total.

## Representation and PCA protocol

**Frozen pretrained native representation → PCA reduction where requested → downstream evaluation.**

`pilot/model.py` uses evaluation mode, disabled gradients and inference mode. Plain
texts are tokenized with right padding and truncation to the recorded maximum of
256 tokens. The native representation is the final hidden state at the last
attention-mask-valid token, converted to float32. It is a contextual representation,
not a learned replacement for the model's token embedding table.

One shared training-only calibration manifest contains 3,072 deterministically
selected, normalized-text-deduplicated Banking77 train texts and STSB train sentences.
Each model fits its own centered, unwhitened randomized PCA (seed 42) on normalized
native calibration vectors at the largest reduced dimension. Smaller dimensions
use prefixes of that ordered basis, reused across all four tasks. Evaluation vectors
are normalized, centered using the fitted calibration mean, projected and normalized
again. Native vectors are normalized without PCA centering; comparisons therefore
change geometry as well as dimension.

The preparation code excludes Banking77 test and STSB validation/test texts. SciFact
query/corpus and Arxiv test-text exclusions require supplying their optional bundle
paths. The original calibration bundle is absent from this checkout, so historical
exclusions cannot be independently verified from its recorded hash alone. All 20
results record the same calibration hash and count and one artifact path per model.

## Tasks and metrics

| Task | Actual evaluation protocol | Metrics |
|---|---|---|
| STSB | All 1,500 STSBenchmark **validation** pairs; cosine similarity | Cosine Spearman |
| Banking77 | Official 10,003 train / 3,080 test rows; logistic regression per dimension | Accuracy, macro-F1 |
| SciFact | 300 official MTEB test queries; exact cosine ranking over the corpus, lexical document-ID tie-breaking, linear relevance gain | nDCG@10, MRR@10, Recall@10, Recall@100 |
| Arxiv-Clustering | `ArxivClusteringS2S`; MiniBatchKMeans per official test set, K from its label taxonomy; unweighted mean across 30 non-degenerate sets | V-measure, NMI, adjusted Rand index |

Arxiv excludes degenerate set `test:29` and evaluates 707,723 sentence occurrences
across the remaining sets. Centroids are fitted on each test set under this
clustering protocol; PCA is fitted only on external calibration. V-measure and
arithmetic NMI are equivalent under the implemented definitions, not independent
confirmation. Banking77 preserves official duplicate rows and records overlap
information during bundle validation rather than silently removing benchmark rows.

## Recorded loading and hardware

Qwen and Mistral use bitsandbytes NF4 four-bit weight loading with double quantization.
Other models select float16 or bfloat16 by GPU capability. The `t4-auto` identifier
names a loading policy; most final files record **NVIDIA L4**, while Phi4-mini/STSB
records **Tesla T4**. Actual dtype and historical package versions are not recorded.
Extracted vectors and PCA use float32. Do not present all final runs as T4 runs or
interpret cross-model differences as isolated effects of model identity.

## Reproduce the analysis without experiments

Python 3.12 was used for the verified analysis. From the repository root:

```bash
python -m venv .venv-analysis
# POSIX shell: source .venv-analysis/bin/activate
# PowerShell: .\.venv-analysis\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python analyze_results.py
python -m pytest tests/test_analysis.py -q
```

The analysis reads only existing JSON, rejects incomplete/duplicate/malformed or
inconsistent results, and verifies SHA-256 input hashes after generation. It never
loads a language model, fits PCA or reruns task evaluation. It writes:

- `results/tables/`: complete task CSV/Markdown comparisons, long metric table,
  native-baseline changes, smallest-tested-dimension retention summaries, best observed
  scores, provenance, validation hashes and descriptive empirical findings.
- `results/figures/`: eight observed-score/relative-change plots in PDF, SVG and
  300-dpi PNG. Cross-model dimension fractions make compression levels comparable.

See [analysis usage](docs/analysis_usage.md) for formulas and options. The default
95% retention threshold is descriptive, not statistical equivalence. Aggregate
single-seed scores do not support confidence intervals or significance claims.
Generated tables and figure formats are eligible for Git tracking; raw JSON remains
unchanged. Regeneration overwrites derived artifacts only.

## Test dependencies and checks

For all offline data/preparation tests in addition to analysis tests:

```bash
python -m pip install -r requirements-test-offline.txt
python -m pytest -q
```

This environment excludes PyTorch and Transformers, so tests requiring them skip,
including the tiny random-model inference test. No pretrained model is downloaded.
The full collection can be run safely this way during submission preparation.
For experiment-development tests requiring those libraries, `requirements-dev.txt`
includes the full experiment dependencies; the tiny-model test does perform inference.
Do not run it when model execution is prohibited. No GPU compatibility is established
by offline tests or configuration-only dry runs.

## Preserved preparation and execution code

The final results already exist. Commands here describe the implementation for
future reproduction; **do not rerun experiments during submission preparation**.
Full historical reproduction additionally requires the missing original data exports,
calibration/PCA artifacts, predictions and environment export. Repository dependency
pins are supported environment specifications, not proof of the historical versions.
CUDA-capable hardware and any required checkpoint access/license acceptance are needed.
`requirements.txt` specifies experiment dependencies including PyTorch;
`requirements-colab.txt` preserves Colab's existing CUDA PyTorch and aligns the
non-torch dependencies, including bitsandbytes.

See [task data documentation](docs/task_data.md) for bundle schemas and preparation
commands. Use `data/shared_calibration.json` consistently; the suite expects that name.
For configuration-only inspection with already prepared exports:

```bash
python run_experiment.py --model Phi4-mini --task Banking77 --data data/banking77_official.json --calibration data/shared_calibration.json --dry-run
```

`run_final_suite.py` is the final matrix runner. It loads one frozen model once,
loads/fits its shared PCA, and evaluates all four tasks. A future run using
`--data-dir data --output-root artifacts/new_suite` writes task outputs under
`artifacts/new_suite/results/<model>/<task>/`, separate from the submitted
`results/raw/` archive. It also writes PCA artifacts and suite progress. Its current
resume checks cover dimension presence, not complete metric/provenance validation.
Arxiv checkpoints also have limited identity validation. These limitations and a
non-3072 cache-width issue in `run_experiment.py` remain documented in the
[audit](docs/submission_audit.md); experiment code has not been changed in cleanup.

## Project structure and historical code

```text
README.md
requirements*.txt              analysis, offline tests, and experiment environments
analyze_results.py             offline validation/tables/figures from raw JSON
prepare_*.py                   data and shared calibration exporters
run_final_suite.py             final five-model/four-task suite
run_experiment.py              individual task runner (see known limitations)
run_pilot.py                   historical Phi/STSB pilot entry point
smoke_test_models.py            preserved model loading/batching checks
pilot/                         shared final implementation plus pilot compatibility
  config.py, task_config.py    model matrix and task protocols
  model.py, core.py            frozen extraction, normalization, PCA, native cache
  pca_artifact.py              per-model PCA persistence
  evaluation.py, task_runner.py downstream evaluation and output export
  arxiv.py, banking77.py,
  scifact.py, task_data.py     official bundles and validation
results/
  raw/<model>/<task>/results.json  immutable final source results
  tables/                     comparison, change, retention and provenance artifacts
  figures/                    publication figures: PDF, SVG, PNG
tests/                         preserved offline tests plus final analysis tests
docs/                          data protocols, analysis instructions and audit
```

The `pilot/` package also contains the final implementation; its directory name does
not make the final matrix a pilot. `run_pilot.py`, compatibility APIs and their tests
are retained as historical code. The old Phi/STSB 300-pair pilot output itself is
not included in this checkout and is not part of the 84 final configurations.

## Submission limitations

The archive validates final scores and recorded provenance; it cannot reconstruct
missing original artifact bytes, verify calibration exclusions independently,
audit historical checkpoint reuse, recover actual dtype/package versions, or provide
uncertainty estimates. STSB is a validation evaluation, not a test-set result.
Read the [submission audit](docs/submission_audit.md) before interpreting the results.
The academic report is intentionally deferred.
