# Embedding Dimensionality and NLP Performance

This project evaluates frozen pretrained text representations at different output
widths using training-only PCA and task-specific evaluation. The final deliverables
will be reproducible code and an academic PDF report. No LLM fine-tuning is used.

## Execution status

**Phi4-mini × English STSB: all four required dimensions executed successfully**
in Google Colab on a Tesla T4 (~15 GB VRAM), PyTorch `2.11.0+cu128`, CUDA available.
The owner reported 34 passing tests in that environment before this extension.
The executed configuration used 2,000 PCA training sentences, **300 validation
pairs**, seed 42, batch size 4 and a 256-token limit. This is a subset experiment,
not a full STSB benchmark evaluation.

| Dimension | Representation | Cosine Spearman |
|---:|---|---:|
| 3072 | Native | 0.3620005218131407 |
| 1536 | PCA | 0.342132 |
| 768 | PCA | 0.330040 |
| 384 | PCA | 0.315128 |

These are the owner's executed four-dimension results, quoted at the supplied
precision. Artifacts under `results/phi_stsb_4dims/` were backed up to Google Drive;
they are not reconstructed here. Earlier three-dimension pilot scores belong to a
separate run. A 1536-component randomized PCA fit can produce slightly different
768/384 scores from a 768-component fit; do not splice the two runs.

**SciFact, Banking77 and Arxiv-Clustering:** evaluation infrastructure implemented
and tested offline on synthetic data; **no real datasets/models evaluated** for
these tasks. Verified dataset bundles are still required before execution. Other
models, a full benchmark sweep, and the final PDF report remain future work.

## Representation and PCA

- Frozen `microsoft/Phi-4-mini-instruct`, final normalized hidden state at the
  **last attention-mask-valid token**. Plain text; tokenizer-default special tokens,
  no chat template or added EOS. Existing extraction/batching checks are preserved.
- Existing BF16-if-supported / FP16-otherwise loading; no quantization or fine-tuning.
- Native float32 vectors are cached independently of requested output dimensions.
- Fit randomized PCA on L2-normalized **training/reference vectors only**, seed 42,
  centering enabled, no whitening. Fit the largest requested reduced width once;
  smaller widths take prefixes of the ordered PCA basis. Normalize projected vectors.
  The native baseline is normalized without centering. PCA changes centering as well
  as width and does not reduce backbone VRAM or encoding cost.
- For Banking77/arXiv, sample 2,000 reference/train rows deterministically for PCA; train
  the classifier or clusterer on the entire designated fit partition. Banking77 PCA
  samples distinct training texts, while its classifier retains all 10,003 train rows. PCA requires
  more rows than components and sufficient effective rank (checked at runtime).
- Reject canonical text overlap between fit and evaluation inputs, including all
  retrieval queries and corpus documents. Dataset provenance also requires human
  review: text checks cannot establish split authenticity or detect paraphrases.

## Task protocols

All dimensions use identical texts, splits, seeds and downstream settings. Scores
are reported on their native scales, not multiplied by 100. Different task metrics
must not be averaged as if they were interchangeable.

| Task | Protocol | Primary / additional metrics |
|---|---|---|
| STSB | Unchanged pilot: 300 validation pairs; cosine similarity; 2,000 train sentences with overlap exclusion against **all validation/test sentences** | Cosine Spearman [-1, 1] |
| SciFact | Exact cosine ranking over the complete supplied corpus; same frozen PCA transform for queries and documents; macro average over supplied evaluation queries | nDCG@10 / Recall@10, Recall@100, MRR@10 |
| Banking77 | L2-normalized vectors; logistic regression fitted only on designated train; predict all 3,080 official test rows | Accuracy / macro-F1 |
| Arxiv-Clustering | **Inductive** MiniBatchKMeans: fit reference vectors, freeze centroids, predict held-out vectors; one explicitly selected subset/variant per run | V-measure / adjusted Rand, NMI |

**Retrieval:** linear relevance gain with log2 discount; unjudged documents count as
zero relevance; positive grade means relevant for recall/MRR. Every selected query
must have a positive judgment. Ties use ascending document ID. Query IDs, corpus
IDs and qrels remain separate; query/corpus fitting is forbidden. Corpus text is
`title + "\n" + abstract`; queries remain claim text. Evaluation uses the supplied
corpus; do not silently restrict it to relevant documents. Results from corpus
subsets must be labelled as subsets. SciFact PCA uses all eligible unique `claim` texts from `allenai/scifact`, config
`claims`, train + validation only. The 1,711 source rows may contain repeats and MTEB
test-query overlaps; these are excluded and counted. No 2,000-row quota applies.

**Classification:** verified dataset `PolyAI/banking77`, fields `text` and `label`,
77 intents, official train (10,003 rows) for all fitting and official test (3,080 rows)
for evaluation only. scikit-learn logistic regression, `C=1`, `solver=lbfgs`,
`max_iter=1000`, `tol=1e-4`, no class weighting, fixed seed. No evaluation-based
hyperparameter tuning; convergence warnings stop execution. Macro-F1 uses the fixed
training label universe (all 77 for full Banking77), zero for undefined class scores.
This is a full-designated-training protocol, not MTEB's few-shot classification score.

**Clustering:** `init=k-means++`, `n_init=10`, `max_iter=100`, `batch_size=1024`,
`tol=0`, `max_no_improvement=10`, `reassignment_ratio=0.01`, fixed seed. Supply `K`
from a predefined taxonomy or a training-only decision, **never evaluation labels**.
NMI uses arithmetic averaging. No evaluation examples fit centroids. This differs
from conventional/MTEB clustering that fits clusters on the set being scored, so
scores must be labelled **custom inductive clustering**, not standard MTEB results.
Arxiv S2S/P2P, revision, subset, reference source and K remain unverified. Do not
repurpose an official test partition as training. The current runner evaluates one
subset per invocation; multi-subset aggregation is future work.

The project does not use the MTEB runner. It preserves the working STSB data/metric,
uses standard task metrics, and records explicit custom protocols. See
[task bundle specification](docs/task_data.md) for the required data contract and
remaining source verification. SciFact uses verified `mteb/scifact`; its official test export and separate AllenAI claim
reference are documented in the bundle specification. ArXiv IDs remain unset.
Banking77 has a dedicated data-only preparation command; it discovers the configuration and pins the resolved
dataset commit before loading. No configuration is guessed.

## Required experiment matrix

Each model is required on each of the four tasks: **84 model/task/dimension
configurations**, before repetitions. Only Phi has an executable model adapter.

| Model | Native width | Required widths | Status |
|---|---:|---|---|
| Qwen3-8B | 4096 | 4096, 2048, 1024, 512, 256 | ID/revision/loading unverified |
| Gemma3-4B | 2560 | 2560, 1280, 640, 320 | ID/revision/text adapter unverified |
| Llama3.2-3B | 3072 | 3072, 1536, 768, 384 | ID/revision/loading unverified |
| Phi4-mini | 3072 | 3072, 1536, 768, 384 | STSB subset executed; three task evaluators offline-tested |
| Mistral-7B | 4096 | 4096, 2048, 1024, 512 | Version/ID/loading unverified |

## Installation and hardware

Preserve the existing working Colab environment. Do not install `requirements.txt`
or `requirements-dev.txt` there: their standalone PyTorch pin differs from the
verified Colab version. For a **fresh** Colab with compatible CUDA-enabled PyTorch:

```python
%cd /content/NLP-Embedding-Dimension-Analysis
%pip install -r requirements-colab.txt
```

This pins non-PyTorch dependencies; it is not a complete lock of the historical
Colab runtime. Check `python -m pip check` after any installation. Standalone local
validation uses Python 3.11/3.12 and the original requirements/dev files. No new
packages are needed for this extension.

Phi is verified on the T4 for the stated STSB workload. The unchanged loader refuses
CPU execution and requires >=10 GiB free GPU memory. Use batch 4 and max length 256;
new-task runtime/peak memory are not yet measured. Long arXiv texts may be truncated;
counts are recorded. Embeddings/PCA/downstream fitting also require host RAM; the
current implementation holds each task's native vectors in memory. Estimate
`4 × number_of_texts × 3072` bytes for raw vectors, plus PCA/normalized copies and
working memory; large arXiv variants need a separate capacity review.

The explicit loading strategy remains `native-auto-16bit`. 7B/8B weights plus
activations may exceed ~15 GB T4 VRAM. No automatic quantization/offload changes the
methodology. Larger GPUs or explicitly reviewed loading strategies are later work.

## Offline validation and next Colab commands

Transfer/review the changes in the existing checkout first. These commands do not
load Phi, download datasets, or run GPU inference:

```python
%cd /content/NLP-Embedding-Dimension-Analysis
!HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q
!python run_pilot.py --dry-run --dimensions 3072 1536 768 384
!python run_experiment.py --task STSB --dry-run
!python run_experiment.py --task SciFact --dry-run
!python run_experiment.py --task Banking77 --dry-run
!python run_experiment.py --task Arxiv-Clustering --dry-run
```

Dry-run uses only the standard library. For new tasks, a valid configuration without
data prints `executable: false` with actionable blockers and exits successfully.
This is a configuration inspection, **not** evidence that a dataset is ready.
Invalid supplied bundles/configurations fail. Once reviewed bundles exist, add
`--data data/<bundle>.json`; arXiv additionally needs `--clusters <verified-K>`.
No arbitrary K or dataset variant is supplied as a default. Actual execution rejects
blockers before importing the model loader. Dry-run cannot verify GPU fit, PCA rank,
dataset authenticity or scientific suitability of the reference pool.

## Banking77 data preparation (CPU only, no language model)

The official train/test protocol is verified; real Banking77 experiments are still
**unexecuted**. In the existing Colab environment, after transferring these changes:

```python
!python prepare_banking77.py --dry-run
!python prepare_banking77.py --output data/banking77_official.json
!python run_experiment.py --task Banking77 --data data/banking77_official.json --dimensions 3072 1536 768 384 --train-sentences 2000 --seed 42 --dry-run
```

Only the second command accesses Hugging Face and downloads **dataset files**, not
models. It resolves `main` to an immutable commit SHA, discovers the sole available
configuration (or requires an explicit `--configuration` if ambiguous), and records
both. Reproduce that export with `--revision <recorded-SHA>` and a fresh output path.
The preparation dry-run needs only the standard library and does not verify remote
contents. Do not set Hugging Face offline environment variables for the actual
preparation command. Existing dependency files suffice; no new packages are added.

Preparation checks exactly 10,003 train / 3,080 test rows, fields `text`/`label`,
matching 77-class label definitions, label coverage, and train/test text separation.
Texts and row order are preserved; split-prefixed IDs and content hashes are saved.
Duplicate same-label training rows are retained for classification; PCA samples
only distinct training texts. Normalized train/test overlap, conflicting duplicate
training labels, or unexpected source structure **stops preparation** without silently
filtering official rows. Such a failure requires review before running an experiment.
The command never overwrites an existing bundle; bundles live under ignored `data/`.

The runner accepts only the full official Banking77 contract, not arbitrary subsets
or validation splits. Classifier settings and data remain identical across dimensions.
Accuracy is primary; macro-F1 is secondary. The default 2,000-row PCA calibration
sample is entirely training-derived. No test data fit PCA or classifier parameters.

## SciFact data preparation (no model execution)

Retrieval stays `mteb/scifact` official test: 5,183 documents, 300 test queries,
339 judgments; existing nDCG@10/Recall@10/Recall@100/MRR@10 metrics are unchanged.
PCA uses `allenai/scifact`, config `claims`, **train + validation only**, reading
`claim` texts. Both dataset revisions and their provenance are recorded separately.
AllenAI test is never requested or iterated.

```python
!python prepare_scifact.py --dry-run
!python prepare_scifact.py --output data/scifact_allenai_reference.json
!python run_experiment.py --task SciFact --data data/scifact_allenai_reference.json --dimensions 3072 1536 768 384 --dry-run
```

The 1,261 + 450 source rows do not guarantee 1,711 unique eligible claims:
[AllenAI's loader](https://huggingface.co/datasets/allenai/scifact/blob/main/scifact.py)
repeats claims across evidence annotations. Preparation deduplicates texts and
excludes canonical matches to MTEB test queries and corpus documents, recording
all exclusions. Every remaining eligible claim is used for PCA in source order;
`--train-sentences` does not subsample SciFact. No test qrels fit any component.

The full-dimension dry-run passes the sample-count check only with **at least
1,537 eligible texts**, because centered 1,536-component PCA needs rank 1,536.
Numerical rank is checked later during PCA fitting. The code does not claim that
the real source satisfies this condition before preparation. It does not add
corpus/test texts or repeat rows to reach the required count.

Only preparation downloads dataset data/code. The AllenAI loading script is resolved
to a commit and explicitly trusted; streaming iterates train and validation only.
Use the current script-compatible datasets environment. Its upstream archive is
mutable, so retain the exact exported bundle and recorded input hashes as well as
both revisions. See the bundle specification for details. Real preparation and
SciFact experiments remain **unexecuted**.

## Execution interface and compatibility

`run_pilot.py` is unchanged: its default remains 3072/768/384 and its original
outputs/cache keys remain intact. The new entry point defaults to all four required
Phi dimensions. For STSB it delegates to the existing pilot without changing its
sampling, PCA, scoring, or original result files. For example, **only when a future
GPU execution is approved**, a fresh STSB run would use:

```bash
python run_experiment.py --task STSB --dimensions 3072 1536 768 384 --train-sentences 2000 --validation-pairs 300 --batch-size 4 --max-length 256 --seed 42 --output results/phi_stsb_new_run
```

New tasks use the same interface with `--data` (and arXiv `--clusters`). Use a fresh
output directory; nonempty directories are rejected. Existing backed-up results
must not be overwritten. This update performs no real runs.

## Outputs and reproducibility

New-task runs write `metrics.csv`, `results.json`, `predictions.json`, and `pca.npz`
when reduction is requested. STSB through the new entry point adds `metrics.csv`
to its usual `scores.csv`, JSONs, PCA and plot. Historical files are not converted
or edited. Common metric rows contain:

`model, task, dataset, dataset_revision, split, dimension, reduction, metric, score,
seed, protocol, n_eval`.

Metadata records the full plan/protocol, dataset and reference provenance, bundle
hash, fit/evaluation IDs, PCA calibration IDs/text hashes, representation/revision,
package versions, hardware, seed, cache hits, truncation counts, elapsed time and peak
GPU allocation. Predictions preserve evaluation ID alignment; retrieval saves the
top 100 ranked document IDs and cosine scores per query. Keep the exact input bundle
with the run to recover labels/qrels; its hash is recorded. Seeded PCA, fitting and
sampling are reproducible within a compatible environment, not necessarily bitwise
identical across hardware/library versions. Save `python -m pip freeze` and back up
real outputs and source bundles. Generated data/results/caches remain Git-ignored.

Pinned existing revisions:

- Phi: `cfbefacb99257ffa30c83adab238a50856ac3083`
- English STSB: `96943a16ea6a35129e253c659081cb59daf81b30`

## Architecture

- `run_pilot.py`: unchanged backward-compatible STSB entry point.
- `run_experiment.py`: task-aware CLI, offline planning, legacy delegation, Phi orchestration.
- `pilot/config.py`: assignment model/task registry; explicit loading policy.
- `pilot/task_config.py`: task protocols and readiness checks.
- `pilot/task_data.py`: local bundle loading, provenance and leakage validation.
- `pilot/banking77.py`, `prepare_banking77.py`: official Banking77 contract and data-only export.
- `pilot/scifact.py`, `prepare_scifact.py`: official SciFact retrieval export and AllenAI non-test claim reference.
- `pilot/task_runner.py`: shared native extraction -> training-only PCA -> task evaluation -> common exports.
- `pilot/evaluation.py`: retrieval, classifier and inductive clustering evaluators.
- `pilot/core.py`, `pilot/model.py`: unchanged verified PCA/cache/extraction utilities.
- `tests/`: existing regressions plus synthetic task/data/runner tests.

Later model adapters can supply native vectors to the shared engine, after checkpoint,
representation and memory validation. Real SciFact reference eligibility/rank validation, the arXiv dataset exporter, MTEB integration, additional
model loaders, multi-subset arXiv aggregation, actual new-task runs and the report
remain outside this step.

## References

- [Phi model](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [English STS dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [BEIR retrieval evaluation](https://github.com/beir-cellar/beir)
- [Logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
- [MTEB clustering tasks](https://docs.mteb.org/overview/available_tasks/clustering/)
