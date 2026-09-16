# Embedding Dimensionality and NLP Performance

This project evaluates frozen pretrained text representations at different output
widths using training-only PCA and task-appropriate evaluation. The final deliverables
will be reproducible code and an academic PDF report. No LLM fine-tuning is used.

## Completed pilot

The original Phi-4-mini / English STS pilot **successfully ran in Google Colab**:
NVIDIA Tesla T4 (~15 GB VRAM), PyTorch `2.11.0+cu128`, CUDA available, and 9 tests passed.
It used 2,000 PCA training sentences, 300 validation pairs, and seed 42.

| Dimension | Representation | Cosine Spearman |
|---:|---|---:|
| 3072 | Native | 0.3620005218131407 |
| 768 | PCA | 0.33006447482450463 |
| 384 | PCA | 0.31506961347171786 |

These are actual results reported by the project owner. The original `scores.csv`,
`results.json`, `predictions.json`, `pca.npz`, and plot were generated and backed up
in Google Drive. They were not present in this local checkout during this update;
no result artifacts have been reconstructed or independently verified here.

**New, not yet GPU-executed:** configurable dimensions including the missing **1536**.
The other models/tasks remain planned, not implemented or evaluated. The final report
has not been generated. This validation-subset pilot does not complete the full STS task.

## Methodology (preserved)

- Model: `microsoft/Phi-4-mini-instruct`; dataset: English `mteb/stsbenchmark-sts`.
- Freeze the backbone. Extract its final normalized hidden state at the **last
  attention-mask-valid token**. Plain text, tokenizer-default special tokens, no
  chat template or added EOS. The pinned tokenizer disables automatic BOS/EOS.
- Use the existing BF16-if-supported / FP16-otherwise loading policy. No quantization.
- Deduplicate training texts and exclude normalized matches against **all validation
  and test sentences**. Test labels never fit PCA or enter pilot scoring.
- Cache native float32 vectors. Fit randomized PCA on L2-normalized training
  embeddings only, with centering, no whitening, and seed 42. Fit enough components
  for the largest requested reduced width; smaller widths use prefixes of this PCA
  basis, not prefixes of the original hidden coordinates.
- Apply the frozen transform to evaluation vectors and L2-normalize again. The
  native baseline is L2-normalized without centering. Evaluate cosine Spearman.

The evaluator uses MTEB's English STS dataset and cosine-Spearman metric directly,
not the full MTEB runner. Exact/normalized overlap exclusion does not detect every
semantic paraphrase. PCA changes centering as well as width, and does not reduce
backbone VRAM or encoding computation.

### Backward compatibility

`python run_pilot.py` retains the original **3072/768/384** defaults and 768-component
PCA. Existing filenames, extraction settings, sampling, and Phi native-cache keys
are preserved. Pass `--dimensions 3072 1536 768 384` for all required Phi widths.

The four-width run fits **1536** PCA components instead of 768. Because the existing
solver is randomized, its 768/384 outputs may differ slightly from the original run.
Compare each complete run using its own metadata; do not overwrite or splice the
historical results. At least 1537 eligible calibration sentences are required;
2,000 remains the default. Runtime checks additionally enforce sufficient rank.

## Required experiment matrix

Every row is required on **each of the four tasks**: 84 model/dimension/task
configurations before repetitions. Metadata lives in `pilot/config.py`.

| Model | Native width | Required widths | Checkpoint/implementation status |
|---|---:|---|---|
| Qwen3-8B | 4096 | 4096, 2048, 1024, 512, 256 | ID/revision/loading adapter unverified |
| Gemma3-4B | 2560 | 2560, 1280, 640, 320 | ID/revision/text adapter unverified |
| Llama3.2-3B | 3072 | 3072, 1536, 768, 384 | ID/revision/loading adapter unverified |
| Phi4-mini | 3072 | 3072, 1536, 768, 384 | Existing checkpoint verified in pilot; 1536 unexecuted |
| Mistral-7B | 4096 | 4096, 2048, 1024, 512 | Version/ID/loading adapter unverified |

| Task | Assignment dataset | Implementation status |
|---|---|---|
| Retrieval | SciFact | Future; evaluator/revision/reference-data policy pending |
| Classification | Banking77 | Future; classifier protocol/version pending |
| Clustering | Arxiv-Clustering | Future; S2S/P2P/version/reference-data policy unresolved |
| STS | STSB (English STS Benchmark) | Phi validation-subset pilot only |

Unverified Hugging Face IDs are `None`, not guesses. Metadata-only dry-runs return
`executable: false` and blockers for future combinations. Execution rejects them
before importing model libraries or accessing the network.

## Hardware and loading strategy

The existing Phi pilot is verified on a Colab T4 with ~15 GB VRAM. The runner keeps
its >=10 GiB free-memory guard and refuses CPU model execution. Use the verified
batch size (default 4), 256-token limit, and the original CUDA-enabled environment.
The new PCA fit uses more CPU time/RAM; 32 GB host RAM provides comfortable headroom,
although actual usage has not been measured for this extension.

The explicit current strategy is `native-auto-16bit`. Future model loaders must
implement and record their strategy; unknown strategies fail validation. A 7B/8B
model may approach/exceed T4 VRAM with 16-bit weights alone, before activations;
naive FP32 loading is even larger. There is **no automatic quantization/offload**.
Options needing later approval/testing are a larger GPU, explicit CPU offload
(with throughput implications), or separately labelled quantization experiments.
Checkpoint access/licensing, text-backbone selection, attention implementation,
batch size, and peak VRAM still need validation for each new model.

## Installation

Python 3.11/3.12 was used for local validation. The original standalone environment
remains specified in `requirements.txt` (PyTorch 2.7.1) and `requirements-dev.txt`.
It is not the same as the reported successful Colab PyTorch 2.11.0 environment.

**Existing working Colab:** preserve installed dependencies. Do not install
`requirements.txt` or `requirements-dev.txt` there, since they would request a
PyTorch downgrade. Recover original non-PyTorch package versions from the saved
`results.json`/environment export for exact historical reproduction.

**Fresh Colab only**, with an already compatible CUDA-enabled PyTorch:

```python
%cd /content/NLP-Embedding-Dimension-Analysis
%pip install -r requirements-colab.txt
```

This installs the project's non-PyTorch pins and pytest while keeping the current
PyTorch. It is a setup candidate, not an exact lock of the historical Colab runtime.
Restart the runtime if requested after installation, then check `python -m pip check`.

## Offline validation (no models or inference)

From the repository root:

```bash
python -m pytest -q
python run_pilot.py --list-matrix
python run_pilot.py --dry-run --dimensions 3072 1536 768 384
python run_pilot.py --dry-run --model Qwen3-8B --task SciFact --train-sentences 3000
```

Dry-run uses only the standard library. It validates widths, duplicate dimensions,
PCA sample-count bounds and loading strategy; it does not verify actual dataset
availability/rank or GPU feasibility. Offline tests use tiny arrays, test doubles,
and a tiny randomly initialized architecture, never pretrained weights. Temporary
fixtures are not experimental results.

## Next Colab run (after transferring these local changes)

In the existing verified environment, first run:

```python
%cd /content/NLP-Embedding-Dimension-Analysis
!python -m pytest -q
!python run_pilot.py --dry-run --dimensions 3072 1536 768 384
```

Then, when ready to execute remotely:

```python
!python run_pilot.py --check-hardware
!python run_pilot.py --dimensions 3072 1536 768 384 --train-sentences 2000 --validation-pairs 300 --batch-size 4 --max-length 256 --seed 42 --output results/phi_stsb_4dims
```

Use a fresh output directory. No push or remote run is performed by this update.
New metadata records requested widths, fitted PCA width, and loading strategy.

## Outputs and reproducibility

Outputs retain `scores.csv`, `results.json`, `predictions.json`, `pca.npz`, and
`dimension_vs_spearman.png`; CSV rows and plot ticks follow the requested widths.
Native-only runs skip PCA fitting and `pca.npz`. Native caches remain under
`cache/native/`; dimension changes do not invalidate an otherwise matching cache.
A rerun still initializes the encoder and validates batching, even with cache hits.

Pinned revisions remain:

- Model: `cfbefacb99257ffa30c83adab238a50856ac3083`
- Dataset: `96943a16ea6a35129e253c659081cb59daf81b30`

Results include resolved revisions, calibration hashes, validation indices, package
versions, hardware, and truncation counts. Save `python -m pip freeze` alongside
real results and back them up. Numerical identity across environments is not guaranteed.
Generated artifacts and credentials remain ignored by Git.

## Architecture and next boundaries

- `run_pilot.py`: backward-compatible Phi/STSB runner, CLI validation, exports/plot.
- `pilot/config.py`: assignment registry, explicit loading policy, offline plan validation.
- `pilot/model.py`: unchanged verified Phi loader/pooling; future adapters remain separate.
- `pilot/core.py`: shared validation, native cache (configurable width), PCA, scoring.
- `tests/`: offline regression and configuration tests.

Later work can add model adapters returning native vectors and task evaluators
consuming transformed vectors. The runner should orchestrate **reference extraction
-> fit reducer -> freeze reducer -> evaluation**. Retrieval queries/documents must
share one fitted transform. Each new task needs an explicit training/reference
manifest; none may silently fit PCA on evaluation data. MTEB adapters, additional
loaders, and a general experiment runner are future work, not implemented stubs.

## References

- [Phi model](https://huggingface.co/microsoft/Phi-4-mini-instruct)
- [English STS dataset](https://huggingface.co/datasets/mteb/stsbenchmark-sts)
- [MTEB task definitions](https://docs.mteb.org/overview/available_tasks/semantic-similarity/)
- [PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
