# Final task data and shared calibration

This documents the current final implementation, replacing obsolete task-specific
PCA and inductive-Arxiv plans. Final results for all five models and four tasks
are already present under `results/raw/`; no preparation or experiment is needed
for submission analysis. The following commands are for future reproduction only.

## Bundle preparation

```bash
python prepare_banking77.py --output data/banking77_official.json
python prepare_scifact.py --output data/scifact_official.json
python prepare_arxiv.py --output data/arxiv_official.json
python prepare_calibration.py --banking data/banking77_official.json --scifact data/scifact_official.json --arxiv data/arxiv_official.json --output data/shared_calibration.json
```

All three task exporters refuse an existing output. `prepare_calibration.py` does
not have that guard: choose a fresh path and preserve original exports. Task source
revisions are resolved to immutable commits before download and recorded in bundles.
Defaults resolving `main` are not guarantees of reproducing historical bytes; use
recorded revisions and archive exact exported bundles. STSB is loaded separately
by the final runner and evaluated on its full validation split.

Banking77 uses the script-free official Parquet revision
`0b62e79500b84f0fff5a36a4404028f48fbc8621`. SciFact's separate AllenAI reference
loader uses pinned dataset code with `trust_remote_code=True`; its upstream archive
can still be mutable. Its reference fields are retained in the bundle but **are not
used to fit final PCA**. The final PCA input is the shared calibration manifest.

## Schemas and validation

Prepared bundles use `schema_version: 1`; final result metadata uses version 2.
Read the validators in `pilot/task_data.py`, `pilot/banking77.py`,
`pilot/scifact.py` and `pilot/arxiv.py` for authoritative constraints.

| Bundle | Collections and provenance |
|---|---|
| Banking77 | `task: Banking77`; `train` and `evaluation` records with `id`, `text`, `label`; `source` and `fit_source` record official dataset/revision/configuration/splits/counts/content hashes |
| SciFact | `task: SciFact`; `queries`, `corpus`, `qrels`, and legacy `reference`; independent MTEB `source` and AllenAI `fit_source` provenance/content hashes |
| Arxiv | `task: Arxiv-Clustering`; `sets`, each with `id: test:<index>`, `sentences`, `labels`; `source` includes `task_name: ArxivClusteringS2S`, revision/configuration, test split, set count/hash and degenerate sets |
| Calibration | `role: shared_pca_calibration`; `texts` records with ID/text/source; training-source revisions, deterministic selection metadata, exclusion count and `manifest_hash` |

Banking77 preserves official duplicate texts, checks conflicting labels, and records
normalized train/test overlap during validation. This is separate from calibration
selection, which excludes held-out texts. SciFact exports 300 test queries and 5,183
corpus documents; its exact ranking uses only these records and test relevance
judgments. AllenAI reference data does not enter the final PCA or ranking procedure.

Arxiv's exporter is implemented. Each official test set is clustered independently
with MiniBatchKMeans and K equal to the number of unique labels in that set.
There is no custom reference/heldout inductive protocol and `--clusters` is rejected.
The recorded final aggregation is the unweighted mean of 30 non-degenerate sets;
`test:29` is excluded, leaving 707,723 sentence occurrences. Fitting centroids on
these vectors is the implemented clustering evaluation, not PCA fitting.

## Shared calibration and reduction

`prepare_calibration.py` collects Banking77 train texts and both sentences from
STSB train pairs, deduplicates normalized texts, ranks candidates by SHA-256, and
selects 3,072 records. It always excludes Banking77 test and STSB validation/test
texts. Supplying `--banking`, `--scifact`, and `--arxiv` additionally loads task bundles
for evaluation-text exclusions. The historical manifest is not in this repository;
its common hash/count in result JSON cannot independently prove all exclusions.

Each frozen model embeds this same manifest. Normalized native vectors fit one
model-specific centered, unwhitened randomized PCA at the largest reduced dimension.
Smaller dimensions use component prefixes. Native evaluation uses normalization;
PCA evaluation uses normalization, calibration centering, projection and normalization.
No task evaluation examples fit PCA. Identity-checked artifact persistence is in
`pilot/pca_artifact.py`; `run_final_suite.py` supplies the same per-model PCA to all tasks.

## Configuration inspection and paths

With existing bundles, these dry runs check configuration without model execution:

```bash
python run_experiment.py --model Phi4-mini --task Banking77 --data data/banking77_official.json --calibration data/shared_calibration.json --dry-run
python run_experiment.py --model Phi4-mini --task SciFact --data data/scifact_official.json --calibration data/shared_calibration.json --dry-run
python run_experiment.py --model Phi4-mini --task Arxiv-Clustering --data data/arxiv_official.json --calibration data/shared_calibration.json --dry-run
```

The suite expects `shared_calibration.json`, `banking77_official.json`,
`scifact_official.json` and `arxiv_official.json` under `--data-dir`.
`--output-root` receives `results/<model>/<task>/`, `pca/<model>.npz` and suite progress.
Task exports include metrics and result JSON plus predictions; Arxiv can additionally
use per-set checkpoints. These execution outputs are distinct from the submitted
`results/raw/` JSON archive. Never point future execution at that archive.

Native-only plans do not require calibration; reductions do. Sample-count checks
and dry runs do not verify actual embedding rank, GPU compatibility or historical
artifact integrity. Suite resume and Arxiv checkpoint identity checks remain limited;
see [submission audit](submission_audit.md). No historical provenance is inferred or
filled in from current dependencies. To reproduce analysis, use
[analysis usage](analysis_usage.md), which requires none of these dataset exports.
