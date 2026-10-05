# Submission audit and cleanup status

The first-pass observations below are retained as an audit record. Safe submission
cleanup has now corrected README and `docs/task_data.md`, aligned calibration paths
and task protocols, added the missing Colab bitsandbytes requirement using the exact
existing experiment constraint, added a separate offline-test requirements file,
and allowed narrow tracking of derived tables/figures. No experiment implementation,
raw results, historical code or existing tests were modified. No files were deleted.
References below to the old README or ignore rules describe the inspected snapshot,
not their corrected current state. Implementation/provenance proposals remain open.

Source inspected: `assignment-completion`, HEAD
`88c9536f04dda2194c6cf20d5388588b4298171e`. All tracked repository files and all
20 raw JSON files were read before changes. No experimental outputs were edited.
This document records methodology and proposed follow-up work, not the academic report.

## What the implementation does

`pilot/model.py` loads pretrained Hugging Face backbones, calls `.eval()` and
`requires_grad_(False)`, and uses `torch.inference_mode()`. Plain input texts are
tokenized with right padding and truncation at the recorded maximum length (256).
Representations are final hidden states at the last attention-mask-valid token,
converted to float32. They are contextual sentence representations, not the internal
token embedding table. No LM fine-tuning or token embedding retraining occurs.

`prepare_calibration.py` deterministically ranks deduplicated Banking77 train texts
and STSB train sentences, selecting 3,072 records. It excludes Banking77 test and
STSB validation/test texts; SciFact query/corpus and Arxiv text exclusions are applied
when their optional bundle paths are supplied. The README command omits these paths.
The archived calibration bundle is needed to verify which exclusions occurred in the
historical run; existing result hashes alone do not establish that fact independently.

`pilot/core.py` L2-normalizes calibration vectors and fits centered, unwhitened,
randomized PCA with seed 42 at each model's largest reduced dimension. Evaluation
vectors are L2-normalized, centered using that fitted mean, projected into prefixes
of the same ordered basis, and normalized again. Native vectors are normalized
without PCA centering. Thus observed gains cannot be attributed solely to fewer
dimensions. `pilot/pca_artifact.py` persists a model/calibration/revision-specific
basis; `run_final_suite.py` loads each frozen model once and passes one PCA to all
four tasks. Recorded artifact paths and calibration hashes are consistent with this.

The sequence is **frozen pretrained native representation → fitted PCA projection
where requested → downstream evaluation**. There are five pretrained models and
84 evaluated model/task/dimension configurations, not 84 trained LMs. Banking77 fits
a separate logistic-regression classifier per dimension on the official 10,003 train
rows and evaluates 3,080 test rows. SciFact ranks exactly by cosine similarity with
lexical document-ID tie-breaking and linear relevance gains, evaluating 300 queries.
STSB evaluates all 1,500 **validation** pairs. Arxiv fits MiniBatchKMeans on each
official test set with K from its label taxonomy, then macro-averages across 30
non-degenerate sets (707,723 sentence occurrences); `test:29` is excluded.
Clustering fits centroids on test-set vectors under its benchmark protocol; PCA
does not fit on those vectors. Do not conflate those two fitting operations.

Qwen and Mistral use NF4 four-bit weight loading with double quantization. Other
models use float16 or bfloat16 selected by GPU capability; the final metadata does
not record actual dtype. Most result files record NVIDIA L4; Phi STSB records Tesla
T4. `t4-auto` is a loading-policy identifier, not proof all results used a T4.

## Documentation findings — corrected during cleanup

1. Rewrite `docs/task_data.md`: it incorrectly says Arxiv preparation is unimplemented,
   describes obsolete inductive/reference clustering, recommends rejected `--clusters`,
   treats the old SciFact reference as PCA input, and says final tasks remain unexecuted.
   Several CLI examples omit required `--model` and shared `--calibration` arguments.
2. Update README with verified final-result inventory, actual STSB validation split,
   Arxiv exclusions/aggregation, mixed T4/L4 hardware, native-versus-centered-PCA geometry,
   analysis command and links. Clarify that the historical pilot output is not present
   in this checkout even though its runner and tests remain.
3. Align calibration naming: README uses `pca_calibration.json`, while final suite
   requires `shared_calibration.json`. Document final suite's `output_root/results/`
   layout versus the submitted `results/raw/` archive and its resume behavior.
4. Show optional SciFact/Arxiv bundle arguments in the calibration preparation example;
   distinguish implemented exclusion policy from independently verified historical data.

## Remaining reproducibility and code proposals

1. Archive existing calibration/data exports, PCA artifacts, predictions/per-set
   checkpoints, environment export and suite progress outside Git if large. Preserve
   their actual hashes; do not regenerate experiments. Raw JSON lacks historical package
   versions and actual model dtype, so requirements pins are not a verified run lockfile.
2. The missing Colab bitsandbytes dependency is now corrected without changing the
   existing version range. README now distinguishes environments and checkpoint access.
   Preserve the historical environment separately from the new analysis environment;
   it still cannot be recovered from the submitted aggregate JSON.
3. Strengthen `run_final_suite.complete`: currently it checks dimension coverage only,
   so incomplete metric sets or conflicting provenance can count as completed. Reuse
   explicit metric/provenance checks in future execution code without rerunning it now.
4. Strengthen Arxiv checkpoint identity: reuse checks only set ID and dimensions,
   not model/revision/bundle/PCA hash/seed. This is a potential stale-checkpoint risk,
   not proof that these final results are wrong. Original checkpoints are needed to audit it.
5. `run_experiment.py` calls native cache without the selected native width, relying on
   the cache's default 3072. This affects non-3072 models in that entry point; final suite
   passes native width explicitly. Fix and test in a separate implementation pass.
6. Improve PCA artifact finite-value checks and checksum validation; recorded path equality
   cannot prove the underlying basis bytes were identical across sessions.
7. Reformat dense one-line scripts/comments after methodological corrections. Consider a
   clearer module name than `pilot/`, retaining import compatibility if renamed. Harmonize
   `Arxiv-Clustering` assignment naming with `ArxivClusteringS2S` benchmark naming.

## Proposed repository hygiene

The tracked tree contains no obvious temporary weights, caches, credentials or dead
experiment output duplicates. Retain historical pilot code until its archival purpose
is decided; do not label it dead based only on its age. No deletions are proposed now.
The existing `.gitignore` appropriately excludes data, models, caches and credentials,
but `results/*` also ignores all generated tables/figures. Add narrow exceptions for
submission tables and vector/PNG figures, or document intentional forced inclusion.
Review license/citation and data/model attribution requirements for submission.
The first pass left existing files untouched. Subsequent safe cleanup updated only
documentation, Colab requirements and ignore rules. Historical code and experimental
results remain intact. License/citation review and missing original-artifact recovery
remain submission considerations; no provenance was fabricated.
