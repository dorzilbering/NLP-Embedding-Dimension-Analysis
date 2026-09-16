# Local task data bundles (schema version 1)

The new runner reads UTF-8 JSON via `--data`; it does not fetch new-task datasets.
This separates verified dataset preparation from model execution. Banking77 has a dedicated data-only exporter (`prepare_banking77.py`) for the
verified `PolyAI/banking77` official train/test protocol. SciFact now has `prepare_scifact.py` for official test retrieval and separate AllenAI claims
reference data. The arXiv exporter remains unimplemented because its source choices
are unresolved. Once a reviewed bundle is
provided, the Phi extraction and downstream pipeline are executable.

## Required provenance

Every bundle contains `schema_version: 1`, `task` (exact assignment name), and:

| Object | Required fields |
|---|---|
| `source` | `dataset_id`, `revision`, `configuration`, `evaluation_split`, `selection`, `text_format` |
| `fit_source` | `dataset_id`, `revision`, `configuration`, `split`, `role`, `split_method` |

These fields are nonempty strings. Record an immutable dataset commit, release
identifier or archive checksum as `revision`, not `main`, `master`, `latest`, or an
unresolved placeholder. Use an explicit configuration name, or `none` if the source
has none. `selection` is `full` or `subset`; `evaluation_split` is `validation`,
`test`, or explicitly custom `heldout`. `split_method` describes the original split,
any deduplication/exclusions and deterministic selection (including seed).

`fit_source.role` is `train`, `external_reference`, or `derived_reference`.
Training role requires `split: train`. No fitting split may be labelled test, evaluation or heldout. The explicit
SciFact exception uses AllenAI train+validation as a separate reference source,
with MTEB test-query text matches excluded; AllenAI validation is not the retrieval
evaluation split. Other task fitting policies are unchanged. A derived reference is allowed only for an explicitly
custom arXiv reference/heldout protocol (`split: reference`, evaluation `heldout`).
It must come from a non-evaluation source pool; **never divide an official benchmark
test set into training and testing and call that the official benchmark**. Prefer
an independent reference source when there is no official train split. Record the
original source and split procedure so the custom protocol is auditable.

The loader validates declarations and text disjointness, not their authenticity.
Review the original dataset documentation and immutable source before execution.
Keep the exact bundle beside backed-up outputs; results contain its SHA-256 hash.
Use `data/` (Git-ignored) for bundles. Do not put credentials in provenance fields.

## Rows and overlap policy

Rows contain nonempty string `id` and `text`; classification and evaluation
clustering rows additionally contain a string `label`. Convert numeric labels/IDs
to strings consistently. IDs must be unique within each collection; prefix
split-local IDs (for example `train:123` / `test:123`) for classification/clustering.
Never truncate texts during export; the existing tokenizer handles the recorded
max length. Preserve deterministic row ordering.

SciFact/arXiv fit/reference texts must be deduplicated after Unicode NFKC, case
folding and whitespace normalization. Banking77 preserves official training rows;
same-label duplicates remain in classifier training but are deduplicated for PCA
sampling only. Conflicting labels for duplicate training texts cause an error. Any normalized match with supplied evaluation texts
causes an error. For SciFact/arXiv resolve overlaps in the **fitting** partition before export and
record exclusions; never silently remove evaluation examples to improve scores.
For Banking77 any overlap stops preparation: do not remove official train or test
rows without separately agreeing a revised protocol.
Where other official held-out splits exist, exclude their texts from fitting during
preparation too. The runner can only check texts present in the supplied bundle.
Review labels when deduplicating training examples; conflicting labels require an
explicit data-quality decision. Duplicate JSON keys and non-finite qrels are rejected.

## Banking77

Verified dataset: `PolyAI/banking77`. Official fields: `text`, `label`. Official
splits: **train = 10,003**, **test = 3,080**; **77 intent classes**. Use the entire
official train split for logistic regression and only the official test split for
Accuracy (primary) and macro-F1 (secondary). All dimensions use the same classifier
settings. PCA uses a deterministic sample of 2,000 distinct training texts by default;
no test example fits any learned component.

Prepare later in Colab (CPU only; dataset download permitted, no model loading):

```bash
python prepare_banking77.py --dry-run
python prepare_banking77.py --output data/banking77_official.json
python run_experiment.py --task Banking77 --data data/banking77_official.json --dry-run
```

The preparation command resolves the requested `--revision` (default `main`) to a
40-character immutable commit SHA **before** discovering configuration or loading
splits. If exactly one configuration exists it records that name; otherwise it
requires `--configuration` explicitly. A reproduction uses the recorded SHA with
`--revision` and a fresh output file. It does not load remote dataset scripts with
`trust_remote_code`, change dependencies, or switch dataset sources as a fallback.
A loader/source incompatibility must be reviewed if encountered.

Export collections are `train` and `evaluation` (the latter is the official test).
Each row contains `id`, verbatim `text`, and string-encoded numeric `label`. Row IDs
are `train:0` ... `train:10002` and `test:0` ... `test:3079` in original split order.
Both provenance objects identify the same source/revision/configuration. Source
records `selection: full`, `evaluation_split: test`, `text_format: text_verbatim`,
official label names, counts, unique training text count and split content hashes.
Preparation package versions are retained in the bundle.

Validation checks exact counts/schema, matching 77-class label definitions, all
label IDs 0..76, row order/IDs, hashes and canonical train/test text disjointness.
The production runner rejects subsets, validation evaluation, wrong source IDs,
unresolved revisions, missing classes, or altered hashed data. Hashes establish
bundle consistency, not independent proof of source authenticity. Keep the exact
export and revision with future results.

No rows are silently dropped: same-label training duplicates remain in classifier
training; only PCA calibration candidates are deduplicated. Cross-split text overlap
or conflicting duplicate training labels abort preparation and require review.
The official train/test protocol must not be silently changed to make validation
pass. Existing bundles are never overwritten. Real-data preparation and GPU
execution have **not** been performed in this update.

## SciFact

Verified source: `mteb/scifact`. Evaluation is **official test only**. Primary metric
is nDCG@10; secondary metrics remain Recall@10, Recall@100 and MRR@10, with the existing
exact cosine ranking, linear relevance gain and deterministic document-ID tie-break.
No real SciFact preparation or model experiment has been executed here.

The [official dataset metadata](https://huggingface.co/datasets/mteb/scifact/blob/main/README.md)
defines these configurations (all loaded at the **same resolved immutable commit**):

| Configuration | Physical split | Schema | Rows |
|---|---|---|---:|
| corpus | corpus | `_id`, `title`, `text` | 5,183 |
| queries | queries | `_id`, `text` | 1,109 |
| default | train | `query-id`, `corpus-id`, `score` | 919 judgments / 809 queries |
| default | test | `query-id`, `corpus-id`, `score` | 339 judgments / 300 queries |

The query table combines train and test texts. Physical split name `queries` does
**not** imply training permission: logical membership comes from the corresponding
qrels split. Preparation rejects overlapping train/test query IDs, unknown IDs,
duplicate IDs/judgment pairs, incomplete query membership, invalid relevance grades,
unexpected counts/configurations/splits, or queries without positive judgments.

### Export and exact reference source

Retrieval remains unchanged: all 5,183 corpus documents (`title + "\n" + text`),
300 official test queries, and 339 test qrels. Query/corpus IDs and judgments are
preserved. Metrics remain nDCG@10, Recall@10/100, and MRR@10.

PCA reference now comes **only** from `allenai/scifact`, configuration `claims`:

- `train`: 1,261 input rows.
- `validation`: 450 input rows.
- Read only the `claim` text field; evidence labels and annotations are not features.
- Never request or iterate the AllenAI `test` split.
- Process train then validation in source order. Exclude canonical text matches to
  MTEB test queries and the retrieval corpus, then keep the first occurrence of each
  remaining canonical text. Record every exclusion with its row ID, reason and hash.
- Use **all remaining eligible texts** for PCA. SciFact no longer uses a 2,000-row
  quota or random reference subsampling; `--train-sentences` is ignored for SciFact.
  Banking77/STSB and arXiv retain their existing sampling policies.

**1,711 input rows are not a guarantee of 1,711 distinct eligible texts.** The
[official AllenAI loader](https://huggingface.co/datasets/allenai/scifact/blob/main/scifact.py)
repeats claims for evidence annotations. Cross-dataset split names alone also do
not establish separation from MTEB test queries. The exporter measures and reports
actual eligibility instead of claiming that 1,536-component PCA is automatically
valid. Test qrels provide evaluation membership/scoring only; their relevance
grades and relevant-document choices never affect PCA fitting.

PCA keeps the existing centered, non-whitened randomized solver. It requires at
least the requested number of components **and sufficient rank**. Because centering
limits rank to `n - 1`, this implementation requires **more than 1,536 eligible
texts** for a nondegenerate 1,536-component fit, then checks numerical rank at
execution. Exactly 1,536 rows cannot yield 1,536 nonzero centered components.
If all 1,711 rows are eligible and independent, the count check passes and all are
used. If exclusions/deduplication leave too few rows, full-matrix validation fails;
there is no test-data, corpus, duplicate-padding or smaller-dimension fallback.

`fit_source` separately records AllenAI dataset ID, resolved commit SHA, `claims`
configuration, explicit `[train, validation]` splits, `claim` field, raw split
counts, input text hashes, exclusions, eligible count and all-eligible policy.
The MTEB revision/configuration/official counts remain separate in `source`.
Existing bundle hashes and actual PCA-fit ID/hash metadata are retained.

### Preparation and offline validation

```bash
python prepare_scifact.py --dry-run
python prepare_scifact.py --output data/scifact_allenai_reference.json
python run_experiment.py --task SciFact --data data/scifact_allenai_reference.json --dimensions 3072 1536 768 384 --dry-run
```

Only preparation accesses Hugging Face for dataset data/code. It resolves both
`--revision` (MTEB) and `--reference-revision` (AllenAI) to separate immutable SHAs.
The current AllenAI source uses a Python dataset loader; preparation explicitly
uses that pinned official loader with `trust_remote_code=True` and streaming,
iterating only train and validation. It neither requests nor iterates AllenAI test
examples. The upstream archive can contain other splits, but they are not used.
Use the existing script-compatible `datasets==3.6.0` environment; newer versions
that remove dataset-script support may require a reviewed loader migration.
No dependency changes are made automatically.

The AllenAI script points to a mutable upstream archive: a dataset-script commit
alone does not pin its bytes. Input-text hashes record the actual loaded calibration
content; archive the exact exported bundle for reproducibility. To repeat preparation,
pass both recorded revisions and use a fresh output path, then compare content hashes.
The script never imports a language model or overwrites an existing export.

Real preparation has not been executed. Remaining checks are source-loader
compatibility, actual post-exclusion reference count, and eventual embedding rank.
If fewer than 1,537 texts remain, this source cannot support the full Phi dimension
matrix and a separately approved strategy will be required.

## Arxiv-Clustering

Collections: `train` (unlabelled reference `{id, text}` rows) and `evaluation`
(`{id, text, label}` rows). The field `train` is the runner's fitting partition;
its actual provenance must be recorded in `fit_source`.

Also require `source.variant` to be `S2S` or `P2P`, and a nonempty `source.subset_id`.
Record the exact text construction in `source.text_format`. No variant is assumed.
Each invocation evaluates **one subset**; do not concatenate official subsets or
label taxonomies without a separate, documented protocol. The same reference pool,
evaluation rows and K must be used across dimensions.

Pass `--clusters K` only after choosing K from a predefined taxonomy or training-only
decision. Record its rationale in an additional `source.cluster_count_basis` field
(retained in metadata). Evaluation labels are only for final scoring. The runner
never infers K from them and never fits centroids on evaluation vectors. This is an
**inductive clustering** experiment and is not directly comparable with standard
MTEB clustering scores. Multiple subsets and their aggregation remain future work.

Outstanding: instructor/source confirmation of S2S vs P2P, dataset/configuration/
revision, intended subsets and taxonomy, independent reference split and K. If no
eligible reference pool exists, leave execution blocked; do not reuse official test
texts for fitting. The inductive interpretation may need instructor confirmation.

## Validation and later execution

After preparing reviewed bundles, validate without model imports or network access:

```bash
python run_experiment.py --task Banking77 --data data/banking77_official.json --dry-run
python run_experiment.py --task SciFact --data data/scifact_allenai_reference.json --dry-run
```

For arXiv, additionally supply `--clusters` with the verified integer K. Until that
choice and a bundle exist, `python run_experiment.py --task Arxiv-Clustering --dry-run`
reports the blockers. With valid data, dry-run prints counts, source provenance and
bundle hash. It checks PCA sample availability, not embedding rank or GPU capacity.

For a subsequently approved GPU run, remove `--dry-run` and supply a fresh `--output`
directory. Other options default to all four Phi dimensions, 2,000 PCA rows, batch
4, max length 256 and seed 42. No new-task results exist yet.
