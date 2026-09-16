# Local task data bundles (schema version 1)

The new runner reads UTF-8 JSON via `--data`; it does not fetch new-task datasets.
This separates verified dataset preparation from model execution. Banking77 has a dedicated data-only exporter (`prepare_banking77.py`) for the
verified `PolyAI/banking77` official train/test protocol. SciFact/arXiv
exporters remain unimplemented because their source choices are unresolved. Once a reviewed bundle is
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
Training role requires `split: train`. No fitting split may be labelled validation,
test, evaluation or heldout. A derived reference is allowed only for an explicitly
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

Collections: `reference`, `queries`, `corpus`, and mapping `qrels`.

- `reference`: fitting-only `{id, text}` rows from a documented independent pool
  or eligible official training source; never evaluation claims/corpus documents.
- `queries`: evaluation claims as `{id, text}`.
- `corpus`: all benchmark candidate documents as `{id, text}`, with
  `text = title + "\n" + abstract`. If the source abstract is a sentence list, join
  sentences with a single space and record this in provenance. Set
  `source.text_format: title_newline_abstract`.
- `qrels`: `{query_id: {document_id: relevance_grade}}`, using the original judgments.
  Cover exactly the selected evaluation query IDs. Every judged document must exist
  in the supplied corpus, and each selected query must have a positive judgment.
  Unjudged documents are not removed. Preserve query/corpus ID namespaces separately.

PCA fits only `reference`, and the same frozen transform is applied to both query
and corpus vectors. There is no trained retriever. Query-by-query exact scoring
avoids allocating a full query/corpus similarity matrix. Rankings save top 100.

Outstanding: verify source/revision, evaluation query split and complete corpus/qrels
mapping; choose and document an independent calibration pool with >=2,000 eligible
sentences for the default PCA run. Small SciFact training query counts cannot meet
that requirement by themselves. Corpus-derived PCA would violate this project's
current fitting constraint. External calibration introduces domain effects that
must later be reported as a limitation.

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
python run_experiment.py --task SciFact --data data/scifact.json --dry-run
```

For arXiv, additionally supply `--clusters` with the verified integer K. Until that
choice and a bundle exist, `python run_experiment.py --task Arxiv-Clustering --dry-run`
reports the blockers. With valid data, dry-run prints counts, source provenance and
bundle hash. It checks PCA sample availability, not embedding rank or GPU capacity.

For a subsequently approved GPU run, remove `--dry-run` and supply a fresh `--output`
directory. Other options default to all four Phi dimensions, 2,000 PCA rows, batch
4, max length 256 and seed 42. No new-task results exist yet.
