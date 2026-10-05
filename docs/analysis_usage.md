# Final-results analysis

This is analysis of existing results only. It does not load language models, fit PCA,
evaluate benchmark examples, or regenerate experiments. The required source snapshot
is `assignment-completion` at `88c9536f04dda2194c6cf20d5388588b4298171e`.

Use Python 3.12 and an isolated environment:

```bash
python -m venv .venv-analysis
# POSIX shell: source .venv-analysis/bin/activate
# PowerShell: .\.venv-analysis\Scripts\Activate.ps1
python -m pip install -r requirements-analysis.txt
python analyze_results.py
python -m pytest tests/test_analysis.py -q
```

Defaults resolve relative to the script location, so analysis can be invoked from
another directory. Optional arguments: `--raw PATH`, `--output PATH`, and
`--retention-threshold 95`. The output root receives `tables/` and `figures/`.
The validator checks all raw files before creating outputs, rejects duplicate JSON
keys, nonfinite/bool scores, missing/duplicate metric cells, inconsistent dimensions,
and inconsistent task/model/calibration provenance. It records SHA-256 digests and
verifies that raw files remain unchanged after analysis.

## Outputs and interpretation

- `comparison_long.csv`: all 210 metric observations with source-file provenance.
- Four task comparison CSV/Markdown tables: all 84 configurations, with each task's metrics.
- `relative_changes.csv`: native baseline, absolute score difference, relative change,
  and retained native percentage for every metric observation.
- `retention_summary.csv` and `.md`: smallest **tested** dimension retaining at least
  the chosen percentage per metric and, separately, simultaneously across task metrics.
- `best_observed.csv`: highest observed score per model/task/metric; ties prefer
  the smaller tested dimension. This is descriptive post-hoc selection.
- `provenance.csv`: task hardware, revisions, calibration, loading and PCA provenance.
- `validation.json`: counts, caveats, input hashes, formulas and analysis environment.
- `empirical_findings.md`: generated descriptive ranges, best observed scores and retention tables.
- Eight figures, each exported as PDF, SVG, and 300-dpi PNG: four task score plots
  against actual dimension and four relative-change plots against dimension/native.

Relative change is `100 * (score - native_score) / native_score`. Absolute difference
is in score units; multiply accuracy/F1 differences by 100 to express points on the
0–100 scale. Ratios are undefined for zero or negative native baselines and are left
blank, not imputed. A 95% threshold is an explicit descriptive convention, not an
equivalence test. Retaining a weak baseline does not imply good absolute performance.

Plots join observed points for readability; they do not predict untested dimensions.
Axes auto-scale to reveal changes. No raw-score average across incompatible tasks is
computed. Single-seed aggregate JSON scores provide neither independent replicates nor
paired per-example/per-set outcomes; no confidence intervals, significance tests, or
claims of statistical equivalence are justified. Cross-model comparisons reflect the
recorded checkpoint, loading, representation and downstream protocol together.

V-measure and arithmetic NMI are mathematically equivalent under the implemented
definitions, explaining their matching values; they are not independent evidence.
Native-to-PCA comparisons also change centering/geometry, not dimension alone.

Submission tables (`.csv`, `.md`, `validation.json`) and figures (`.pdf`, `.svg`,
`.png`) have narrow `.gitignore` exceptions and are eligible for tracking. Raw
results, large data/model artifacts and caches retain their existing policies.
Analysis regeneration overwrites derived tables/figures only; original JSON is
never written. No commit or push is needed to review the generated files.

For the full offline collection, install `requirements-test-offline.txt` and run
`python -m pytest -q`. It adds data-preparation package metadata used by mocked
tests. In an isolated environment without torch/transformers, all tests are
collected, and three model-related tests skip before any model execution. The
full experiment-development environment is separately described in README.
