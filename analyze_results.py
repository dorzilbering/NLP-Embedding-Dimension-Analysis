"""Offline, read-only analysis of final JSON results; never imports model code.

Run: python analyze_results.py
Only output directories are written. Validation fails closed before plotting.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import re

ROOT = Path(__file__).resolve().parent
# Assignment specifications, not experimental scores. Kept independent of runners.
DIMENSIONS = {
    "Qwen3-8B": (4096, 2048, 1024, 512, 256),
    "Gemma3-4B": (2560, 1280, 640, 320),
    "Llama3.2-3B": (3072, 1536, 768, 384),
    "Phi4-mini": (3072, 1536, 768, 384),
    "Mistral-7B": (4096, 2048, 1024, 512),
}
METRICS = {
    "STSB": ("cosine_spearman",),
    "Banking77": ("accuracy", "macro_f1"),
    "SciFact": ("ndcg_at_10", "mrr_at_10", "recall_at_10", "recall_at_100"),
    "Arxiv-Clustering": ("v_measure", "nmi", "adjusted_rand"),
}
DATASETS = dict(zip(METRICS, ("mteb/stsbenchmark-sts", "PolyAI/banking77",
                            "mteb/scifact", "mteb/arxiv-clustering-s2s")))
LABELS = {"cosine_spearman": "Cosine Spearman", "accuracy": "Accuracy",
          "macro_f1": "Macro-F1", "ndcg_at_10": "nDCG@10", "mrr_at_10": "MRR@10",
          "recall_at_10": "Recall@10", "recall_at_100": "Recall@100",
          "v_measure": "V-measure", "nmi": "NMI", "adjusted_rand": "Adjusted Rand index"}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def validate(raw):
    """Validate every raw file, every metric cell, and comparable provenance."""
    raw = Path(raw)
    expected = {(m, t, d, k) for m, dims in DIMENSIONS.items()
                for t, metrics in METRICS.items() for d in dims for k in metrics}
    seen = collections.Counter()
    rows, provenance, issues, hashes = [], [], [], {}
    files = sorted(p for p in raw.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(raw).as_posix()
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        try:
            if path.suffix != ".json":
                raise ValueError("Unexpected non-JSON raw file")
            data = json.loads(path.read_text(encoding="utf-8-sig"),
                              object_pairs_hook=unique_keys, parse_constant=reject_constant)
            meta, scores = data["metadata"], data["scores"]
            plan = meta["plan"]
            model, task = plan["model"], plan["task"]
            dims, metrics = DIMENSIONS[model], METRICS[task]
            def require(condition, message):
                if not condition:
                    raise ValueError(message)
            require(rel == f"{model}/{task}/results.json", "Path/model/task mismatch")
            require(meta["schema_version"] == 2, "Unexpected schema version")
            require(plan["native_dimension"] == dims[0] and plan["dimensions"] == list(dims),
                    "Plan does not match assignment dimensions")
            require(plan["pca_components"] == dims[1], "Unexpected PCA component count")
            require(plan["pca_train_sentences"] == meta["calibration_count"] == 3072,
                    "Unexpected calibration count")
            require(bool(re.fullmatch(r"[0-9a-f]{64}", meta["calibration_hash"])), "Invalid calibration hash")
            require(bool(re.fullmatch(r"[0-9a-f]{40}", meta["model_revision"])), "Invalid model revision")
            require(set(plan["protocol"]["metrics"]) == set(metrics), "Plan metric mismatch")
            require(isinstance(scores, list) and bool(scores), "Scores must be a nonempty list")
            source = meta.get("source", {})
            dsrev = meta.get("dataset_revision", source.get("revision"))
            for row in scores:
                key = (row["model"], row["task"], row["dimension"], row["metric"])
                seen[key] += 1
                require(row["model"] == model and row["task"] == task, "Row identity mismatch")
                require(type(row["dimension"]) is int and key in expected, "Unexpected configuration/metric")
                value = row["score"]
                require(type(value) in (int, float) and math.isfinite(value), "Invalid score")
                low = -1 if row["metric"] in ("cosine_spearman", "adjusted_rand") else 0
                require(low <= value <= 1, "Score outside metric range")
                require(row["reduction"] == ("native" if row["dimension"] == dims[0] else "pca"),
                        "Incorrect reduction label")
                require(type(row["seed"]) is int and row["seed"] == plan["seed"], "Seed mismatch")
                require(row["protocol"] == plan["protocol"]["id"], "Protocol mismatch")
                require(row["dataset"] == DATASETS[task] and row["dataset_revision"] == dsrev,
                        "Dataset provenance mismatch")
                require(bool(re.fullmatch(r"[0-9a-f]{40}", row["dataset_revision"])), "Invalid dataset revision")
                require(row["split"] == ("validation" if task == "STSB" else "test"), "Split mismatch")
                require(type(row["n_eval"]) is int and row["n_eval"] > 0, "Invalid evaluation count")
                rows.append({**row, "source_file": rel})
            provenance.append({"model": model, "task": task, "model_id": plan["model_id"],
                               "model_revision": meta["model_revision"], "calibration_hash": meta["calibration_hash"],
                               "calibration_count": meta["calibration_count"], "dataset_revision": dsrev,
                               "gpu": meta["hardware"]["gpu"], "cuda": meta["hardware"]["cuda"],
                               "quantization": plan["quantization"], "max_length": plan["max_length"],
                               "batch_size": plan["batch_size"], "seed": plan["seed"],
                               "pca_artifact": meta["pca_artifact"],
                               "pca_artifact_reused": meta["pca_artifact_reused"],
                               "source_file": rel})
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            issues.append(f"{rel}: {error}")
    missing, extra = expected - set(seen), set(seen) - expected
    issues += [f"Missing metric cell: {k}" for k in sorted(missing)]
    issues += [f"Unexpected metric cell: {k}" for k in sorted(extra)]
    issues += [f"Duplicate metric cell ({n}): {k}" for k, n in seen.items() if n > 1]
    if len(provenance) != 20:
        issues.append(f"Expected 20 result files; found {len(provenance)} valid files")
    for task in METRICS:
        group = [r for r in rows if r["task"] == task]
        identities = {(r["dataset_revision"], r["split"], r["n_eval"], r["seed"], r["protocol"]) for r in group}
        if len(identities) != 1:
            issues.append(f"Inconsistent task provenance: {task}")
    if len({(p["calibration_hash"], p["calibration_count"]) for p in provenance}) != 1:
        issues.append("Inconsistent shared calibration provenance")
    for model in DIMENSIONS:
        group = [p for p in provenance if p["model"] == model]
        for field in ("model_revision", "model_id", "pca_artifact", "quantization", "max_length", "seed"):
            if len({p[field] for p in group}) != 1:
                issues.append(f"Inconsistent {field}: {model}")
    warnings = ["Aggregate single-seed scores only: no significance tests or confidence intervals are supported.",
                "Calibration bundle, PCA arrays, predictions, and historical package versions are absent from this checkout."]
    for model in DIMENSIONS:
        gpus = sorted({p["gpu"] for p in provenance if p["model"] == model})
        if len(gpus) > 1:
            warnings.append(f"{model}: mixed task hardware {gpus}; historical dtype cannot be inferred from the loading-policy name alone.")
    report = {"status": "pass" if not issues else "fail", "expected_configurations": 84,
              "represented_configurations": len({k[:3] for k in seen if k in expected}),
              "expected_metric_rows": 210, "metric_rows": sum(seen.values()), "raw_file_count": len(files),
              "issues": issues, "warnings": warnings, "sha256": hashes}
    return rows, provenance, report


def changes(rows):
    baselines = {(r["model"], r["task"], r["metric"]): r["score"] for r in rows
                 if r["dimension"] == DIMENSIONS[r["model"]][0]}
    output = []
    for row in rows:
        native = DIMENSIONS[row["model"]][0]
        base = baselines[row["model"], row["task"], row["metric"]]
        delta = row["score"] - base
        # Ratios of signed/zero baselines have no useful retention interpretation.
        output.append({**row, "native_dimension": native, "dimension_fraction": row["dimension"] / native,
                       "native_score": base, "delta_score": delta,
                       "relative_change_pct": 100 * delta / base if base > 0 else None,
                       "retained_native_pct": 100 * (row["score"] / base) if base > 0 else None})
    return output


def retention_summary(rows, threshold=95):
    """Smallest tested dimension meeting threshold for each metric and all task metrics.

    Descriptive threshold, not equivalence testing. Unknown when native <= 0.
    """
    result = []
    for model in DIMENSIONS:
        for task, metrics in METRICS.items():
            for selected in [(k,) for k in metrics] + ([tuple(metrics)] if len(metrics) > 1 else []):
                group = [r for r in rows if r["model"] == model and r["task"] == task and r["metric"] in selected]
                eligible = [d for d in DIMENSIONS[model] if all(
                    r["retained_native_pct"] is not None and r["retained_native_pct"] >= threshold
                    for r in group if r["dimension"] == d)]
                dimension = min(eligible) if eligible else None
                result.append({"model": model, "task": task, "metric": selected[0] if len(selected) == 1 else "all_task_metrics",
                               "threshold_pct": threshold, "smallest_tested_dimension": dimension,
                               "dimension_fraction": dimension / DIMENSIONS[model][0] if dimension else None})
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows, fields):
    def fmt(value):
        return "NA" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)
    return "| " + " | ".join(fields) + " |\n| " + " | ".join("---" for _ in fields) + " |\n" + "\n".join(
        "| " + " | ".join(fmt(r[f]) for f in fields) + " |" for r in rows) + "\n"


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 300, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    colors = dict(zip(DIMENSIONS, plt.get_cmap("tab10").colors))
    markers = dict(zip(DIMENSIONS, ("o", "s", "^", "D", "v")))
    for task, metrics in METRICS.items():
        for mode in ("dimension", "relative"):
            fig, axes = plt.subplots(1, len(metrics), figsize=(max(6, len(metrics) * 4), 4.5), squeeze=False)
            for ax, metric in zip(axes[0], metrics):
                for model in DIMENSIONS:
                    group = sorted((r for r in rows if r["model"] == model and r["task"] == task and r["metric"] == metric), key=lambda r: r["dimension"])
                    x = [r["dimension"] if mode == "dimension" else r["dimension_fraction"] for r in group]
                    y = [r["score"] if mode == "dimension" else r["relative_change_pct"] for r in group]
                    ax.plot(x, y, marker=markers[model], color=colors[model], label=model, linewidth=1.5, markersize=5)
                ax.set_xscale("log", base=2)
                if mode == "dimension":
                    ticks = sorted({r["dimension"] for r in rows if r["task"] == task})
                    ax.set_xticks(ticks)
                    ax.xaxis.set_major_formatter(ScalarFormatter())
                    ax.tick_params(axis="x", rotation=60, labelsize=8)
                    ax.set_xlabel("Embedding dimension")
                    ax.set_ylabel(LABELS[metric])
                else:
                    ax.set_xticks([1/16, 1/8, 1/4, 1/2, 1], ["1/16", "1/8", "1/4", "1/2", "1"])
                    ax.axhline(0, color="black", linewidth=.8)
                    ax.axhline(-5, color="gray", linestyle=":", linewidth=.8)
                    ax.set_xlabel("Dimension / native dimension")
                    ax.set_ylabel("Change from native (%)")
                ax.set_title(LABELS[metric])
                ax.grid(alpha=.2)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=3 if len(metrics) == 1 else 5, fontsize=9)
            fig.suptitle(task + (" — observed scores" if mode == "dimension" else " — relative change from native"))
            fig.tight_layout(rect=(0, .16 if len(metrics) == 1 else .1, 1, .93))
            stem = task.lower().replace("-", "_") + "_" + mode
            for extension in ("pdf", "svg", "png"):
                fig.savefig(output / f"{stem}.{extension}")
            plt.close(fig)


def generate(raw, output, threshold=95):
    raw, output = Path(raw).resolve(), Path(output).resolve()
    for destination in (output / "tables", output / "figures"):
        if destination == raw or raw in destination.parents or destination in raw.parents:
            raise ValueError("Analysis output must not overlap raw results")
    rows, provenance, report = validate(raw)
    if report["issues"]:
        raise ValueError(json.dumps(report, indent=2))
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    relative = changes(rows)
    summary = retention_summary(relative, threshold)
    write_csv(tables / "comparison_long.csv", rows)
    write_csv(tables / "relative_changes.csv", relative)
    write_csv(tables / "retention_summary.csv", summary)
    write_csv(tables / "provenance.csv", provenance)
    best = []
    for task, metrics in METRICS.items():
        wide = []
        for model, dimensions in DIMENSIONS.items():
            for dim in dimensions:
                group = [r for r in rows if r["model"] == model and r["task"] == task and r["dimension"] == dim]
                wide.append({"model": model, "dimension": dim, **{r["metric"]: r["score"] for r in group}})
            for metric in metrics:
                group = [r for r in relative if r["model"] == model and r["task"] == task and r["metric"] == metric]
                winner = max(group, key=lambda r: (r["score"], -r["dimension"]))
                best.append({k: winner[k] for k in ("model", "task", "metric", "dimension", "score", "native_score", "delta_score", "relative_change_pct")})
        stem = task.lower().replace("-", "_")
        write_csv(tables / f"{stem}_comparison.csv", wide)
        (tables / f"{stem}_comparison.md").write_text(markdown_table(wide, list(wide[0])), encoding="utf-8")
    write_csv(tables / "best_observed.csv", best)
    (tables / "retention_summary.md").write_text(markdown_table(summary, list(summary[0])), encoding="utf-8")
    sections = ["# Descriptive empirical findings\n",
                "Generated exclusively from validated raw JSON. Relative changes compare each metric to the same model's native score. These are single-seed observations, not significance claims.\n"]
    for task, metrics in METRICS.items():
        sections.append(f"## {task}\n")
        for metric in metrics:
            group = [r for r in relative if r["task"] == task and r["metric"] == metric]
            winner = max(group, key=lambda r: r["score"])
            half = [r for r in group if r["dimension_fraction"] == .5]
            smallest = [r for r in group if r["dimension"] == min(DIMENSIONS[r["model"]])]
            half_delta = [r["relative_change_pct"] for r in half]
            small_delta = [r["relative_change_pct"] for r in smallest]
            sections.append(f"- {LABELS[metric]}: highest observed score {winner['score']:.6f} ({winner['model']}, {winner['dimension']} dimensions). Half-dimension changes range from {min(half_delta):+.2f}% to {max(half_delta):+.2f}%; smallest-tested-dimension changes range from {min(small_delta):+.2f}% to {max(small_delta):+.2f}%.\n")
        sections.append("\nBest observed dimension within each model (post-hoc):\n\n" +
                        markdown_table([r for r in best if r["task"] == task], list(best[0])))
        selected = [r for r in summary if r["task"] == task and
                    (r["metric"] == "all_task_metrics" or len(metrics) == 1)]
        sections.append(f"\nSmallest tested dimension retaining at least {threshold:g}% simultaneously across this task's metrics:\n\n" +
                        markdown_table(selected, list(selected[0])))
    sections.append("\n## Limits\n\nPCA also changes centering and geometry. Cross-model results reflect checkpoints and loading policies together. V-measure and arithmetic NMI are equivalent definitions here. Low SciFact baselines make relative changes numerically large; always read absolute scores alongside ratios. No pooled task score, statistical significance, uncertainty interval, or prediction at untested dimensions is inferred.\n")
    (tables / "empirical_findings.md").write_text("\n".join(sections), encoding="utf-8")
    plots(relative, figures)
    import importlib.metadata
    report["analysis_environment"] = {"python": platform.python_version(), **{
        package: importlib.metadata.version(package) for package in ("matplotlib", "numpy")}}
    report["retention_threshold_pct"] = threshold
    report["formulas"] = {"delta_score": "score - native_score", "relative_change_pct": "100 * (score - native_score) / native_score, only if native_score > 0",
                          "retained_native_pct": "100 * score / native_score, only if native_score > 0"}
    after = {p.relative_to(raw).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in raw.rglob("*") if p.is_file()}
    if after != report["sha256"]:
        raise RuntimeError("Raw inputs changed during analysis")
    (tables / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "represented_configurations", "metric_rows", "warnings")}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "results/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--retention-threshold", type=float, default=95)
    args = parser.parse_args()
    if not 0 < args.retention_threshold <= 100:
        parser.error("Retention threshold must be in (0, 100]")
    generate(args.raw, args.output, args.retention_threshold)


if __name__ == "__main__":
    main()
