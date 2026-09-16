"""Shared offline evaluation pipeline; the caller supplies a frozen native encoder."""
import csv
import json
from pathlib import Path

import numpy as np

from pilot.core import checked, fit_pca, normalize, project
from pilot.evaluation import classification, clustering, retrieval
from pilot.task_data import fingerprint, text_key, validate_bundle


def evaluate_bundle(bundle, plan, encode):
    """Fit on reference/train only; reuse a single PCA basis across dimensions."""
    task = plan["task"]
    validate_bundle(bundle, task)
    fit_rows = bundle["reference"] if task == "SciFact" else bundle["train"]
    width, seed = plan["native_dimension"], plan["seed"]
    groups = {"fit": fit_rows}
    if task == "SciFact":
        groups.update(queries=bundle["queries"], corpus=bundle["corpus"])
    else:
        groups["evaluation"] = bundle["evaluation"]
    vectors = {name: checked(encode([r["text"] for r in records]), len(records), width)
               for name, records in groups.items()}
    count = len(fit_rows) if task == "SciFact" else plan["pca_train_sentences"]
    candidates = list(range(len(fit_rows)))
    if task == "Banking77":
        first_by_text = {}
        for i, row in enumerate(fit_rows):
            first_by_text.setdefault(text_key(row["text"]), i)
        candidates = list(first_by_text.values())
    if plan["pca_components"] and count > len(candidates):
        raise ValueError("Insufficient fitting rows for PCA.")
    indices = ((candidates if task == "SciFact" else np.random.default_rng(seed).permutation(candidates)[:count])
               if plan["pca_components"] else [])
    pca = fit_pca(vectors["fit"][indices], plan["pca_components"], seed) if len(indices) else None
    rows, predictions = [], {}
    for dimension in plan["dimensions"]:
        transformed = {name: normalize(x) if dimension == width else project(x, pca, dimension)
                       for name, x in vectors.items()}
        if task == "SciFact":
            scores, pred = retrieval(transformed["queries"], [r["id"] for r in bundle["queries"]],
                                     transformed["corpus"], [r["id"] for r in bundle["corpus"]], bundle["qrels"])
            evaluation_rows = bundle["queries"]
        elif task == "Banking77":
            evaluation_rows = bundle["evaluation"]
            scores, pred = classification(transformed["fit"], [r["label"] for r in fit_rows],
                                          transformed["evaluation"], [r["label"] for r in evaluation_rows], seed)
        else:
            evaluation_rows = bundle["evaluation"]
            scores, pred = clustering(transformed["fit"], transformed["evaluation"],
                                      [r["label"] for r in evaluation_rows], plan["clusters"], seed)
        predictions[str(dimension)] = pred
        for metric, score in scores.items():
            if not np.isfinite(score):
                raise ValueError("Non-finite task score.")
            rows.append({"model": plan["model"], "task": task, "dataset": bundle["source"]["dataset_id"],
                         "dataset_revision": bundle["source"]["revision"],
                         "split": bundle["source"]["evaluation_split"], "dimension": dimension,
                         "reduction": "native" if dimension == width else "pca", "metric": metric,
                         "score": float(score), "seed": seed, "protocol": plan["protocol"]["id"],
                         "n_eval": len(evaluation_rows)})
    metadata = {"schema_version": 1, "plan": plan, "bundle_hash": fingerprint(bundle),
                "source": bundle["source"], "fit_source": bundle["fit_source"],
                "pca_fit_ids": [fit_rows[i]["id"] for i in indices],
                "pca_fit_text_hashes": [fingerprint(fit_rows[i]["text"]) for i in indices],
                "fit_ids": [r["id"] for r in fit_rows],
                "evaluation_ids": [r["id"] for r in evaluation_rows],
                "pca": {"components": plan["pca_components"], "pre_normalize": True, "whiten": False,
                        "solver": "randomized", "native_centered": False},
                "leakage_check": "canonical fit/evaluation text disjointness; provenance asserted by bundle author"}
    return rows, {"evaluation_ids": metadata["evaluation_ids"], "by_dimension": predictions}, metadata, pca


def write_metrics(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_task_outputs(output, rows, predictions, metadata, pca):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a fresh directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_metrics(output / "metrics.csv", rows)
    (output / "results.json").write_text(json.dumps({"metadata": metadata, "scores": rows}, indent=2, allow_nan=False), encoding="utf-8")
    (output / "predictions.json").write_text(json.dumps(predictions, indent=2, allow_nan=False), encoding="utf-8")
    if pca is not None:
        np.savez(output / "pca.npz", mean=pca.mean_, components=pca.components_,
                 explained_variance_ratio=pca.explained_variance_ratio_)


def sts_metrics(result):
    """Adapt a newly executed legacy result without changing its original schema."""
    return [{"model": "Phi4-mini", "task": "STSB", "dataset": r["dataset"],
             "dataset_revision": result["metadata"]["dataset_revision"], "split": r["split"],
             "dimension": r["dimension"], "reduction": r["reduction"], "metric": r["metric"],
             "score": r["cosine_spearman"], "seed": r["seed"], "protocol": "sts-validation-pilot-v1",
             "n_eval": r["pairs"]} for r in result["scores"]]
