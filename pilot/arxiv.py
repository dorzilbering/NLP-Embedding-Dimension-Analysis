"""ArxivClusteringS2S: preserve and evaluate each official clustering set separately."""
import re

from pilot.task_data import fingerprint

DATASET_ID = "mteb/arxiv-clustering-s2s"
TASK_NAME = "ArxivClusteringS2S"
PCA_BLOCKER = ("ArxivClusteringS2S has only test clustering sets and no independent PCA fitting split. "
               "Reduced dimensions (including 1536) are blocked: no test-fitted PCA, invented train split "
               "or external reference is permitted. Native embeddings can be evaluated.")


def validate_arxiv_bundle(bundle):
    if not isinstance(bundle, dict) or bundle.get("schema_version") != 1 or bundle.get("task") != "Arxiv-Clustering":
        raise ValueError("Unexpected ArxivClusteringS2S bundle schema/task.")
    if any(key in bundle for key in ("train", "reference", "evaluation", "fit_source")):
        raise ValueError("ArxivClusteringS2S uses clustering sets, not train/reference/evaluation partitions.")
    source = bundle.get("source", {})
    if (source.get("dataset_id") != DATASET_ID or source.get("task_name") != TASK_NAME
            or source.get("evaluation_split") != "test" or source.get("selection") != "full"
            or not isinstance(source.get("configuration"), str) or not source["configuration"].strip()
            or not re.fullmatch(r"[0-9a-fA-F]{40}", str(source.get("revision", "")))):
        raise ValueError("Expected official ArxivClusteringS2S test provenance and immutable revision.")
    sets = bundle.get("sets")
    if not isinstance(sets, list) or not sets or source.get("set_count") != len(sets):
        raise ValueError("Missing clustering sets or inconsistent set count.")
    for index, group in enumerate(sets):
        if not isinstance(group, dict) or group.get("id") != f"test:{index}":
            raise ValueError("Preserve clustering-set IDs/order.")
        sentences, labels = group.get("sentences"), group.get("labels")
        if (not isinstance(sentences, list) or not isinstance(labels, list) or len(sentences) != len(labels)
                or len(sentences) < 2 or any(not isinstance(t, str) or not t.strip() for t in sentences)
                or any(not isinstance(label, str) or not label for label in labels) or len(set(labels)) < 2):
            raise ValueError("Each clustering set needs aligned nonempty sentence/label lists and >=2 classes.")
    if source.get("sets_hash") != fingerprint(sets):
        raise ValueError("Arxiv clustering-set hash mismatch.")
    return bundle


def build_bundle(rows, revision, configuration):
    sets = [{"id": f"test:{i}", "sentences": row["sentences"], "labels": row["labels"]}
            for i, row in enumerate(rows)]
    return validate_arxiv_bundle({"schema_version": 1, "task": "Arxiv-Clustering", "sets": sets,
                                 "source": {"dataset_id": DATASET_ID, "task_name": TASK_NAME,
                                            "configuration": configuration, "revision": revision,
                                            "evaluation_split": "test", "selection": "full",
                                            "set_count": len(sets), "sets_hash": fingerprint(sets)}})


def evaluate_arxiv(bundle, plan, encode):
    from pilot.core import checked
    from pilot.evaluation import clustering
    validate_arxiv_bundle(bundle)
    width = plan["native_dimension"]
    if plan["pca_components"] or plan["dimensions"] != [width]:
        raise ValueError(PCA_BLOCKER)  # Before encoding, even when called without CLI validation.
    per_set, predictions = [], {}
    for group in bundle["sets"]:
        vectors = checked(encode(group["sentences"]), len(group["sentences"]), width)
        metrics, pred = clustering(vectors, group["labels"], plan["seed"])
        per_set.append({"set_id": group["id"], "sentences": len(group["sentences"]),
                        "clusters": len(set(group["labels"])), **metrics})
        predictions[group["id"]] = pred
    source = bundle["source"]
    total = sum(len(g["sentences"]) for g in bundle["sets"])
    rows = [{"model": plan["model"], "task": "Arxiv-Clustering", "dataset": DATASET_ID,
             "dataset_revision": source["revision"], "split": "test", "dimension": width,
             "reduction": "native", "metric": metric,
             "score": sum(s[metric] for s in per_set) / len(per_set), "seed": plan["seed"],
             "protocol": plan["protocol"]["id"], "n_eval": total} for metric in plan["protocol"]["metrics"]]
    metadata = {"schema_version": 1, "plan": plan, "source": source, "bundle_hash": fingerprint(bundle),
                "per_set_scores": per_set, "aggregation": "unweighted arithmetic mean over official sets",
                "group_order": [g["id"] for g in bundle["sets"]],
                "pca": {"components": 0, "fit_ids": [], "policy": PCA_BLOCKER},
                "clustering": "fit_predict each complete set; labels provide K and scoring only",
                "representation": "L2 normalization only; no learned transformation"}
    return rows, {"by_dimension": {str(width): predictions}, "alignment": "original sentence order within each set"}, metadata, None
