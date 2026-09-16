"""Task-aware plans. No model, network, or scientific-library imports."""
from pilot.config import MODEL_SPECS, TASK_SPECS, LOADING_STRATEGY, validate_dimensions
from pilot.task_data import fingerprint, validate_bundle

PROTOCOLS = {
    "STSB": {"id": "sts-validation-pilot-v1", "metrics": ["cosine_spearman"]},
    "SciFact": {"id": "exact-cosine-linear-ndcg-v1", "metrics": ["ndcg_at_10", "recall_at_10", "recall_at_100", "mrr_at_10"],
                "dataset_id": "mteb/scifact", "evaluation_split": "test",
                "pca_reference": "allenai/scifact claims train+validation; all eligible unique texts",
                "ranking": "exact cosine; descending score then ascending document ID", "gain": "linear relevance"},
    "Banking77": {"id": "banking77-official-train-test-logistic-v1", "metrics": ["accuracy", "macro_f1"],
                  "dataset_id": "PolyAI/banking77", "fit_split": "train", "evaluation_split": "test",
                  "train_rows": 10003, "test_rows": 3080, "classes": 77,
                  "classifier": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000, "tol": 1e-4, "class_weight": None}},
    "Arxiv-Clustering": {"id": "arxiv-s2s-per-set-minibatch-kmeans-v1", "metrics": ["v_measure", "adjusted_rand", "nmi"],
                         "task_name": "ArxivClusteringS2S", "k_policy": "number of unique labels per official set",
                         "aggregation": "unweighted arithmetic mean over sets",
                         "clustering": {"init": "k-means++", "n_init": 10, "max_iter": 100, "batch_size": 1024,
                                        "tol": 0.0, "max_no_improvement": 10, "reassignment_ratio": 0.01}},
}


def task_plan(task, model="Phi4-mini", dimensions=None, train_count=2000, seed=42,
              batch_size=4, max_length=256, clusters=None, bundle=None, loading_strategy=LOADING_STRATEGY):
    if task not in TASK_SPECS or model not in MODEL_SPECS:
        raise ValueError("Unknown assignment task/model.")
    spec = MODEL_SPECS[model]
    dims = validate_dimensions(spec["dimensions"] if dimensions is None else dimensions, spec["native_dimension"])
    if set(dims) - set(spec["dimensions"]):
        raise ValueError("Only assignment-required dimensions are supported.")
    components = max((d for d in dims if d < spec["native_dimension"]), default=0)
    if (type(seed) is not int or seed < 0 or
            (task not in ("SciFact", "Arxiv-Clustering") and (type(train_count) is not int or train_count <= components))):
        raise ValueError(f"Need >{components} calibration rows and a nonnegative integer seed.")
    if task in ("SciFact", "Arxiv-Clustering"):
        train_count = None  # Resolved from all eligible reference rows; never the CLI's 2000 default.
    if batch_size < 1 or not 8 <= max_length <= 512 or loading_strategy != LOADING_STRATEGY:
        raise ValueError("Invalid batch size, input length, or unsupported loading strategy.")
    if clusters is not None:
        raise ValueError("--clusters is no longer supported; ArxivClusteringS2S determines K separately for each official set.")
    blockers = []
    if model != "Phi4-mini":
        blockers.append("Checkpoint, model adapter and loading feasibility are not verified for this model.")
    summary = None
    if task == "Arxiv-Clustering":
        from pilot.arxiv import PCA_BLOCKER
        if components:
            blockers.append(PCA_BLOCKER)
        if bundle is None:
            blockers.append("Provide --data with prepared official ArxivClusteringS2S clustering sets.")
        else:
            validate_bundle(bundle, task)
            summary = {"source": bundle["source"], "bundle_hash": fingerprint(bundle),
                       "set_count": len(bundle["sets"]),
                       "evaluation_count": sum(len(g["sentences"]) for g in bundle["sets"])}
    elif task != "STSB":
        if bundle is None:
            blockers.append("Provide --data with a reviewed local bundle including dataset revision and fit/evaluation provenance.")
        else:
            validate_bundle(bundle, task)
            if task == "Banking77":
                from pilot.banking77 import validate_official_bundle
                validate_official_bundle(bundle)
            if task == "SciFact":
                from pilot.scifact import validate_official_bundle
                validate_official_bundle(bundle)
            fit = bundle["reference"] if task == "SciFact" else bundle["train"]
            from pilot.task_data import text_key
            eligible_count = len({text_key(r["text"]) for r in fit}) if task == "Banking77" else len(fit)
            if task == "SciFact":
                train_count = eligible_count
                if components and eligible_count <= components:
                    raise ValueError(f"SciFact has {eligible_count} eligible reference texts; centered PCA requires "
                                     f">{components} rows plus sufficient numerical rank. No test-data fallback.")
            if components and eligible_count < train_count:
                raise ValueError("Not enough distinct fit/reference texts for the requested PCA calibration count.")
            summary = {"source": bundle["source"], "fit_source": bundle["fit_source"],
                       "bundle_hash": fingerprint(bundle), "fit_count": len(fit),
                       "evaluation_count": len(bundle["queries"] if task == "SciFact" else bundle["evaluation"])}
            if task == "SciFact":
                summary["corpus_count"] = len(bundle["corpus"])
    elif bundle is not None:
        raise ValueError("STSB delegates to the unchanged pilot loader; local bundles are not supported for STSB.")
    return {"model": model, "task": task, "native_dimension": spec["native_dimension"],
            "dimensions": list(dims), "pca_components": components, "pca_train_sentences": train_count,
            "seed": seed, "batch_size": batch_size, "max_length": max_length, "clusters": clusters,
            "loading_strategy": loading_strategy, "protocol": PROTOCOLS[task], "data": summary,
            "implemented": model == "Phi4-mini", "executable": not blockers, "blockers": blockers,
            "validation_scope": "offline provenance/shape configuration; no authenticity, GPU-memory or PCA-rank verification"}
