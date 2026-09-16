"""Deterministic task evaluators; fitting inputs are separate from scoring inputs."""
import warnings
import numpy as np
from pilot.core import normalize
from pilot.task_config import PROTOCOLS


def classification(train_x, train_labels, eval_x, eval_labels, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.exceptions import ConvergenceWarning
    train_x, eval_x = normalize(train_x), normalize(eval_x)
    if len(train_x) != len(train_labels) or len(eval_x) != len(eval_labels) or train_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Classification embedding/label alignment mismatch.")
    labels = sorted(set(train_labels))
    if len(labels) < 2 or set(eval_labels) - set(labels):
        raise ValueError("Invalid classification label space.")
    classifier = LogisticRegression(**PROTOCOLS["Banking77"]["classifier"], random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(train_x, train_labels)
    predictions = classifier.predict(eval_x)
    return {"accuracy": float(accuracy_score(eval_labels, predictions)),
            "macro_f1": float(f1_score(eval_labels, predictions, labels=labels, average="macro", zero_division=0))}, predictions.tolist()


def clustering(reference_x, eval_x, eval_labels, clusters, seed):
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import v_measure_score, adjusted_rand_score, normalized_mutual_info_score
    reference_x, eval_x = normalize(reference_x), normalize(eval_x)
    if len(eval_labels) != len(eval_x) or reference_x.shape[1] != eval_x.shape[1]:
        raise ValueError("Clustering embedding/label alignment mismatch.")
    if clusters < 2 or len(np.unique(reference_x, axis=0)) < clusters:
        raise ValueError("Not enough distinct reference vectors for the requested clusters.")
    estimator = MiniBatchKMeans(n_clusters=clusters, random_state=seed, **PROTOCOLS["Arxiv-Clustering"]["clustering"])
    estimator.fit(reference_x)  # Never fit or fit_predict on evaluation vectors.
    predictions = estimator.predict(eval_x)
    return {"v_measure": float(v_measure_score(eval_labels, predictions)),
            "adjusted_rand": float(adjusted_rand_score(eval_labels, predictions)),
            "nmi": float(normalized_mutual_info_score(eval_labels, predictions, average_method="arithmetic"))}, predictions.tolist()


def retrieval(query_x, query_ids, corpus_x, corpus_ids, qrels):
    query_x, corpus_x = normalize(query_x), normalize(corpus_x)
    if len(query_x) != len(query_ids) or len(corpus_x) != len(corpus_ids) or query_x.shape[1] != corpus_x.shape[1]:
        raise ValueError("Retrieval ID/embedding alignment mismatch.")
    if len(set(query_ids)) != len(query_ids) or len(set(corpus_ids)) != len(corpus_ids) or set(qrels) != set(query_ids):
        raise ValueError("Duplicate IDs or missing query judgments.")
    # Lexical ID order gives an explicit stable tie-break across dimensions/platforms.
    ordering = sorted(range(len(corpus_ids)), key=lambda i: corpus_ids[i])
    corpus_x = corpus_x[ordering]
    doc_ids = [corpus_ids[i] for i in ordering]
    scores_by_query, rankings = [], {}
    for vector, qid in zip(query_x, query_ids):
        judgments = qrels[qid]
        if set(judgments) - set(doc_ids) or not judgments or any(not np.isfinite(v) or v < 0 for v in judgments.values()):
            raise ValueError("Invalid relevance judgments.")
        relevant = {d for d, grade in judgments.items() if grade > 0}
        if not relevant:
            raise ValueError("Every evaluation query must have a positive relevance judgment.")
        scores = corpus_x @ vector  # One query at a time: no full query-by-corpus matrix.
        ranked = np.argsort(-scores, kind="stable")[:100]
        ids = [doc_ids[i] for i in ranked]
        gains = np.asarray([judgments.get(d, 0.0) for d in ids[:10]], dtype=float)
        ideal = np.asarray(sorted(judgments.values(), reverse=True)[:10], dtype=float)
        dcg = float(np.sum(gains / np.log2(np.arange(len(gains)) + 2)))
        idcg = float(np.sum(ideal / np.log2(np.arange(len(ideal)) + 2)))
        rr = next((1.0 / rank for rank, d in enumerate(ids[:10], 1) if d in relevant), 0.0)
        scores_by_query.append({"ndcg_at_10": dcg / idcg,
                                "recall_at_10": len(set(ids[:10]) & relevant) / len(relevant),
                                "recall_at_100": len(set(ids) & relevant) / len(relevant), "mrr_at_10": rr})
        rankings[qid] = [{"doc_id": doc_ids[i], "score": float(scores[i])} for i in ranked]
    metrics = {name: float(np.mean([s[name] for s in scores_by_query])) for name in scores_by_query[0]}
    return metrics, rankings
