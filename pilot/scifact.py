"""Official mteb/scifact retrieval export; PCA reference is eligible AllenAI train/validation claims."""
import math
import re

from pilot.task_data import fingerprint, text_key, validate_bundle

DATASET_ID = "mteb/scifact"
REFERENCE_ID = "allenai/scifact"
REFERENCE_COUNTS = {"train": 1261, "validation": 450}
COUNTS = {"corpus": 5183, "queries": 1109, "train_queries": 809,
          "test_queries": 300, "train_qrels": 919, "test_qrels": 339}
CONFIG_SPLITS = {"corpus": ["corpus"], "queries": ["queries"], "default": ["train", "test"]}


def _id(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("SciFact IDs must be nonempty strings; IDs are not coerced.")
    return value


def _index(rows, corpus=False):
    result = {}
    for row in rows:
        identifier = _id(row.get("_id"))
        if identifier in result:
            raise ValueError("Duplicate corpus/query ID.")
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Expected nonempty text.")
        if corpus and not isinstance(row.get("title"), str):
            raise ValueError("Expected corpus title string.")
        text = row["title"] + "\n" + row["text"] if corpus else row["text"]
        result[identifier] = {"id": identifier, "text": text}
    return result


def _qrels(rows, queries, corpus):
    result = {}
    for row in rows:
        qid, did = _id(row.get("query-id")), _id(row.get("corpus-id"))
        grade = row.get("score")
        if qid not in queries or did not in corpus:
            raise ValueError("Qrels refer to unknown query/corpus IDs.")
        if type(grade) not in (int, float) or not math.isfinite(grade) or grade < 0:
            raise ValueError("Qrels require finite nonnegative numeric grades.")
        judgments = result.setdefault(qid, {})
        if did in judgments:
            raise ValueError("Duplicate query/document judgment pair.")
        judgments[did] = grade
    if any(not any(g > 0 for g in grades.values()) for grades in result.values()):
        raise ValueError("Every query needs at least one positive judgment.")
    return result


def validate_official_bundle(bundle):
    validate_bundle(bundle, "SciFact")
    source, fit = bundle["source"], bundle["fit_source"]
    if (source["dataset_id"] != DATASET_ID or source["configuration"] != "corpus+queries+default"
            or source["evaluation_split"] != "test" or source["selection"] != "full"):
        raise ValueError("SciFact requires full official mteb/scifact test retrieval data.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]):
        raise ValueError("SciFact requires an immutable 40-character dataset commit SHA.")
    if (fit["dataset_id"] != REFERENCE_ID or fit["configuration"] != "claims"
            or fit["split"] != "train+validation" or fit["role"] != "external_reference"
            or fit.get("splits") != ["train", "validation"] or fit.get("text_field") != "claim"
            or fit.get("calibration_policy") != "all_eligible_unique_claims"):
        raise ValueError("PCA reference must be AllenAI claims train+validation only; never AllenAI test.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", fit["revision"]):
        raise ValueError("AllenAI reference requires its own immutable commit SHA.")
    if fit.get("candidate_counts") != REFERENCE_COUNTS:
        raise ValueError("Unexpected AllenAI candidate split counts.")
    allowed_ids = {f"{split}:{i}" for split, count in REFERENCE_COUNTS.items() for i in range(count)}
    reference_ids = {r["id"] for r in bundle["reference"]}
    exclusions = fit.get("excluded_rows", [])
    excluded_ids = [r["id"] for r in exclusions]
    if (any(r.get("reason") not in ("retrieval_test_query", "retrieval_corpus", "duplicate_claim") for r in exclusions)
            or len(excluded_ids) != len(set(excluded_ids)) or reference_ids & set(excluded_ids)
            or reference_ids | set(excluded_ids) != allowed_ids):
        raise ValueError("AllenAI reference/exclusion manifest must account for all candidate rows exactly once.")
    if len(bundle["corpus"]) != COUNTS["corpus"] or len(bundle["queries"]) != COUNTS["test_queries"]:
        raise ValueError("Unexpected official corpus/test query counts.")
    if sum(map(len, bundle["qrels"].values())) != COUNTS["test_qrels"]:
        raise ValueError("Unexpected official test qrels count.")
    train_ids = source.get("train_query_ids")
    if (not isinstance(train_ids, list) or any(not isinstance(i, str) for i in train_ids)
            or len(train_ids) != COUNTS["train_queries"] or len(set(train_ids)) != len(train_ids)):
        raise ValueError("Invalid official train-query membership manifest.")
    if set(train_ids) & {r["id"] for r in bundle["queries"]}:
        raise ValueError("Train and test query IDs overlap.")
    if len(bundle["reference"]) != fit.get("eligible_reference_count"):
        raise ValueError("Reference count does not match provenance.")
    for name in ("corpus", "queries", "qrels", "reference"):
        if source.get("content_hashes", {}).get(name) != fingerprint(bundle[name]):
            raise ValueError(f"SciFact {name} content hash mismatch.")
    return bundle


def build_bundle(corpus_rows, query_rows, train_rows, test_rows, revision, reference_rows, reference_revision):
    """Preserve retrieval evaluation; separately select eligible AllenAI claim texts."""
    for name, rows in (("corpus", corpus_rows), ("queries", query_rows),
                       ("train_qrels", train_rows), ("test_qrels", test_rows)):
        if len(rows) != COUNTS[name]:
            raise ValueError(f"Unexpected {name} count; expected {COUNTS[name]}.")
    corpus, queries = _index(corpus_rows, corpus=True), _index(query_rows)
    train, test = _qrels(train_rows, queries, corpus), _qrels(test_rows, queries, corpus)
    if len(train) != COUNTS["train_queries"] or len(test) != COUNTS["test_queries"]:
        raise ValueError("Unexpected train/test query membership counts.")
    if set(train) & set(test) or set(train) | set(test) != set(queries):
        raise ValueError("Train/test query membership must be disjoint and cover the query table.")
    evaluation_queries = [r for i, r in queries.items() if i in test]
    reference, reference_metadata = select_reference(reference_rows, reference_revision,
                                                      evaluation_queries, list(corpus.values()))
    bundle = {"schema_version": 1, "task": "SciFact", "reference": reference,
              "corpus": list(corpus.values()), "queries": evaluation_queries, "qrels": test}
    bundle["source"] = {"dataset_id": DATASET_ID, "revision": revision,
                        "configuration": "corpus+queries+default", "configuration_splits": CONFIG_SPLITS,
                        "evaluation_split": "test", "selection": "full", "text_format": "title_newline_abstract",
                        "official_counts": dict(COUNTS), "train_query_ids": list(train),
                        "train_qrels_hash": fingerprint(list(train_rows)),
                        "content_hashes": {k: fingerprint(bundle[k]) for k in ("corpus", "queries", "qrels", "reference")}}
    bundle["fit_source"] = reference_metadata
    return validate_official_bundle(bundle)


def select_reference(partitions, revision, evaluation_queries, corpus):
    """Exclude held-out text matches and repeats; never read labels or qrels here."""
    if set(partitions) != set(REFERENCE_COUNTS):
        raise ValueError("Reference must contain only AllenAI train and validation; test is forbidden.")
    test_texts = {text_key(r["text"]) for r in evaluation_queries}
    corpus_texts = {text_key(r["text"]) for r in corpus}
    reference, excluded, seen, input_hashes = [], [], set(), {}
    for split, expected in REFERENCE_COUNTS.items():
        rows = partitions[split]
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} AllenAI {split} rows.")
        candidates = []
        for i, row in enumerate(rows):
            claim = row.get("claim")
            if not isinstance(claim, str) or not text_key(claim):
                raise ValueError("AllenAI claim field must be nonempty text.")
            identifier, key = f"{split}:{i}", text_key(claim)
            candidates.append({"id": identifier, "claim": claim})
            reason = ("retrieval_test_query" if key in test_texts else
                      "retrieval_corpus" if key in corpus_texts else "duplicate_claim" if key in seen else None)
            if reason:
                excluded.append({"id": identifier, "reason": reason, "text_hash": fingerprint(claim)})
            else:
                seen.add(key)
                reference.append({"id": identifier, "text": claim})
        input_hashes[split] = fingerprint(candidates)
    metadata = {"dataset_id": REFERENCE_ID, "revision": revision, "configuration": "claims",
                "split": "train+validation", "splits": ["train", "validation"], "role": "external_reference",
                "text_field": "claim", "calibration_policy": "all_eligible_unique_claims",
                "candidate_counts": dict(REFERENCE_COUNTS), "input_text_hashes": input_hashes,
                "eligible_reference_count": len(reference), "excluded_rows": excluded,
                "split_method": "All train then validation claim texts in source order; canonical deduplication "
                                "and exclusion of retrieval test-query/corpus text matches. No labels, qrels "
                                "or AllenAI test data fit PCA."}
    return reference, metadata
