"""Local, provenance-labelled task data. Validation is offline and read-only."""
import hashlib
import json
import math
from pathlib import Path
import unicodedata

NEW_TASKS = ("SciFact", "Banking77", "Arxiv-Clustering")


def text_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _required_strings(obj, names, description):
    if not isinstance(obj, dict) or any(not isinstance(obj.get(n), str) or not obj[n].strip() for n in names):
        raise ValueError(f"{description} requires nonempty strings: {', '.join(names)}")


def _records(rows, role, labelled=False):
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{role} must be a nonempty list.")
    ids = set()
    for row in rows:
        _required_strings(row, ["id", "text"] + (["label"] if labelled else []), role)
        if not text_key(row["text"]) or row["id"] in ids:
            raise ValueError(f"Empty text or duplicate ID in {role}.")
        ids.add(row["id"])
    return rows


def _no_overlap(fit, evaluation, allow_training_duplicates=False):
    fit_keys = [text_key(r["text"]) for r in fit]
    if not allow_training_duplicates and len(set(fit_keys)) != len(fit_keys):
        raise ValueError("Fit/reference texts must be deduplicated before export.")
    if set(fit_keys) & {text_key(r["text"]) for r in evaluation}:
        raise ValueError("Leakage: fit/reference text overlaps evaluation data.")


def validate_bundle(bundle, task):
    if task == "Arxiv-Clustering":
        from pilot.arxiv import validate_arxiv_bundle
        return validate_arxiv_bundle(bundle)
    if not isinstance(bundle, dict) or task not in NEW_TASKS or bundle.get("schema_version") != 1 or bundle.get("task") != task:
        raise ValueError("Task data bundle has an unexpected schema/task.")
    source, fit_source = bundle.get("source"), bundle.get("fit_source")
    _required_strings(source, ["dataset_id", "revision", "configuration", "evaluation_split", "selection", "text_format"], "source")
    _required_strings(fit_source, ["dataset_id", "revision", "configuration", "split", "role", "split_method"], "fit_source")
    for provenance in (source, fit_source):
        if provenance["revision"].strip().casefold() in ("main", "master", "latest", "todo", "unknown", "unverified"):
            raise ValueError("Record an immutable revision/checksum, not a mutable or unresolved revision.")
    if source["selection"] not in ("full", "subset"):
        raise ValueError("source.selection must explicitly be full or subset.")
    if source["evaluation_split"] not in ("validation", "test", "heldout"):
        raise ValueError("Evaluation split must be validation, test, or explicitly derived heldout.")
    if fit_source["role"] not in ("train", "external_reference", "derived_reference"):
        raise ValueError("Invalid fit/reference role.")
    if fit_source["split"].casefold() in ("validation", "test", "eval", "evaluation", "heldout"):
        raise ValueError("Evaluation splits may not be used as fitting splits.")
    if fit_source["role"] == "train" and fit_source["split"] != "train":
        raise ValueError("Training role must use the designated train split.")
    if fit_source["role"] == "derived_reference":
        raise ValueError("Derived reference partitions are unsupported; Arxiv uses official clustering sets.")
    if task == "Banking77":
        if fit_source["role"] != "train" or source["evaluation_split"] != "test":
            raise ValueError("Banking77 requires the official train split and official test evaluation.")
        if any(fit_source[k] != source[k] for k in ("dataset_id", "revision", "configuration")):
            raise ValueError("Banking77 train and evaluation must share the same dataset revision/configuration.")
    if task == "SciFact":
        if source["text_format"] != "title_newline_abstract":
            raise ValueError("SciFact corpus formatting must explicitly be title_newline_abstract.")
        reference = _records(bundle.get("reference"), "reference")
        queries = _records(bundle.get("queries"), "queries")
        corpus = _records(bundle.get("corpus"), "corpus")
        _no_overlap(reference, queries + corpus)
        qids, dids = {r["id"] for r in queries}, {r["id"] for r in corpus}
        qrels = bundle.get("qrels")
        if not isinstance(qrels, dict) or set(qrels) != qids:
            raise ValueError("Qrels must cover exactly the selected evaluation queries.")
        for qid, judgments in qrels.items():
            if not isinstance(judgments, dict) or not judgments or set(judgments) - dids:
                raise ValueError(f"Missing/unknown corpus IDs in qrels for {qid}.")
            grades = list(judgments.values())
            if any(type(g) not in (int, float) or not math.isfinite(g) or g < 0 for g in grades) or not any(g > 0 for g in grades):
                raise ValueError("Each query needs finite nonnegative qrels and at least one positive judgment.")
    else:
        fit = _records(bundle.get("train"), "train/reference", labelled=task == "Banking77")
        evaluation = _records(bundle.get("evaluation"), "evaluation", labelled=True)
        _no_overlap(fit, evaluation, allow_training_duplicates=task == "Banking77")
        if {r["id"] for r in fit} & {r["id"] for r in evaluation}:
            raise ValueError("Training and evaluation IDs must be distinct (qualify split-local IDs).")
        if len(evaluation) < 2:
            raise ValueError("At least two evaluation examples are required.")
        if task == "Banking77":
            labels_by_text = {}
            for row in fit:
                key = text_key(row["text"])
                if key in labels_by_text and labels_by_text[key] != row["label"]:
                    raise ValueError("Conflicting training labels for duplicate text; review source without silently filtering rows.")
                labels_by_text[key] = row["label"]
            labels = {r["label"] for r in fit}
            if len(labels) < 2 or {r["label"] for r in evaluation} - labels:
                raise ValueError("Need >=2 training classes and no unseen evaluation classes.")
            if source["selection"] == "full" and len(labels) != 77:
                raise ValueError("A full Banking77 export must contain all 77 training labels.")
    return bundle


def load_bundle(path, task):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    bundle = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_keys)
    if not isinstance(bundle, dict):
        raise ValueError("Task data must be a JSON object.")
    return validate_bundle(bundle, task)
