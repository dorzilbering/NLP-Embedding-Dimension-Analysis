"""Official Banking77 export contract; all validation is offline and stdlib-only."""
import hashlib
import re

from pilot.task_data import fingerprint, text_key, validate_bundle

DATASET_ID = "PolyAI/banking77"
SPLIT_COUNTS = {"train": 10003, "test": 3080}
NUM_CLASSES = 77
KNOWN_OVERLAP_TEXTS = 7
OVERLAP_POLICY = "banking77-preserve-raw-exclude-overlapping-train-v1"


def validate_official_bundle(bundle):
    validate_bundle(bundle, "Banking77")
    _validate_official_structure(bundle)
    return bundle


def _validate_official_structure(bundle):
    """Full raw-data contract, without recursively calling the generic validator."""
    source, fit = bundle["source"], bundle["fit_source"]
    if (source["dataset_id"] != DATASET_ID or source["selection"] != "full"
            or source["evaluation_split"] != "test" or source["text_format"] != "text_verbatim"
            or fit["split"] != "train" or fit["role"] != "train"):
        raise ValueError("Banking77 execution requires full official PolyAI/banking77 train/test data, verbatim text.")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", source["revision"]):
        raise ValueError("Banking77 requires a resolved immutable 40-character dataset commit SHA.")
    names = source.get("label_names")
    if (not isinstance(names, list) or len(names) != NUM_CLASSES
            or any(not isinstance(n, str) or not n.strip() for n in names) or len(set(names)) != NUM_CLASSES):
        raise ValueError("Expected 77 distinct official ClassLabel names.")
    for split, key in (("train", "train"), ("test", "evaluation")):
        rows = bundle[key]
        if len(rows) != SPLIT_COUNTS[split]:
            raise ValueError(f"Expected {SPLIT_COUNTS[split]} official {split} rows; no filtering/subsampling allowed.")
        if [r["id"] for r in rows] != [f"{split}:{i}" for i in range(SPLIT_COUNTS[split])]:
            raise ValueError("Official split row order/IDs must be preserved.")
        if {r["label"] for r in rows} != {str(i) for i in range(NUM_CLASSES)}:
            raise ValueError("Expected all 77 numeric label IDs in each official split.")
        if source.get("split_hashes", {}).get(split) != fingerprint(rows):
            raise ValueError("Official split content hash mismatch.")


def allow_known_official_overlap(bundle):
    """Narrow exception for raw official Banking77; fitting policy is separate.

    Counts/hashes establish consistency, not dataset authenticity. Unexpected
    overlap counts fail; the seven actual text hashes are exported for review.
    """
    _validate_official_structure(bundle)
    overlap = {text_key(r["text"]) for r in bundle["train"]} & {
        text_key(r["text"]) for r in bundle["evaluation"]}
    if len(overlap) != KNOWN_OVERLAP_TEXTS:
        raise ValueError(f"Unexpected official Banking77 overlap: {len(overlap)} unique canonical texts; "
                         f"expected {KNOWN_OVERLAP_TEXTS}. Review source; do not silently relax the policy.")


def official_overlap_audit(bundle):
    """Read-only audit; test labels never determine exclusions."""
    validate_official_bundle(bundle)
    test_keys = {text_key(r["text"]) for r in bundle["evaluation"]}
    excluded = [r for r in bundle["train"] if text_key(r["text"]) in test_keys]
    def canonical_hash(row):
        return hashlib.sha256(text_key(row["text"]).encode("utf-8")).hexdigest()
    return {"policy": OVERLAP_POLICY, "canonicalization": "NFKC-casefold-whitespace-v1",
            "exclusion_applied": False,
            "hash_algorithm": "sha256(canonical_text_utf8)",
            "raw_train_count": len(bundle["train"]), "raw_test_count": len(bundle["evaluation"]),
            "raw_split_hashes": dict(bundle["source"]["split_hashes"]),
            "overlap_unique_text_count": len({text_key(r["text"]) for r in excluded}),
            "excluded_train_row_count": len(excluded),
            "eligible_train_row_count": len(bundle["train"]) - len(excluded),
            "excluded_train_ids": [r["id"] for r in excluded],
            "excluded_train_rows": [{"id": r["id"], "canonical_text_sha256": canonical_hash(r)} for r in excluded],
            "overlap_text_sha256": sorted({canonical_hash(r) for r in excluded}),
            "test_text_used_for_exclusions": True, "test_labels_used_for_exclusions": False}


def build_bundle(dataset, revision, configuration):
    """Validate loaded official splits and export every row without changing text."""
    if set(dataset) != set(SPLIT_COUNTS):
        raise ValueError("Expected exactly the official train and test splits.")
    exported, label_names = {}, None
    for split, expected in SPLIT_COUNTS.items():
        partition = dataset[split]
        if len(partition) != expected or set(partition.column_names) != {"text", "label"}:
            raise ValueError(f"Unexpected {split} count/schema; expected {expected} rows with text and label.")
        names = getattr(partition.features["label"], "names", None)
        if names is None or len(names) != NUM_CLASSES or (label_names is not None and list(names) != label_names):
            raise ValueError("Expected matching 77-class ClassLabel definitions in train and test.")
        label_names = list(names)
        rows = []
        for i, row in enumerate(partition):
            label = row["label"]
            if type(label) is not int or not 0 <= label < NUM_CLASSES:
                raise ValueError("Official labels must be integer IDs in [0, 76].")
            rows.append({"id": f"{split}:{i}", "text": row["text"], "label": str(label)})
        exported[split] = rows
    source = {"dataset_id": DATASET_ID, "revision": revision, "configuration": configuration,
              "evaluation_split": "test", "selection": "full", "text_format": "text_verbatim",
              "label_names": label_names, "split_hashes": {s: fingerprint(r) for s, r in exported.items()},
              "official_counts": dict(SPLIT_COUNTS),
              "train_unique_texts": len({text_key(r["text"]) for r in exported["train"]})}
    fit_source = {"dataset_id": DATASET_ID, "revision": revision, "configuration": configuration,
                  "split": "train", "role": "train",
                  "split_method": "Official split; all rows and original order preserved; no filtering. "
                                  "Classifier uses all train rows; PCA samples distinct training texts only."}
    bundle = {"schema_version": 1, "task": "Banking77", "source": source, "fit_source": fit_source,
              "train": exported["train"], "evaluation": exported["test"]}
    return validate_official_bundle(bundle)
