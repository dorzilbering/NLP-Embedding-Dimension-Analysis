"""Deterministic, class-stratified grouped holdout from official train rows only."""
import hashlib
import math

from pilot.task_data import fingerprint, text_key


def eligible_training_rows(bundle):
    """Exclude every test-text-matching train row from optimization AND validation.

    The raw official bundle is never altered. Deterministic IDs and canonical
    hashes record the test-text-informed exclusion separately from its source.
    """
    from pilot.banking77 import official_overlap_audit
    audit = official_overlap_audit(bundle)
    excluded = set(audit["excluded_train_ids"])
    eligible = [row for row in bundle["train"] if row["id"] not in excluded]
    if {text_key(r["text"]) for r in eligible} & {text_key(r["text"]) for r in bundle["evaluation"]}:
        raise RuntimeError("Leakage remains after official Banking77 training exclusions.")
    if {r["label"] for r in eligible} != {str(i) for i in range(77)}:
        raise ValueError("Overlap exclusion removed a training class; review source.")
    audit["eligible_train_hash"] = fingerprint(eligible)
    audit["exclusion_applied"] = True
    audit["applied_to"] = "optimization_and_internal_validation"
    return eligible, audit


def split_training_rows(rows, fraction=0.1, seed=42):
    """Keep duplicate canonical texts together; preserve every row and its order.

    Per class, choose a subset of text groups whose total row count is nearest
    round(class_count * fraction), retaining at least one group on each side.
    A subset-sum DP makes the target exact when achievable. Seeded SHA256 group
    ordering breaks equivalent subset ties; smaller holdout wins count ties.
    Group constraints can prevent exact stratification. Record achieved counts.
    This function accepts no test data and performs no model/tokenizer work.
    """
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError("Validation fraction must lie in (0, 1).")
    if type(seed) is not int or seed < 0:
        raise ValueError("Validation seed must be nonnegative.")
    groups, ids = {}, set()
    for row in rows:
        if not row["id"].startswith("train:") or row["id"] in ids:
            raise ValueError("Split accepts unique official train IDs only.")
        ids.add(row["id"])
        key = text_key(row["text"])
        label = row["label"]
        if not key or label not in {str(i) for i in range(77)}:
            raise ValueError("Invalid training text or class label.")
        if key in groups and groups[key][0] != label:
            raise ValueError("Conflicting labels for duplicate canonical training text.")
        groups.setdefault(key, (label, []))[1].append(row["id"])
    if {row["label"] for row in rows} != {str(i) for i in range(77)}:
        raise ValueError("Expected all 77 classes in official training data.")
    validation_ids, counts = set(), {}
    for label in map(str, range(77)):
        ordered = [(key, group_ids) for key, (group_label, group_ids) in groups.items()
                   if group_label == label]
        ordered.sort(key=lambda item: (hashlib.sha256(f"{seed}:{label}:{item[0]}".encode()).digest(), item[0]))
        if len(ordered) < 2:
            raise ValueError(f"Class {label} needs at least two distinct text groups for isolated validation.")
        total = sum(len(group_ids) for _, group_ids in ordered)
        target = min(total - 1, max(1, int(math.floor(total * fraction + 0.5))))
        subsets = {0: ()}
        for index, (_, group_ids) in enumerate(ordered):
            for size, indices in sorted(list(subsets.items())):
                new_size = size + len(group_ids)
                if new_size < total and new_size not in subsets:
                    subsets[new_size] = indices + (index,)
        chosen_size = min((size for size in subsets if size), key=lambda size: (abs(size - target), size))
        for index in subsets[chosen_size]:
            validation_ids.update(ordered[index][1])
        counts[label] = {"train": total - chosen_size, "validation": chosen_size, "target_validation": target}
    train = [row for row in rows if row["id"] not in validation_ids]
    validation = [row for row in rows if row["id"] in validation_ids]
    metadata = {"method": "canonical-text-grouped-stratified-subset-sum-v1", "source_split": "train",
                "seed": seed, "requested_fraction": fraction,
                "achieved_fraction": len(validation) / len(rows), "class_counts": counts,
                "train_ids": [row["id"] for row in train], "validation_ids": [row["id"] for row in validation],
                "train_hash": fingerprint(train), "validation_hash": fingerprint(validation)}
    return train, validation, metadata
