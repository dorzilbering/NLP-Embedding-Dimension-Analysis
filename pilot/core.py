"""Small, model-independent data, reduction, and scoring utilities."""
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def canonical(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def select_data(dataset, train_count, eval_count, seed):
    """Exclude all validation/test sentences from PCA training candidates."""
    if train_count <= 768 or eval_count < 2:
        raise ValueError("Need >768 calibration sentences and >=2 validation pairs.")
    held_out = {canonical(row[key]) for split in ("validation", "test")
                for row in dataset[split] for key in ("sentence1", "sentence2")}
    unique = {}
    for row in dataset["train"]:
        for key in ("sentence1", "sentence2"):
            text = row[key]
            normalized = canonical(text)
            if normalized and normalized not in held_out:
                unique.setdefault(normalized, text)
    candidates = list(unique.values())
    if len(candidates) < train_count or len(dataset["validation"]) < eval_count:
        raise ValueError("Requested sample count exceeds eligible data.")
    rng = np.random.default_rng(seed)
    train = [candidates[i] for i in rng.permutation(len(candidates))[:train_count]]
    indices = sorted(rng.choice(len(dataset["validation"]), eval_count, replace=False).tolist())
    pairs = [dict(dataset["validation"][i]) for i in indices]
    if {canonical(t) for t in train} & held_out:
        raise ValueError("PCA calibration overlaps held-out data.")
    return train, pairs, indices


def checked(array, rows=None, dim=None):
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or not len(array) or (rows is not None and len(array) != rows):
        raise ValueError("Invalid embedding row shape.")
    if dim is not None and array.shape[1] != dim:
        raise ValueError(f"Expected width {dim}, got {array.shape[1]}.")
    if not np.isfinite(array).all():
        raise ValueError("NaN/Inf in embeddings.")
    return array


def normalize(array):
    array = checked(array)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Zero-length embedding.")
    return array / norms


def fit_pca(train, max_dim=768, seed=42):
    from sklearn.decomposition import PCA
    train = normalize(train)
    if max_dim >= min(train.shape):
        raise ValueError("Insufficient calibration rows/features for PCA.")
    pca = PCA(n_components=max_dim, whiten=False, svd_solver="randomized", random_state=seed)
    pca.fit(train)
    if pca.singular_values_[-1] <= pca.singular_values_[0] * 1e-6:
        raise ValueError("Calibration embeddings have insufficient effective rank.")
    return pca


def project(array, pca, dimension):
    if not 0 < dimension <= pca.components_.shape[0]:
        raise ValueError("Requested dimension exceeds fitted PCA basis.")
    # Prefix selection is appropriate AFTER projection into ordered PCA coordinates.
    reduced = (normalize(array) - pca.mean_) @ pca.components_[:dimension].T
    return normalize(checked(reduced, rows=len(array), dim=dimension))


def score_pairs(left, right, gold):
    from scipy.stats import spearmanr
    left, right = normalize(left), normalize(right)
    if left.shape != right.shape:
        raise ValueError("Sentence-pair embedding shapes differ.")
    gold = np.asarray(gold, dtype=np.float64)
    if gold.shape != (len(left),) or not np.isfinite(gold).all():
        raise ValueError("Invalid gold scores.")
    similarities = np.sum(left * right, axis=1)
    if np.ptp(similarities) == 0 or np.ptp(gold) == 0:
        raise ValueError("Undefined Spearman correlation (constant scores).")
    score = float(spearmanr(similarities, gold).statistic)
    if not np.isfinite(score):
        raise ValueError("Undefined Spearman correlation (constant scores?).")
    return score, similarities


def cached_embeddings(directory, identity, texts, encode):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    key = digest({"identity": identity, "texts": texts})
    path = directory / f"{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["key"]) != key:
                raise ValueError("Cache identity mismatch.")
            array = checked(saved["embeddings"], len(texts), 3072)
            if str(saved["checksum"]) != hashlib.sha256(array.tobytes()).hexdigest():
                raise ValueError("Cache checksum mismatch.")
        return array, True
    array = checked(encode(texts), len(texts), 3072)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, embeddings=array, key=key,
                 checksum=hashlib.sha256(array.tobytes()).hexdigest())
    temporary.replace(path)
    return array, False
