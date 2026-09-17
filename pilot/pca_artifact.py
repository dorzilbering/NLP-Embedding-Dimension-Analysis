"""Persistence and validation for the one shared PCA basis fitted per model."""
import hashlib, json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from pilot.core import checked, fit_pca


def artifact_identity(plan, model_revision, calibration, max_length):
    return {"model": plan["model"], "model_id": plan["model_id"], "model_revision": model_revision,
            "native_dimension": plan["native_dimension"], "components": plan["pca_components"],
            "seed": plan["seed"], "max_length": max_length, "quantization": plan.get("quantization"),
            "calibration_hash": calibration["manifest_hash"]}


def artifact_key(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def save_pca(path, pca, identity):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); key = artifact_key(identity)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as h:
        np.savez(h, mean=pca.mean_, components=pca.components_, explained_variance_ratio=pca.explained_variance_ratio_, key=key, identity=json.dumps(identity, sort_keys=True))
    tmp.replace(path)


def load_pca(path, identity):
    path = Path(path); expected = artifact_key(identity)
    with np.load(path, allow_pickle=False) as z:
        if str(z["key"]) != expected or json.loads(str(z["identity"])) != identity: raise ValueError("PCA artifact identity mismatch.")
        mean=np.asarray(z["mean"],dtype=np.float32); components=np.asarray(z["components"],dtype=np.float32); ratio=np.asarray(z["explained_variance_ratio"],dtype=np.float32)
    if mean.shape != (identity["native_dimension"],) or components.shape != (identity["components"], identity["native_dimension"]): raise ValueError("PCA artifact shape mismatch.")
    return SimpleNamespace(mean_=mean, components_=components, explained_variance_ratio_=ratio)


def get_or_fit_pca(path, identity, calibration, encode):
    path=Path(path)
    if path.exists(): return load_pca(path, identity), True
    records=calibration["texts"]; texts=[r["text"] if isinstance(r,dict) else r for r in records]
    x=checked(encode(texts),len(texts),identity["native_dimension"]); pca=fit_pca(x,identity["components"],identity["seed"]); save_pca(path,pca,identity); return pca,False
