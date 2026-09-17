"""Build one fixed training-only calibration manifest shared by every model/task.

The manifest contains exactly 3072 deterministic, uniquely identified records drawn
from official Banking77 train and STSBenchmark train. Held-out/evaluation text is
excluded. Every model embeds this same manifest and fits one model-specific PCA basis.
"""
import argparse, hashlib, json
from pathlib import Path
from pilot.task_data import fingerprint, load_bundle, text_key
from prepare_banking77 import PARQUET_REVISION

CALIBRATION_SIZE = 3072


def evaluation_keys(paths):
    keys = set()
    for task, path in paths:
        if not path:
            continue
        b = load_bundle(Path(path), task)
        if task == "Banking77": rows = b["evaluation"]
        elif task == "SciFact": rows = b["queries"] + b["corpus"]
        elif task == "Arxiv-Clustering": rows = [{"text": t} for g in b["sets"] for t in g["sentences"]]
        else: rows = []
        keys.update(text_key(r["text"]) for r in rows)
    return keys


def _rank(record):
    return hashlib.sha256((record["id"] + "\0" + text_key(record["text"])).encode("utf-8")).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--banking", type=Path)
    p.add_argument("--scifact", type=Path)
    p.add_argument("--arxiv", type=Path)
    a = p.parse_args(argv)
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    bank_rev = api.dataset_info("PolyAI/banking77", revision=PARQUET_REVISION).sha
    sts_rev = api.dataset_info("mteb/stsbenchmark-sts").sha
    # Pin the same immutable, script-free Banking77 Parquet revision used by prepare_banking77.py.
    bank_base = f"hf://datasets/PolyAI/banking77@{bank_rev}/data"
    bank = load_dataset("parquet", data_files={
        "train": f"{bank_base}/train-00000-of-00001.parquet",
        "test": f"{bank_base}/test-00000-of-00001.parquet",
    })
    sts = load_dataset("mteb/stsbenchmark-sts", revision=sts_rev)
    excluded = evaluation_keys((("Banking77", a.banking), ("SciFact", a.scifact), ("Arxiv-Clustering", a.arxiv)))
    excluded.update(text_key(x["text"]) for x in bank["test"])
    for split in ("validation", "test"):
        for r in sts[split]:
            excluded.add(text_key(r["sentence1"])); excluded.add(text_key(r["sentence2"]))
    candidates = []
    for i, r in enumerate(bank["train"]): candidates.append({"id": f"banking77:train:{i}", "text": r["text"], "source": "Banking77"})
    for i, r in enumerate(sts["train"]):
        candidates.append({"id": f"stsb:train:{i}:a", "text": r["sentence1"], "source": "STSB"})
        candidates.append({"id": f"stsb:train:{i}:b", "text": r["sentence2"], "source": "STSB"})
    unique = {}
    for record in candidates:
        key = text_key(record["text"])
        if key and key not in excluded: unique.setdefault(key, record)
    eligible = sorted(unique.values(), key=_rank)
    if len(eligible) < CALIBRATION_SIZE:
        raise ValueError(f"Need at least {CALIBRATION_SIZE} eligible calibration texts; got {len(eligible)}")
    texts = eligible[:CALIBRATION_SIZE]
    bundle = {"schema_version": 1, "role": "shared_pca_calibration", "texts": texts,
              "selection": {"method": "sha256 deterministic rank", "size": CALIBRATION_SIZE},
              "sources": [{"dataset_id":"PolyAI/banking77","revision":bank_rev,"split":"train"},
                          {"dataset_id":"mteb/stsbenchmark-sts","revision":sts_rev,"split":"train"}],
              "excluded_normalized_count": len(excluded)}
    bundle["manifest_hash"] = fingerprint(texts)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved":str(a.output),"eligible_pool":len(eligible),"selected":len(texts),"manifest_hash":bundle["manifest_hash"]}, indent=2))

if __name__ == "__main__": main()
