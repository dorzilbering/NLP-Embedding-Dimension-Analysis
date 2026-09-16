"""Build one fixed training-only calibration manifest shared by every model/task.

Sources: official Banking77 train + STSBenchmark train. Exact normalized texts that
occur in any supplied evaluation bundle are excluded. The exported text manifest is
model-independent; each model fits its own PCA basis on embeddings of these texts.
"""
import argparse, json
from pathlib import Path
from pilot.task_data import fingerprint, load_bundle, text_key


def evaluation_keys(paths):
    keys = set()
    for task, path in paths:
        if not path:
            continue
        b = load_bundle(Path(path), task)
        if task == "Banking77": rows = b["evaluation"]
        elif task == "SciFact": rows = b["queries"] + b["corpus"]
        elif task == "Arxiv-Clustering":
            rows = [{"text": t} for g in b["sets"] for t in g["sentences"]]
        else: rows = []
        keys.update(text_key(r["text"]) for r in rows)
    return keys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--banking", type=Path)
    p.add_argument("--scifact", type=Path)
    p.add_argument("--arxiv", type=Path)
    a = p.parse_args()
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi()
    bank_rev = api.dataset_info("PolyAI/banking77").sha
    sts_rev = api.dataset_info("mteb/stsbenchmark-sts").sha
    bank = load_dataset("PolyAI/banking77", revision=bank_rev, trust_remote_code=True)
    sts = load_dataset("mteb/stsbenchmark-sts", revision=sts_rev)
    excluded = evaluation_keys((("Banking77", a.banking), ("SciFact", a.scifact), ("Arxiv-Clustering", a.arxiv)))
    # Always exclude official Banking test and STS validation/test, even if no local bundle was supplied.
    excluded.update(text_key(x["text"]) for x in bank["test"])
    for split in ("validation", "test"):
        for r in sts[split]:
            excluded.add(text_key(r["sentence1"])); excluded.add(text_key(r["sentence2"]))
    candidates = [r["text"] for r in bank["train"]]
    for r in sts["train"]:
        candidates.extend((r["sentence1"], r["sentence2"]))
    seen, texts = set(), []
    for text in candidates:
        key = text_key(text)
        if key and key not in seen and key not in excluded:
            seen.add(key); texts.append(text)
    if len(texts) <= 4096:
        raise ValueError(f"Need >4096 eligible calibration texts; got {len(texts)}")
    bundle = {"schema_version": 1, "role": "shared_pca_calibration", "texts": texts,
              "sources": [{"dataset_id":"PolyAI/banking77","revision":bank_rev,"split":"train"},
                          {"dataset_id":"mteb/stsbenchmark-sts","revision":sts_rev,"split":"train"}],
              "excluded_normalized_count": len(excluded)}
    bundle["manifest_hash"] = fingerprint(texts)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"saved":str(a.output),"eligible":len(texts),"manifest_hash":bundle["manifest_hash"]}, indent=2))

if __name__ == "__main__": main()
