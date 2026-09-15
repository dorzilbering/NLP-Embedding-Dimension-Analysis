"""Run only the frozen Phi / English STS validation pilot."""
import argparse
import csv
import importlib.metadata
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from pilot.core import cached_embeddings, digest, fit_pca, normalize, project, score_pairs, select_data
from pilot.model import MODEL_ID, PhiEncoder, check_gpu

DATASET_ID = "mteb/stsbenchmark-sts"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
DATASET_REVISION = "96943a16ea6a35129e253c659081cb59daf81b30"


def save_outputs(output, rows, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "results.json").write_text(json.dumps({"metadata": metadata, "scores": rows}, indent=2), encoding="utf-8")
    ordered = sorted(rows, key=lambda row: row["dimension"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([r["dimension"] for r in ordered], [r["cosine_spearman"] for r in ordered], "o-")
    ax.set(xlabel="Embedding dimension", ylabel="Cosine Spearman correlation",
           title="Phi-4-mini: English STS validation pilot", xticks=[384, 768, 3072])
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.01, "3072: native; 768/384: training-only PCA. Subset evaluation.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(output / "dimension_vs_spearman.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/pilot"))
    parser.add_argument("--cache", type=Path, default=Path("cache/native"))
    parser.add_argument("--train-sentences", type=int, default=2000)
    parser.add_argument("--validation-pairs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-revision", default=MODEL_REVISION, help="Pinned model SHA; overrides are resolved before loading.")
    parser.add_argument("--dataset-revision", default=DATASET_REVISION, help="Pinned dataset SHA; overrides are resolved before loading.")
    parser.add_argument("--check-hardware", action="store_true", help="No downloads; report CUDA readiness and exit.")
    args = parser.parse_args()
    hardware = check_gpu()  # Before any network access or dataset/model download.
    print(json.dumps(hardware, indent=2), flush=True)
    if args.check_hardware:
        return
    if args.batch_size < 1 or args.max_length < 8 or args.max_length > 512:
        parser.error("Pilot requires batch-size >=1 and max-length between 8 and 512.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty. Choose a new --output directory.")
    import torch
    from datasets import load_dataset
    from huggingface_hub import HfApi
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    api = HfApi()
    model_revision = api.model_info(MODEL_ID, revision=args.model_revision).sha
    dataset_revision = api.dataset_info(DATASET_ID, revision=args.dataset_revision).sha
    dataset = load_dataset(DATASET_ID, revision=dataset_revision)
    for split in ("train", "validation", "test"):
        if split not in dataset or not {"sentence1", "sentence2", "score"}.issubset(dataset[split].column_names):
            raise ValueError(f"Unexpected English STS schema for {split}.")
    train, pairs, indices = select_data(dataset, args.train_sentences, args.validation_pairs, args.seed)
    texts = list(dict.fromkeys([p[key] for p in pairs for key in ("sentence1", "sentence2")]))
    lookup = {text: i for i, text in enumerate(texts)}
    print(f"Data validated: {len(train)} PCA sentences, {len(pairs)} validation pairs. Loading frozen Phi.", flush=True)
    loading = time.perf_counter()
    encoder = PhiEncoder(model_revision, args.batch_size, args.max_length)
    load_seconds = time.perf_counter() - loading
    batching_checks = encoder.validate_batching()
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "numpy", "scipy", "scikit-learn", "matplotlib",
                 "huggingface-hub", "accelerate", "safetensors")}
    identity = {"schema": 1, "model": MODEL_ID, "revision": model_revision,
                "pooling": "final_normalized_hidden_last_attention_mask_valid_token",
                "format": "plain_text_tokenizer_default_special_tokens_no_chat_no_added_eos",
                "max_length": args.max_length, "dtype": str(encoder.dtype), "padding": "right",
                "attention": "sdpa", "batch_size": args.batch_size, "packages": packages,
                "gpu": hardware["gpu"]}
    encoding = time.perf_counter()
    train_x, train_hit = cached_embeddings(args.cache, identity, train, encoder.encode)
    eval_x, eval_hit = cached_embeddings(args.cache, identity, texts, encoder.encode)
    torch.cuda.synchronize()
    encoding_seconds = time.perf_counter() - encoding
    fitting = time.perf_counter()
    pca = fit_pca(train_x, max_dim=768, seed=args.seed)
    pca_seconds = time.perf_counter() - fitting
    left_ids = [lookup[p["sentence1"]] for p in pairs]
    right_ids = [lookup[p["sentence2"]] for p in pairs]
    gold = [p["score"] for p in pairs]
    rows, predictions = [], {}
    for dimension in (3072, 768, 384):
        transformed = normalize(eval_x) if dimension == 3072 else project(eval_x, pca, dimension)
        score, similarities = score_pairs(transformed[left_ids], transformed[right_ids], gold)
        rows.append({"model": MODEL_ID, "dataset": DATASET_ID, "split": "validation",
                     "pairs": len(pairs), "dimension": dimension,
                     "reduction": "native" if dimension == 3072 else "pca",
                     "metric": "cosine_spearman", "cosine_spearman": score, "seed": args.seed})
        predictions[str(dimension)] = similarities.tolist()
        print(f"dimension={dimension}, cosine_spearman={score:.6f}", flush=True)
    truncation = {}
    for name, collection in (("calibration", train), ("validation_unique", texts)):
        lengths = [len(ids) for ids in encoder.tokenizer(collection, truncation=False)["input_ids"]]
        truncation[name] = {"texts": len(lengths), "truncated": sum(n > args.max_length for n in lengths)}
    metadata = {"schema_version": 1, "model_revision": model_revision, "dataset_revision": dataset_revision,
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "representation": identity, "hardware": hardware, "python": platform.python_version(),
                "platform": platform.platform(), "packages": packages, "batching_checks": batching_checks,
                "calibration_text_hashes": [digest(t) for t in train], "calibration_manifest_hash": digest(train),
                "validation_indices": indices, "validation_text_hash": digest(texts),
                "leakage_check": "NFKC/casefold/whitespace normalized exclusion against ALL validation/test sentences",
                "pca": {"fit_split": "train", "whiten": False, "pre_normalize": True,
                        "components": 768, "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum())},
                "native_centered": False, "score_scale": "correlation [-1, 1]",
                "protocol": "MTEB English STS data and cosine-Spearman metric; custom validation subset, not full MTEB run",
                "cache_hits": {"train": train_hit, "validation": eval_hit}, "truncation": truncation,
                "seconds": {"load": load_seconds, "encode_or_cache": encoding_seconds,
                            "pca_fit": pca_seconds, "total_before_export": time.perf_counter()-started},
                "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated()}
    save_outputs(args.output, rows, metadata)
    np.savez(args.output / "pca.npz", mean=pca.mean_, components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    (args.output / "predictions.json").write_text(json.dumps({"validation_indices": indices, "gold": gold,
                                                              "cosine_similarities": predictions}, indent=2))
    print(f"Saved pilot outputs to {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, ImportError) as error:
        raise SystemExit(f"Pilot stopped: {error}") from error
