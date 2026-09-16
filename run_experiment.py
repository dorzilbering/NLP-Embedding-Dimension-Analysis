"""Task-aware Phi experiments. Dry-run uses only the standard library."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time

from pilot.config import LOADING_STRATEGY, MODEL_SPECS, TASK_SPECS
from pilot.task_config import task_plan
from pilot.task_data import load_bundle
from run_pilot import MODEL_REVISION


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_SPECS, required=True)
    parser.add_argument("--model", choices=MODEL_SPECS, default="Phi4-mini")
    parser.add_argument("--dimensions", nargs="+", type=int)
    parser.add_argument("--data", type=Path, help="Reviewed local JSON bundle; new tasks only.")
    parser.add_argument("--clusters", type=int, help="Predefined/training-only K; Arxiv-Clustering only.")
    parser.add_argument("--train-sentences", type=int, default=2000)
    parser.add_argument("--validation-pairs", type=int, default=300, help="STSB only.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loading-strategy", default=LOADING_STRATEGY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache/native"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.validation_pairs < 2 or (args.task != "STSB" and args.validation_pairs != 300):
        parser.error("--validation-pairs is STSB-only and must be >=2.")
    bundle = load_bundle(args.data, args.task) if args.data else None
    plan = task_plan(args.task, args.model, args.dimensions, args.train_sentences, args.seed,
                     args.batch_size, args.max_length, args.clusters, bundle, args.loading_strategy)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if not plan["executable"]:
        parser.error("Execution blocked: " + " ".join(plan["blockers"]))
    output = args.output or Path("results") / ("phi_" + args.task.lower().replace("-", "_"))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("Output directory is not empty. Choose a fresh --output directory.")
    if args.task == "STSB":
        import run_pilot
        forwarded = ["run_pilot.py", "--dimensions", *map(str, plan["dimensions"]),
                     "--train-sentences", str(args.train_sentences), "--validation-pairs", str(args.validation_pairs),
                     "--batch-size", str(args.batch_size), "--max-length", str(args.max_length),
                     "--seed", str(args.seed), "--output", str(output), "--cache", str(args.cache)]
        original = sys.argv
        try:
            sys.argv = forwarded
            run_pilot.main()
        finally:
            sys.argv = original
        from pilot.task_runner import sts_metrics, write_metrics
        write_metrics(output / "metrics.csv", sts_metrics(json.loads((output / "results.json").read_text(encoding="utf-8"))))
        return

    from pilot.model import PhiEncoder, MODEL_ID, check_gpu
    hardware = check_gpu()  # Refuse CPU execution before model loading/download.
    import numpy as np
    import torch
    from pilot.core import cached_embeddings
    from pilot.task_runner import evaluate_bundle, save_task_outputs
    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = PhiEncoder(MODEL_REVISION, args.batch_size, args.max_length)
    batching_checks = encoder.validate_batching()
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "datasets", "numpy", "scipy", "scikit-learn", "matplotlib",
                 "huggingface-hub", "accelerate", "safetensors")}
    identity = {"schema": 1, "model": MODEL_ID, "revision": MODEL_REVISION,
                "pooling": "final_normalized_hidden_last_attention_mask_valid_token",
                "format": "plain_text_tokenizer_default_special_tokens_no_chat_no_added_eos",
                "max_length": args.max_length, "dtype": str(encoder.dtype), "padding": "right",
                "attention": "sdpa", "batch_size": args.batch_size, "packages": packages, "gpu": hardware["gpu"]}
    cache_hits, truncation = [], []

    def encode(texts):
        vectors, hit = cached_embeddings(args.cache, identity, texts, encoder.encode)
        cache_hits.append(hit)
        lengths = [len(ids) for ids in encoder.tokenizer(texts, truncation=False)["input_ids"]]
        truncation.append({"texts": len(texts), "truncated": sum(n > args.max_length for n in lengths)})
        return vectors

    rows, predictions, metadata, pca = evaluate_bundle(bundle, plan, encode)
    torch.cuda.synchronize()
    metadata.update(representation=identity, hardware=hardware, packages=packages,
                    python=platform.python_version(), platform=platform.platform(), batching_checks=batching_checks,
                    cache_hits_in_group_order=cache_hits, truncation_in_group_order=truncation,
                    group_order=["fit", "queries", "corpus"] if args.task == "SciFact" else ["fit", "evaluation"],
                    seconds_before_export=time.perf_counter() - started,
                    peak_allocated_vram_bytes=torch.cuda.max_memory_allocated())
    save_task_outputs(output, rows, predictions, metadata, pca)
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(f"Experiment stopped: {error}") from error
