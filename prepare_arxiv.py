"""Export official ArxivClusteringS2S test sets; no language-model loading."""
import argparse
import json
from pathlib import Path
import re

from pilot.arxiv import DATASET_ID, TASK_NAME, build_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", type=Path, default=Path("data/arxiv_s2s.json"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"dataset": DATASET_ID, "task": TASK_NAME, "split": "test",
                          "structure": "one sentences/labels clustering set per row",
                          "reduction": "shared training-only calibration PCA; never fitted on Arxiv test sets"}, indent=2))
        return
    if args.output.exists():
        parser.error("Output already exists; use a fresh path.")
    from huggingface_hub import HfApi
    from datasets import get_dataset_config_names, load_dataset
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Cannot resolve immutable dataset revision.")
    configs = get_dataset_config_names(DATASET_ID, revision=revision)
    if len(configs) != 1:
        raise ValueError("Unexpected configuration structure; review dataset before execution.")
    dataset = load_dataset(DATASET_ID, name=configs[0], revision=revision)
    if set(dataset) != {"test"} or set(dataset["test"].column_names) != {"sentences", "labels"}:
        raise ValueError("Expected only official test sets with sentences/labels columns.")
    bundle = build_bundle(dataset["test"], revision, configs[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"saved": str(args.output), "revision": revision, "sets": len(bundle["sets"]),
                      "reduction": "shared calibration PCA"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(f"Arxiv preparation stopped: {error}") from error
