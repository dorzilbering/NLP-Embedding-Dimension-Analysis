"""Prepare official Banking77 data only; never imports/loads a language model."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re

from pilot.banking77 import DATASET_ID, NUM_CLASSES, SPLIT_COUNTS, build_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Resolve this ref to an immutable SHA before loading.")
    parser.add_argument("--configuration", help="If omitted, discover and require exactly one configuration.")
    parser.add_argument("--output", type=Path, default=Path("data/banking77_official.json"))
    parser.add_argument("--dry-run", action="store_true", help="Print preparation contract; no downloads or file writes.")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"dataset": DATASET_ID, "requested_revision": args.revision,
                          "configuration": args.configuration or "discover single configuration at resolved revision",
                          "official_split_counts": SPLIT_COUNTS, "classes": NUM_CLASSES,
                          "fields": ["text", "label"], "output": str(args.output),
                          "network_access": False, "dataset_contents_verified": False}, indent=2))
        return
    if args.output.exists():
        parser.error("Output already exists; choose a new path. Existing bundles are never overwritten.")
    from huggingface_hub import HfApi
    from datasets import get_dataset_config_names, load_dataset
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Could not resolve an immutable dataset commit SHA.")
    configurations = get_dataset_config_names(DATASET_ID, revision=revision)
    configuration = args.configuration
    if configuration is None:
        if len(configurations) != 1:
            raise ValueError(f"Choose --configuration from the verified available names: {configurations}")
        configuration = configurations[0]
    if configuration not in configurations:
        raise ValueError(f"Unknown configuration; available names: {configurations}")
    dataset = load_dataset(DATASET_ID, name=configuration, revision=revision)
    bundle = build_bundle(dataset, revision, configuration)
    bundle["preparation"] = {"packages": {name: importlib.metadata.version(name)
                                         for name in ("datasets", "huggingface-hub")}}
    payload = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    print(json.dumps({"saved": str(args.output), "revision": revision, "configuration": configuration,
                      "counts": SPLIT_COUNTS, "train_unique_texts": bundle["source"]["train_unique_texts"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(f"Banking77 preparation stopped: {error}") from error
