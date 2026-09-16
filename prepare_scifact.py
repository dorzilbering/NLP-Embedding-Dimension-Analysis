"""Prepare MTEB SciFact test retrieval with eligible AllenAI non-test claim reference."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
from itertools import islice

from pilot.scifact import DATASET_ID, COUNTS, CONFIG_SPLITS, REFERENCE_ID, REFERENCE_COUNTS, build_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="main", help="Resolved to immutable dataset SHA before loading.")
    parser.add_argument("--reference-revision", default="main", help="Independent AllenAI dataset revision.")
    parser.add_argument("--output", type=Path, default=Path("data/scifact_official.json"))
    parser.add_argument("--dry-run", action="store_true", help="No downloads, model imports or file writes.")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"dataset": DATASET_ID, "requested_revision": args.revision,
                          "configurations": CONFIG_SPLITS, "expected_counts": COUNTS,
                          "pca_reference": {"dataset": REFERENCE_ID, "configuration": "claims",
                                            "splits": REFERENCE_COUNTS, "field": "claim",
                                            "requested_revision": args.reference_revision,
                                            "selection": "all eligible unique texts after retrieval-test/corpus exclusions"},
                          "limitation": "1711 source rows do not guarantee 1711 unique eligible texts or sufficient PCA rank.",
                          "dataset_contents_verified": False, "output": str(args.output)}, indent=2))
        return
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path. Existing bundles are never overwritten.")
    from huggingface_hub import HfApi
    from datasets import get_dataset_config_names, load_dataset
    revision = HfApi().dataset_info(DATASET_ID, revision=args.revision).sha
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Could not resolve immutable dataset revision.")
    reference_revision = HfApi().dataset_info(REFERENCE_ID, revision=args.reference_revision).sha
    if not isinstance(reference_revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", reference_revision):
        raise ValueError("Could not resolve immutable AllenAI reference revision.")
    if set(get_dataset_config_names(DATASET_ID, revision=revision)) != set(CONFIG_SPLITS):
        raise ValueError("Unexpected SciFact configurations; review source before proceeding.")
    loaded = {}
    for config, splits in CONFIG_SPLITS.items():
        dataset = load_dataset(DATASET_ID, name=config, revision=revision)
        if set(dataset) != set(splits):
            raise ValueError(f"Unexpected split structure for {config}.")
        loaded[config] = dataset
    # Streaming selects/iterates only these two splits. Never request/iterate AllenAI test.
    reference_rows = {}
    for split, expected in REFERENCE_COUNTS.items():
        stream = load_dataset(REFERENCE_ID, name="claims", split=split, revision=reference_revision,
                              streaming=True, trust_remote_code=True)
        reference_rows[split] = list(islice(stream, expected + 1))
    bundle = build_bundle(loaded["corpus"]["corpus"], loaded["queries"]["queries"],
                          loaded["default"]["train"], loaded["default"]["test"], revision,
                          reference_rows, reference_revision)
    bundle["preparation"] = {"packages": {name: importlib.metadata.version(name)
                                         for name in ("datasets", "huggingface-hub")}}
    payload = json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(payload)
    eligible = len(bundle["reference"])
    print(json.dumps({"saved": str(args.output), "revision": revision, "reference_revision": reference_revision,
                      "reference_rows": len(bundle["reference"]), "evaluation_queries": len(bundle["queries"]),
                      "corpus_rows": len(bundle["corpus"]),
                      "reference_excluded_rows": len(bundle["fit_source"]["excluded_rows"]),
                      "phi_1536_sample_count_sufficient": eligible > 1536,
                      "embedding_rank_verified": False,
                      "blocker": "Too few eligible rows for centered 1536 PCA." if eligible <= 1536
                                 else "Actual embedding rank still requires validation at execution."}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, ImportError, OSError) as error:
        raise SystemExit(f"SciFact preparation stopped: {error}") from error
