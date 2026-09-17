"""Assignment model/task metadata shared by runners and preparation scripts."""

MODEL_SPECS = {
    "Qwen3-8B": {"native_dimension": 4096, "dimensions": (4096, 2048, 1024, 512, 256),
                 "model_id": "Qwen/Qwen3-8B", "t4_quantization": "4bit"},
    "Gemma3-4B": {"native_dimension": 2560, "dimensions": (2560, 1280, 640, 320),
                  "model_id": "google/gemma-3-4b-it", "t4_quantization": None},
    "Llama3.2-3B": {"native_dimension": 3072, "dimensions": (3072, 1536, 768, 384),
                    "model_id": "meta-llama/Llama-3.2-3B-Instruct", "t4_quantization": None},
    "Phi4-mini": {"native_dimension": 3072, "dimensions": (3072, 1536, 768, 384),
                  "model_id": "microsoft/Phi-4-mini-instruct", "t4_quantization": None},
    "Mistral-7B": {"native_dimension": 4096, "dimensions": (4096, 2048, 1024, 512),
                   "model_id": "mistralai/Mistral-7B-Instruct-v0.3", "t4_quantization": "4bit"},
}
TASK_SPECS = {
    "SciFact": {"type": "retrieval", "dataset_id": "mteb/scifact"},
    "Banking77": {"type": "classification", "dataset_id": "PolyAI/banking77"},
    "Arxiv-Clustering": {"type": "clustering", "dataset_id": "mteb/arxiv-clustering-s2s"},
    "STSB": {"type": "sts", "dataset_id": "mteb/stsbenchmark-sts"},
}
LOADING_STRATEGY = "t4-auto"
LEGACY_DIMENSIONS = (3072, 768, 384)


def validate_dimensions(dimensions, native_dimension):
    dimensions = tuple(dimensions)
    if not dimensions or any(type(d) is not int or d <= 0 or d > native_dimension for d in dimensions):
        raise ValueError(f"Dimensions must be integers in [1, {native_dimension}].")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate dimensions are not allowed.")
    return dimensions


def make_plan(model="Phi4-mini", task="STSB", dimensions=None, train_count=2000,
              eval_count=300, batch_size=4, max_length=256, loading_strategy=LOADING_STRATEGY):
    """Compatibility plan for the historical Phi/STSB pilot runner.

    Final assignment execution uses task_config.task_plan/run_experiment.py. Keeping this
    function prevents the validated pilot and its tests from breaking during the refactor.
    """
    if model not in MODEL_SPECS or task not in TASK_SPECS:
        raise ValueError("Unknown model or task.")
    spec = MODEL_SPECS[model]
    defaults = LEGACY_DIMENSIONS if model == "Phi4-mini" else spec["dimensions"]
    dims = validate_dimensions(defaults if dimensions is None else dimensions, spec["native_dimension"])
    if any(d not in spec["dimensions"] for d in dims):
        raise ValueError("Pilot supports assignment-required dimensions only.")
    components = max((d for d in dims if d < spec["native_dimension"]), default=0)
    if train_count < 1 or train_count <= components:
        raise ValueError(f"Need more than {components} eligible training sentences for centered PCA.")
    if eval_count < 2 or batch_size < 1 or not 8 <= max_length <= 512:
        raise ValueError("Need >=2 evaluation pairs, batch-size >=1, and max-length in [8, 512].")
    if loading_strategy != LOADING_STRATEGY:
        raise ValueError(f"Pilot loading strategy must be {LOADING_STRATEGY}.")
    blockers = []
    if model != "Phi4-mini" or task != "STSB":
        blockers.append("run_pilot.py is intentionally limited to the historical Phi4-mini/STSB pilot; use run_experiment.py for final assignment runs.")
    return {"model": model, "model_id": spec["model_id"], "task": task,
            "dataset_id": TASK_SPECS[task]["dataset_id"], "native_dimension": spec["native_dimension"],
            "dimensions": list(dims), "pca_components": components,
            "train_sentences": train_count, "validation_pairs": eval_count,
            "batch_size": batch_size, "max_length": max_length,
            "loading_strategy": loading_strategy, "frozen": True,
            "quantization": spec.get("t4_quantization"), "executable": not blockers,
            "blockers": blockers, "validation_scope": "legacy pilot compatibility only"}


def assignment_matrix():
    return {"models": MODEL_SPECS, "tasks": TASK_SPECS,
            "required_evaluation_configurations": sum(len(s["dimensions"]) for s in MODEL_SPECS.values()) * len(TASK_SPECS),
            "reduction": "one training-only calibration PCA basis per model, reused across all tasks/dimensions",
            "note": "Qwen3-8B and Mistral-7B use 4-bit weight loading on a 16GB T4; this must be reported as an execution constraint."}
