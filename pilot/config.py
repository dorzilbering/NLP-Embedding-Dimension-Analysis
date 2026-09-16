"""Offline assignment metadata and plan validation; not a multi-model runner."""

MODEL_SPECS = {
    "Qwen3-8B": {"native_dimension": 4096, "dimensions": (4096, 2048, 1024, 512, 256), "model_id": None},
    "Gemma3-4B": {"native_dimension": 2560, "dimensions": (2560, 1280, 640, 320), "model_id": None},
    "Llama3.2-3B": {"native_dimension": 3072, "dimensions": (3072, 1536, 768, 384), "model_id": None},
    "Phi4-mini": {"native_dimension": 3072, "dimensions": (3072, 1536, 768, 384),
                  "model_id": "microsoft/Phi-4-mini-instruct"},
    "Mistral-7B": {"native_dimension": 4096, "dimensions": (4096, 2048, 1024, 512), "model_id": None},
}
TASK_SPECS = {
    "SciFact": {"type": "retrieval", "dataset_id": None, "status": "retrieval adapter and revision unverified"},
    "Banking77": {"type": "classification", "dataset_id": None, "status": "classification protocol/version unverified"},
    "Arxiv-Clustering": {"type": "clustering", "dataset_id": None, "status": "S2S/P2P and task version unresolved"},
    "STSB": {"type": "sts", "dataset_id": "mteb/stsbenchmark-sts", "status": "English validation pilot implemented"},
}
LEGACY_DIMENSIONS = (3072, 768, 384)
LOADING_STRATEGY = "native-auto-16bit"


def validate_dimensions(dimensions, native_dimension):
    dimensions = tuple(dimensions)
    if not dimensions or any(type(d) is not int or d <= 0 or d > native_dimension for d in dimensions):
        raise ValueError(f"Dimensions must be integers in [1, {native_dimension}].")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate dimensions are not allowed.")
    return dimensions


def make_plan(model="Phi4-mini", task="STSB", dimensions=None, train_count=2000,
              eval_count=300, batch_size=4, max_length=256, loading_strategy=LOADING_STRATEGY):
    if model not in MODEL_SPECS or task not in TASK_SPECS:
        raise ValueError("Unknown model or task.")
    spec = MODEL_SPECS[model]
    defaults = LEGACY_DIMENSIONS if model == "Phi4-mini" else spec["dimensions"]
    dimensions = validate_dimensions(defaults if dimensions is None else dimensions, spec["native_dimension"])
    if any(d not in spec["dimensions"] for d in dimensions):
        raise ValueError("This step supports assignment-required dimensions only.")
    reduced = [d for d in dimensions if d < spec["native_dimension"]]
    components = max(reduced, default=0)
    if train_count < 1 or train_count <= components:
        raise ValueError(f"Need more than {components} eligible training/reference sentences for centered PCA.")
    if eval_count < 2 or batch_size < 1 or not 8 <= max_length <= 512:
        raise ValueError("Need >=2 evaluation pairs, batch-size >=1, and max-length in [8, 512].")
    if loading_strategy != LOADING_STRATEGY:
        raise ValueError("Only native-auto-16bit is implemented; no automatic quantization/offload fallback.")
    blockers = []
    if spec["model_id"] is None:
        blockers.append("Hugging Face checkpoint/revision and model adapter require verification.")
    if model != "Phi4-mini":
        blockers.append("Model-specific loading strategy and T4 memory feasibility are not validated.")
    if task != "STSB":
        blockers.append(TASK_SPECS[task]["status"])
        blockers.append("Leakage-safe reference data and evaluator are not implemented for this task.")
    return {"model": model, "model_id": spec["model_id"], "task": task,
            "dataset_id": TASK_SPECS[task]["dataset_id"], "native_dimension": spec["native_dimension"],
            "dimensions": list(dimensions), "pca_components": components,
            "train_sentences": train_count, "validation_pairs": eval_count,
            "batch_size": batch_size, "max_length": max_length,
            "loading_strategy": loading_strategy, "frozen": True, "quantization": None,
            "executable": not blockers, "blockers": blockers,
            "validation_scope": "static only; eligible data count, rank, GPU memory and access checked at execution"}


def assignment_matrix():
    return {"models": MODEL_SPECS, "tasks": TASK_SPECS,
            "required_evaluation_configurations": sum(len(s["dimensions"]) for s in MODEL_SPECS.values()) * len(TASK_SPECS),
            "implemented_pair": ["Phi4-mini", "STSB"],
            "note": "Metadata is a plan, not evidence of executed experiments; null IDs need verification."}
