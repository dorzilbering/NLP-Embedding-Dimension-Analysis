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


def validate_dimensions(dimensions, native_dimension):
    dimensions = tuple(dimensions)
    if not dimensions or any(type(d) is not int or d <= 0 or d > native_dimension for d in dimensions):
        raise ValueError(f"Dimensions must be integers in [1, {native_dimension}].")
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate dimensions are not allowed.")
    return dimensions


def assignment_matrix():
    return {"models": MODEL_SPECS, "tasks": TASK_SPECS,
            "required_evaluation_configurations": sum(len(s["dimensions"]) for s in MODEL_SPECS.values()) * len(TASK_SPECS),
            "reduction": "one training-only calibration PCA basis per model, reused across all tasks/dimensions",
            "note": "Qwen3-8B and Mistral-7B use 4-bit weight loading on a 16GB T4; this must be reported as an execution constraint."}
