"""Immutable run specification. Importing this module never loads a model."""
from dataclasses import asdict, dataclass
import math
import re

MODEL_IDS = {
    "Phi4-mini": "microsoft/Phi-4-mini-instruct",
    "Llama3.2-3B": "meta-llama/Llama-3.2-3B",
}
DIMENSIONS = (3072, 1536, 768)


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    revision: str
    dimension: int
    seed: int = 42
    epochs: int = 5
    batch_size: int = 1
    accumulation_steps: int = 32
    max_length: int = 256
    embedding_lr: float = 1e-4
    classifier_lr: float = 1e-3
    warmup_fraction: float = 0.1
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    validation_fraction: float = 0.1
    validation_seed: int = 42
    checkpoint_every_steps: int = 50

    def __post_init__(self):
        if self.model not in MODEL_IDS:
            raise ValueError("Unsupported model.")
        if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("Provide an immutable lowercase 40-character model commit SHA.")
        if type(self.dimension) is not int or self.dimension not in DIMENSIONS:
            raise ValueError("Internal token dimension must be 3072, 1536, or 768.")
        for name in ("epochs", "batch_size", "accumulation_steps", "max_length", "checkpoint_every_steps"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if type(self.seed) is not int or self.seed < 0 or not 8 <= self.max_length <= 512:
            raise ValueError("Invalid seed or maximum length.")
        for name in ("embedding_lr", "classifier_lr", "max_grad_norm"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(self.warmup_fraction) or not 0 <= self.warmup_fraction < 1:
            raise ValueError("warmup_fraction must lie in [0, 1).")
        if type(self.gradient_checkpointing) is not bool:
            raise ValueError("gradient_checkpointing must be boolean.")
        if not math.isfinite(self.validation_fraction) or not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must lie in (0, 1).")
        if type(self.validation_seed) is not int or self.validation_seed < 0:
            raise ValueError("validation_seed must be a nonnegative integer.")

    @property
    def model_id(self):
        return MODEL_IDS[self.model]

    def to_dict(self):
        return asdict(self)
