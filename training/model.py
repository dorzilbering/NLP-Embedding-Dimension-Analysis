"""HF backbone adapters with gradient flow through frozen transformer layers."""
import torch
from torch import nn
from torch.nn import functional as F

from .embeddings import TrainableTokenEmbedding


def last_valid_pool(hidden, mask):
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not mask.bool().any(1).all():
        raise ValueError("Invalid hidden states or all-padding input.")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    last = positions.masked_fill(~mask.bool(), -1).max(1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


class Banking77Model(nn.Module):
    def __init__(self, backbone, dimension, gradient_checkpointing=False):
        super().__init__()
        original = backbone.get_input_embeddings()
        hidden = backbone.config.hidden_size
        if original.weight.shape[1] != hidden:
            raise ValueError("Unexpected input embedding/transformer width mismatch.")
        device, dtype = original.weight.device, original.weight.dtype
        backbone.requires_grad_(False)
        embeddings = TrainableTokenEmbedding.from_pretrained(original, dimension).to(device)
        backbone.set_input_embeddings(embeddings)
        if backbone.get_input_embeddings() is not embeddings:
            raise RuntimeError("Internal embedding replacement failed.")
        self.backbone = backbone
        self.classifier = nn.Linear(hidden, 77, device=device, dtype=torch.float32)
        self.backbone_dtype = dtype
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.train()

    @property
    def embeddings(self):
        return self.backbone.get_input_embeddings()

    def train(self, mode=True):
        super().train(mode)
        # Autograd remains enabled. Frozen transformer dropout is disabled;
        # the checkpointing flag needs train mode to activate HF checkpointing.
        if getattr(self.backbone, "is_gradient_checkpointing", False) and mode:
            self.backbone.train()
            for module in self.backbone.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()
        else:
            self.backbone.eval()
        self.embeddings.train(mode)
        return self

    def forward(self, input_ids, attention_mask, labels=None):
        # Call the installed internal embedding, then cast differentiably to
        # the backbone dtype. No inference_mode/no_grad/detach in this path.
        inputs_embeds = self.embeddings(input_ids).to(dtype=self.backbone_dtype)
        output = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                               use_cache=False, return_dict=True)
        pooled = last_valid_pool(output.last_hidden_state, attention_mask).float()
        logits = self.classifier(F.normalize(pooled, dim=-1))
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return {"logits": logits, "loss": loss}

    def trainable_manifest(self):
        return [{"name": name, "shape": list(p.shape), "count": p.numel()}
                for name, p in self.named_parameters() if p.requires_grad]

    def adaptation_state(self):
        return {"embeddings": self.embeddings.state_dict(), "classifier": self.classifier.state_dict()}

    def load_adaptation_state(self, state):
        self.embeddings.load_state_dict(state["embeddings"], strict=True)
        self.classifier.load_state_dict(state["classifier"], strict=True)


def load_model(config, device="cuda"):
    """Only these explicit checkpoint families are supported; no remote code."""
    from transformers import AutoModel, AutoTokenizer
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-checkpoint runs require CUDA; CPU is for tiny unit tests only.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype != torch.bfloat16:
        raise RuntimeError("Phase 1 production training requires BF16-capable CUDA hardware.")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id, revision=config.revision,
                                             trust_remote_code=False)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Checkpoint tokenizer has neither padding nor EOS token.")
    backbone = AutoModel.from_pretrained(config.model_id, revision=config.revision,
                                        trust_remote_code=False, torch_dtype=dtype,
                                        low_cpu_mem_usage=True, attn_implementation="sdpa")
    expected = "phi3" if config.model == "Phi4-mini" else "llama"
    if backbone.config.model_type != expected or backbone.config.hidden_size != 3072:
        raise ValueError("Checkpoint architecture differs from the reviewed contract.")
    if len(tokenizer) > backbone.get_input_embeddings().num_embeddings:
        raise ValueError("Tokenizer vocabulary exceeds pretrained token table.")
    # Factor on CPU before transferring the whole backbone. No automatic
    # quantization/offload fallback: it would change the experimental protocol.
    model = Banking77Model(backbone, config.dimension, config.gradient_checkpointing).to(device)
    return model, tokenizer
