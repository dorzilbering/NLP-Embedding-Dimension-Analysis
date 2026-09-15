"""Phi backbone only: no LM head, generation, or parameter updates."""
import numpy as np
from .core import checked, normalize

MODEL_ID = "microsoft/Phi-4-mini-instruct"


def last_valid_pool(hidden, attention_mask):
    import torch
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden state and mask shapes differ.")
    valid = attention_mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError("All-padding input.")
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    indices = positions.masked_fill(~valid, -1).max(dim=1).values
    return hidden[torch.arange(len(hidden), device=hidden.device), indices]


def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable. CPU model execution is intentionally disabled.")
    free, total = torch.cuda.mem_get_info()
    if free < 10 * 1024**3:
        raise RuntimeError("Need at least 10 GiB free CUDA memory; a 16 GB GPU is recommended.")
    return {"gpu": torch.cuda.get_device_name(), "free_vram_bytes": free,
            "total_vram_bytes": total, "cuda": torch.version.cuda}


class PhiEncoder:
    def __init__(self, revision, batch_size=4, max_length=256):
        import torch
        from transformers import AutoTokenizer, AutoModel
        check_gpu()
        self.torch = torch
        self.batch_size = batch_size
        self.max_length = max_length
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=revision, trust_remote_code=False, torch_dtype=self.dtype,
            device_map={"": "cuda:0"}, attn_implementation="sdpa", low_cpu_mem_usage=True,
        ).eval()
        self.model.requires_grad_(False)
        if self.model.config.hidden_size != 3072:
            raise ValueError("Unexpected Phi hidden width.")

    def encode(self, texts, batch_size=None):
        chunks = []
        size = batch_size or self.batch_size
        with self.torch.inference_mode():
            for start in range(0, len(texts), size):
                # Plain text; tokenizer defaults supply special tokens. No chat template
                # and no manually appended EOS. Pool the final attention-mask=1 token.
                inputs = self.tokenizer(texts[start:start+size], padding=True, truncation=True,
                                        max_length=self.max_length, return_tensors="pt").to("cuda:0")
                output = self.model(**inputs, use_cache=False, output_hidden_states=False,
                                    output_attentions=False, return_dict=True)
                pooled = last_valid_pool(output.last_hidden_state, inputs["attention_mask"])
                chunks.append(pooled.float().cpu().numpy())
        return checked(np.concatenate(chunks), len(texts), 3072)

    def validate_batching(self):
        texts = ["A cat sleeps.", "A scientist examines several carefully prepared samples in a laboratory."]
        single = normalize(self.encode(texts, batch_size=1))
        right = normalize(self.encode(texts, batch_size=2))
        try:
            self.tokenizer.padding_side = "left"
            left = normalize(self.encode(texts, batch_size=2))
        finally:
            self.tokenizer.padding_side = "right"
        agreements = {"single_vs_right": float(np.min(np.sum(single * right, axis=1))),
                      "single_vs_left": float(np.min(np.sum(single * left, axis=1)))}
        if min(agreements.values()) < 0.999:
            raise ValueError(f"Batch/padding invariance failed: {agreements}")
        return agreements
