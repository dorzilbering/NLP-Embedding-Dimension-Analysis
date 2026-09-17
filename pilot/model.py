"""Frozen Hugging Face backbones used only as text encoders."""
import numpy as np
from .core import checked, normalize
from .config import MODEL_SPECS


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
        raise RuntimeError("CPU model execution is intentionally disabled; CUDA GPU unavailable.")
    free, total = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError("Need at least 8 GiB free CUDA memory.")
    return {"gpu": torch.cuda.get_device_name(), "free_vram_bytes": free,
            "total_vram_bytes": total, "cuda": torch.version.cuda}


class FrozenEncoder:
    def __init__(self, model_name, revision, batch_size=4, max_length=256):
        import torch
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {model_name}")
        check_gpu()
        spec = MODEL_SPECS[model_name]
        self.model_name, self.model_id, self.revision = model_name, spec["model_id"], revision
        self.width = spec["native_dimension"]
        self.torch, self.batch_size, self.max_length = torch, batch_size, max_length
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=revision, trust_remote_code=False)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = dict(revision=revision, trust_remote_code=False, device_map={"": "cuda:0"}, low_cpu_mem_usage=True)
        self.quantization = spec.get("t4_quantization")
        if self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
        else: kwargs["torch_dtype"] = self.dtype
        self.model = AutoModel.from_pretrained(self.model_id, **kwargs).eval(); self.model.requires_grad_(False)
        config_width = getattr(self.model.config, "hidden_size", None)
        if config_width is None and hasattr(self.model.config, "text_config"): config_width = self.model.config.text_config.hidden_size
        if config_width != self.width: raise ValueError(f"Unexpected hidden width for {model_name}: {config_width} != {self.width}")

    def encode(self, texts, batch_size=None):
        chunks, size = [], batch_size or self.batch_size
        with self.torch.inference_mode():
            for start in range(0, len(texts), size):
                inputs = self.tokenizer(texts[start:start+size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to("cuda:0")
                output = self.model(**inputs, use_cache=False, output_hidden_states=False, output_attentions=False, return_dict=True)
                chunks.append(last_valid_pool(output.last_hidden_state, inputs["attention_mask"]).float().cpu().numpy())
        return checked(np.concatenate(chunks), len(texts), self.width)

    def validate_batching(self):
        texts=["A cat sleeps.","A scientist examines carefully prepared samples in a laboratory."]
        single=normalize(self.encode(texts,batch_size=1)); together=normalize(self.encode(texts,batch_size=2)); agreement=float(np.min(np.sum(single*together,axis=1)))
        if agreement < .995: raise ValueError(f"Batch invariance failed: {agreement}")
        return {"single_vs_batch":agreement}

MODEL_ID=MODEL_SPECS["Phi4-mini"]["model_id"]
class PhiEncoder(FrozenEncoder):
    def __init__(self,revision,batch_size=4,max_length=256): super().__init__("Phi4-mini",revision,batch_size,max_length)
