"""
Model registry and matched-size presets for GPT-2 vs. LLaMA comparison.

Both architectures expose the exact same interface:
    logits, loss, new_kv_cache = model(idx, targets=None, kv_cache=None)
    out_ids = model.generate(idx, max_new_tokens=64, temperature=0.7, top_p=0.9, stop_token_id=None)
    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type="cuda")
"""

from dataclasses import dataclass
from typing import Literal

from .gpt2 import GPT2, GPT2Config
from .llama import LLaMA, LLaMAConfig

ArchName = Literal["gpt2", "llama"]


@dataclass
class PresetSpec:
    n_layer: int
    n_head: int
    n_kv_head: int   # used by LLaMA GQA (ignored by GPT-2 MHA)
    n_embd: int
    block_size: int
    # Optional SwiGLU hidden_dim override so LLaMA matches GPT-2 param count closely
    llama_hidden_dim: int | None = None


# Presets tuned for vocab_size=16384 (custom Linux BPE tokenizer)
PRESETS: dict[str, PresetSpec] = {
    # Tiny (~1.5M params) — instant local CPU sanity checks & debugging
    "tiny": PresetSpec(
        n_layer=4, n_head=4, n_kv_head=2, n_embd=128, block_size=256,
    ),
    # Small (~18M params) — fast Colab iteration (~10-15 min on L4/A100)
    "15m": PresetSpec(
        n_layer=6, n_head=6, n_kv_head=2, n_embd=384, block_size=512,
    ),
    # Medium (~45-50M params) — sweet spot for fast Ryzen CPU autocomplete + solid Q&A
    "50m": PresetSpec(
        n_layer=10, n_head=8, n_kv_head=2, n_embd=512, block_size=1024,
    ),
    # Large (~115-124M params) — strongest scratch model on Colab A100
    "124m": PresetSpec(
        n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024,
    ),
}


def build_model(
    arch: ArchName,
    preset: str = "50m",
    vocab_size: int = 16384,
    dropout: float = 0.0,
    block_size: int | None = None,
):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESETS.keys())}")
    spec = PRESETS[preset]
    ctx = block_size if block_size is not None else spec.block_size

    if arch == "gpt2":
        cfg = GPT2Config(
            vocab_size=vocab_size,
            block_size=ctx,
            n_layer=spec.n_layer,
            n_head=spec.n_head,
            n_embd=spec.n_embd,
            dropout=dropout,
            bias=True,
        )
        return GPT2(cfg)
    elif arch == "llama":
        cfg = LLaMAConfig(
            vocab_size=vocab_size,
            block_size=ctx,
            n_layer=spec.n_layer,
            n_head=spec.n_head,
            n_kv_head=spec.n_kv_head,
            n_embd=spec.n_embd,
            hidden_dim=spec.llama_hidden_dim,
            dropout=dropout,
            tie_word_embeddings=True,
        )
        return LLaMA(cfg)
    else:
        raise ValueError(f"Unknown arch '{arch}'. Choose 'gpt2' or 'llama'.")


__all__ = ["GPT2", "GPT2Config", "LLaMA", "LLaMAConfig", "PRESETS", "build_model"]
