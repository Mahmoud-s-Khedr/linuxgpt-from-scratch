"""
LLaMA Inspired Decoder-Only Transformer (from scratch in PyTorch).

Key modern upgrades over GPT-2 (2023–2026 LLaMA / Qwen style):
  1. RoPE (Rotary Position Embeddings) applied to Q & K — no learned wpe table
  2. RMSNorm instead of LayerNorm — no mean subtraction, no bias
  3. GQA (Grouped-Query Attention: n_kv_head < n_head) — dramatically smaller KV cache
  4. SwiGLU MLP (W_down(SiLU(W_gate(x)) * W_up(x))) — stronger representation per param
  5. Bias-free linear layers throughout

Exposes the exact same forward() / generate() / configure_optimizers() API as GPT2
so both architectures can be trained and benchmarked side by side.
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LLaMAConfig:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_kv_head: int = 2       # GQA: 8 Q heads share 2 KV heads (4x smaller KV cache!)
    n_embd: int = 512
    hidden_dim: int | None = None  # defaults to LLaMA's ~8/3 * n_embd rounded to multiple of 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def get_hidden_dim(self) -> int:
        if self.hidden_dim is not None:
            return self.hidden_dim
        # SwiGLU has 3 weight matrices instead of 2, so 8/3 * C matches 4 * C parameter count
        raw = int(2 * (4 * self.n_embd) / 3)
        return ((raw + 63) // 64) * 64  # round up to multiple of 64 for fast GPU kernels


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean subtraction, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS in float32 for numerical stability under bf16/fp16 AMP
        norm_x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x.type_as(x) * self.weight


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos and sin tables for Rotary Position Embeddings (RoPE).
    Returns cos, sin each of shape (max_seq_len, head_dim // 2).
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Rotate pairs of features (x_even, x_odd) by position-dependent angles.
    x:   (B, H, T, D)
    cos: (T, D // 2)
    sin: (T, D // 2)
    """
    x_even = x[..., 0::2]  # (B, H, T, D/2)
    x_odd = x[..., 1::2]   # (B, H, T, D/2)

    # Broadcast cos/sin to (1, 1, T, D/2)
    cos = cos.unsqueeze(0).unsqueeze(0).type_as(x)
    sin = sin.unsqueeze(0).unsqueeze(0).type_as(x)

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    # Interleave back to (B, H, T, D)
    return torch.stack((out_even, out_odd), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Causal Self-Attention (GQA) with RoPE and KV caching."""

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head  # how many Q heads share each KV head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        # 1. Project to Q (H heads) and K, V (H_kv heads)
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)       # (B, H, T, D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, H_kv, T, D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)    # (B, H_kv, T, D)

        # 2. Apply RoPE to Q and K (using positions starting at past_len)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # 3. Append to compact KV cache (stores only n_kv_head, NOT n_head!)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        # 4. Expand KV heads to match Q heads for attention computation
        if self.n_rep > 1:
            k_attn = k.repeat_interleave(self.n_rep, dim=1)  # (B, H, T_total, D)
            v_attn = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_attn, v_attn = k, v

        # 5. Scaled Dot-Product Attention
        is_causal = (kv_cache is None) and (T > 1)
        if kv_cache is not None and T > 1:
            T_total = k_attn.size(2)
            T_past = T_total - T
            attn_mask = torch.ones((T, T_total), dtype=torch.bool, device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=T_past)
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k_attn, v_attn,
                is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )

        # 6. Merge heads and project back to C
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.wo(y)), new_kv_cache


class SwiGLU(nn.Module):
    """Gated Feed-Forward Network: W_down(SiLU(W_gate(x)) * W_up(x))."""

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        hidden = cfg.get_hidden_dim()
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class LLaMABlock(nn.Module):
    """Pre-RMSNorm LLaMA Transformer block (GQA + RoPE + SwiGLU)."""

    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.norm1(x), cos, sin, kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, new_kv


class LLaMA(nn.Module):
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        self.cfg = cfg
        head_dim = cfg.n_embd // cfg.n_head

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([LLaMABlock(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.n_embd, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            self.tok_emb.weight = self.lm_head.weight

        # Precompute RoPE tables as non-persistent buffers (not saved as model weights)
        cos, sin = precompute_rope_freqs(head_dim, cfg.block_size, theta=cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Initialize weights
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def kv_cache_bytes(self, seq_len: int | None = None, dtype_bytes: int = 2) -> int:
        """Memory footprint of the GQA KV cache for batch_size=1 at seq_len."""
        L = seq_len or self.cfg.block_size
        head_dim = self.cfg.n_embd // self.cfg.n_head
        # Notice n_kv_head instead of n_head -> n_head // n_kv_head times smaller than GPT-2!
        return 2 * self.cfg.n_layer * self.cfg.n_kv_head * head_dim * L * dtype_bytes

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        B, T = idx.shape
        past_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
        assert past_len + T <= self.cfg.block_size, (
            f"Sequence length {past_len + T} exceeds block_size {self.cfg.block_size}"
        )

        # Slice RoPE cos/sin for the exact token positions [past_len, past_len + T)
        cos = self.rope_cos[past_len : past_len + T]
        sin = self.rope_sin[past_len : past_len + T]

        x = self.drop(self.tok_emb(idx))  # (B, T, C)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_layer_kv = block(x, cos, sin, kv_cache=layer_cache)
            new_kv_cache.append(new_layer_kv)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_kv_cache

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_token_id: int | None = None,
    ) -> torch.Tensor:
        """Fast GQA KV-cached autoregressive generation (preserves caller's train/eval mode)."""
        was_training = self.training
        self.eval()
        try:
            if idx.size(1) > self.cfg.block_size:
                idx = idx[:, -self.cfg.block_size:]

            logits, _, kv_cache = self(idx)
            generated = idx

            for _ in range(max_new_tokens):
                if generated.size(1) >= self.cfg.block_size:
                    break
                next_logits = logits[:, -1, :]

                if temperature <= 0.0:
                    next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None and top_k > 0:
                        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                        next_logits[next_logits < v[:, [-1]]] = float("-inf")
                    if top_p is not None and 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        remove_mask = cum_probs > top_p
                        remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                        remove_mask[:, 0] = False
                        indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
                        next_logits[indices_to_remove] = float("-inf")
                    probs = F.softmax(next_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_id], dim=1)
                if stop_token_id is not None and (next_id == stop_token_id).all():
                    break

                logits, _, kv_cache = self(next_id, kv_cache=kv_cache)

            return generated
        finally:
            if was_training:
                self.train()
