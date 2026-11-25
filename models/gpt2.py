"""
GPT-2 Inspired Decoder-Only Transformer (from scratch in PyTorch).

Key architectural choices (2019 GPT-2 style):
  1. Learned absolute positional embeddings (wpe) added to token embeddings (wte)
  2. Pre-norm LayerNorm (mean + variance, weight + bias)
  3. Multi-Head Causal Self-Attention (MHA: n_kv_head == n_head) with KV cache
  4. Standard 2-layer MLP with GELU activation (C -> 4C -> C)
  5. Weight tying between input token embedding (wte) and output lm_head

Tensor shape legend used in comments:
  B = batch size
  T = sequence length (tokens in current forward pass)
  C = embedding dimension (n_embd)
  H = number of attention heads (n_head)
  D = head dimension (C // H)
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with optional KV caching."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        # Project input x into Q, K, V all at once for efficiency: C -> 3*C
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        # Output projection back to residual stream: C -> C
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        # 1. Compute Q, K, V: (B, T, 3C) -> 3 x (B, T, C)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 2. Split into heads: (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 3. Append to KV cache if provided (used during fast autoregressive generation)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)  # (B, H, T_past + T, D)
            v = torch.cat([prev_v, v], dim=2)
        new_kv_cache = (k, v)

        # 4. Scaled Dot-Product Attention (uses FlashAttention on GPU automatically)
        # When decoding 1 token with KV cache (T == 1 and past exists), every past token
        # is valid context so no causal mask is needed.
        is_causal = (kv_cache is None) and (T > 1)
        if kv_cache is not None and T > 1:
            # Prefill-with-cache edge case: explicit causal mask for new tokens
            T_total = k.size(2)
            T_past = T_total - T
            attn_mask = torch.ones((T, T_total), dtype=torch.bool, device=x.device)
            attn_mask = torch.tril(attn_mask, diagonal=T_past)
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=self.dropout if self.training else 0.0,
            )

        # 5. Recombine heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y, new_kv_cache


class MLP(nn.Module):
    """Standard GPT-2 feed-forward block: C -> 4C -> GELU -> C."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")  # GPT-2 uses tanh-approx GELU
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm GPT-2 Transformer block."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out          # residual connection 1 (communication)
        x = x + self.mlp(self.ln_2(x))  # residual connection 2 (computation)
        return x, new_kv


class GPT2(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # token embeddings
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # learned position embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: share token embedding matrix with output projection
        self.wte.weight = self.lm_head.weight

        # Initialize weights (GPT-2 style)
        self.apply(self._init_weights)
        # Scale residual projections by 1/sqrt(2 * n_layer) to keep residual variance stable
        for name, p in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
        return n_params

    def kv_cache_bytes(self, seq_len: int | None = None, dtype_bytes: int = 2) -> int:
        """Memory footprint of the KV cache for batch_size=1 at seq_len."""
        L = seq_len or self.cfg.block_size
        # 2 (K and V) * n_layer * n_head * L * head_dim * dtype_bytes
        return 2 * self.cfg.n_layer * self.cfg.n_embd * L * dtype_bytes

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

        # Position indices account for any cached prefix tokens
        pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))  # (B, T, C)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x, new_layer_kv = block(x, kv_cache=layer_cache)
            new_kv_cache.append(new_layer_kv)

        x = self.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            # ignore_index=-100 natively supports Stage 2 SFT prompt masking!
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            # Inference optimization: only project the final position's logits
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss, new_kv_cache

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float] = (0.9, 0.95),
        device_type: str = "cuda",
    ) -> torch.optim.Optimizer:
        # Apply weight decay only to 2D+ tensors (Linear/Embedding weights), not biases/LayerNorms
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
        """Fast KV-cached autoregressive generation (preserves caller's train/eval mode)."""
        was_training = self.training
        self.eval()
        try:
            # Truncate prompt if it exceeds block_size
            if idx.size(1) > self.cfg.block_size:
                idx = idx[:, -self.cfg.block_size:]

            # Prefill pass on full prompt
            logits, _, kv_cache = self(idx)
            generated = idx

            for _ in range(max_new_tokens):
                if generated.size(1) >= self.cfg.block_size:
                    break
                next_logits = logits[:, -1, :]  # (B, vocab_size)

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

                # Decode step: feed ONLY the 1 newly generated token using the KV cache!
                logits, _, kv_cache = self(next_id, kv_cache=kv_cache)

            return generated
        finally:
            if was_training:
                self.train()
