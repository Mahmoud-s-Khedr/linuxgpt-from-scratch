# LLaMA Inspired Model — Complete Line-by-Line Study Guide

**Source file:** [`models/llama.py`](../models/llama.py)  
**Concrete running example used throughout this guide (our `50m` preset):**
- Batch size `B = 1`
- Input sequence: `"docker compose up -d"` → 4 token IDs → `T = 4`
- Vocabulary size `V = 16,384` (same custom Linux BPE tokenizer as GPT-2)
- Context window `block_size = 1024`
- Embedding dimension `C = n_embd = 512`
- Layers `n_layer = 10`
- Query heads `H = n_head = 8`, **KV heads `H_kv = n_kv_head = 2`** (`n_rep = 8 // 2 = 4`)
- Head dimension `D = C // H = 64`
- SwiGLU hidden dim `hidden_dim = 1408` (`≈ 8/3 × 512` rounded up to a multiple of 64)

---

## High-Level Data Flow (Notice the 4 Modern Upgrades over GPT-2!)

```text
Token IDs (B, T)
   │
   ▼  tok_emb (Token Embedding)     (B, T, 512)   [NO wpe table! RoPE lives inside attention]
   │
   ▼  ┌─────────────────────── Repeat 10x (LLaMABlock) ───────────────────────┐
      │  x = x + GroupedQueryAttention(RMSNorm(x), cos, sin, kv_cache)        │
      │            └─ applies RoPE to Q & K, caches only 2 KV heads (4x less) │
      │  x = x + SwiGLU(RMSNorm(x))                                           │
      │            └─ W_down(SiLU(W_gate(x)) * W_up(x))                       │
      └───────────────────────────────────────────────────────────────────────┘
   │
   ▼  norm_f (Final RMSNorm)        (B, T, 512)
   ▼  lm_head (Linear 512 → 16384)  (B, T, 16384) [shares weights with tok_emb]
   ▼  Cross-Entropy Loss (training) or Next-Token Sampling (inference)
```

---

## 1. Configuration (`LLaMAConfig`, lines 20–35)

```python
@dataclass
class LLaMAConfig:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    hidden_dim: int | None = None
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_word_embeddings: bool = True
```

### Why `get_hidden_dim()` uses `8/3 * n_embd` (lines 33–38)
```python
raw = int(2 * (4 * self.n_embd) / 3)
return ((raw + 63) // 64) * 64
```
- GPT-2's MLP has **2** matrices (`C → 4C` and `4C → C`), total `8 * C²` parameters.
- LLaMA's SwiGLU has **3** matrices (`w_gate`, `w_up`, `w_down`), total `3 * C * hidden` parameters.
- To keep the **same parameter budget** (`3 * C * hidden ≈ 8 * C²`), we set `hidden ≈ (8/3) * C`.
- Rounding up to a **multiple of 64** (`((raw + 63) // 64) * 64` → `1408` for `C=512`) aligns matrix dimensions to GPU Tensor Core tile sizes for maximum speed!

---

## 2. Upgrade #1: `RMSNorm` (lines 41–51)

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x.type_as(x) * self.weight
```

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{C}\sum_{i=1}^{C} x_i^2 + \epsilon}} \cdot \gamma$$

Line-by-line:
1. **`self.weight = nn.Parameter(torch.ones(dim))`**: Learnable per-channel scale $\gamma$ initialized to all 1s. Unlike `LayerNorm`, there is **no bias $\beta$** and **no mean subtraction**.
2. **`x.float()`**: Temporarily upcasts from `bf16`/`fp16` to `float32` so squaring (`pow(2)`) doesn't overflow or lose precision.
3. **`.pow(2).mean(dim=-1, keepdim=True)`**: Computes the mean of squares across the 512 features (`dim=-1`). `keepdim=True` keeps shape `(B, T, 1)` so it broadcasts cleanly across `(B, T, 512)`.
4. **`torch.rsqrt(... + self.eps)`**: Fused reciprocal square root (`1 / sqrt(...)`), with `eps = 1e-6` preventing division by zero.
5. **`.type_as(x) * self.weight`**: Casts back to `bf16` and scales by learned `weight`.

**Why better than `LayerNorm`?** Rescaling (not re-centering) is what stabilizes deep Transformers. Dropping the mean calculation and bias makes RMSNorm ~15–20% faster with equal or better training stability.

---

## 3. Upgrade #2: RoPE — Rotary Position Embeddings (lines 54–87)

Instead of adding a learned absolute position table `wpe` at the input, LLaMA **rotates** the Query (`q`) and Key (`k`) vectors inside every attention layer by an angle proportional to token position $m$.

### Precomputing Rotation Angles (`precompute_rope_freqs`, lines 54–65)
```python
inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
positions = torch.arange(max_seq_len, dtype=torch.float32)
freqs = torch.outer(positions, inv_freq)  # (1024, 32)
return torch.cos(freqs), torch.sin(freqs)
```
- `head_dim = 64` is grouped into **32 pairs** of numbers (`0::2` and `1::2`) — think of them as 32 2D clocks.
- `inv_freq` (shape `(32,)`): assigns each clock a different angular frequency $\omega_i = 10000^{-2i/D}$:
  - Pair 0: $\omega_0 = 1.0$ rad/token (**fast hand** — sensitive to immediate neighbors like `docker` → `compose`).
  - Pair 31: $\omega_{31} \approx 0.0001$ rad/token (**slow hand** — tracks long-range context hundreds of tokens earlier).
- `torch.outer(positions, inv_freq)`: outer product giving the exact angle $m \cdot \omega_i$ for every position $m \in [0, 1023]$ and every pair $i \in [0, 31]$.
- Stored via `self.register_buffer("rope_cos", cos, persistent=False)` (lines 204–205): automatically moves to GPU with `.to(device)`, never trained by the optimizer, and not saved into checkpoint files.

### Applying the 2D Rotation (`apply_rope`, lines 68–87)
```python
x_even = x[..., 0::2]  # (B, H, T, 32)
x_odd  = x[..., 1::2]  # (B, H, T, 32)

cos = cos.unsqueeze(0).unsqueeze(0).type_as(x)  # (1, 1, T, 32)
sin = sin.unsqueeze(0).unsqueeze(0).type_as(x)

out_even = x_even * cos - x_odd * sin
out_odd  = x_even * sin + x_odd * cos

return torch.stack((out_even, out_odd), dim=-1).flatten(-2)  # (B, H, T, 64)
```
- This is the exact 2D rotation matrix from geometry:
  $$\begin{pmatrix} x'_{\text{even}} \\ x'_{\text{odd}} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_{\text{even}} \\ x_{\text{odd}} \end{pmatrix}$$
- `torch.stack(..., dim=-1)` gives `(B, H, T, 32, 2)`, and `.flatten(-2)` interleaves even/odd back to `(B, H, T, 64)`.
- **Why RoPE beats GPT-2's `wpe`:** When attention computes the dot product $q_m \cdot k_n$, rotating $q$ by angle $m\omega$ and $k$ by angle $n\omega$ gives a dot product that depends **purely on $(m - n)$** — the *relative distance* between the two tokens!
- **KV Cache interaction (lines 239–240):**
  ```python
  cos = self.rope_cos[past_len : past_len + T]
  sin = self.rope_sin[past_len : past_len + T]
  ```
  When decoding token #11 (`past_len = 10, T = 1`), we slice row `10` of `rope_cos`/`rope_sin` and rotate only the new `q` and `k`. Past cached `k` vectors were **already rotated** at positions `0..9` when they were cached!

---

## 4. Upgrade #3: Grouped-Query Attention (`GroupedQueryAttention`, lines 90–157)

**The problem with GPT-2's Multi-Head Attention (MHA):**  
During autoregressive decoding (like inline CLI autocomplete), reading the KV cache from RAM is the #1 speed bottleneck on CPUs and GPUs. With `H = 8` heads, GPT-2 stores 8 separate `K` heads and 8 separate `V` heads per layer.

**LLaMA's solution (GQA):** Keep **8 Query heads**, but share only **2 Key heads and 2 Value heads** (`n_rep = 8 // 2 = 4` query heads share each KV head)!

### `__init__` (lines 93–109)
```python
self.n_head = cfg.n_head          # 8
self.n_kv_head = cfg.n_kv_head    # 2
self.n_rep = cfg.n_head // cfg.n_kv_head  # 4
self.head_dim = cfg.n_embd // cfg.n_head  # 64

self.wq = nn.Linear(512, 8 * 64, bias=False)  # 512 -> 512
self.wk = nn.Linear(512, 2 * 64, bias=False)  # 512 -> 128  (4x smaller!)
self.wv = nn.Linear(512, 2 * 64, bias=False)  # 512 -> 128  (4x smaller!)
self.wo = nn.Linear(8 * 64, 512, bias=False)  # 512 -> 512
```
- Notice `bias=False` on all 4 linear layers (LLaMA drops linear biases everywhere).

### `forward` (lines 111–157)
1. **Project Q, K, V (lines 121–123):**
   - `q`: `(1, 8, 4, 64)` — 8 query heads
   - `k`: `(1, 2, 4, 64)` — only **2** key heads!
   - `v`: `(1, 2, 4, 64)` — only **2** value heads!
2. **Apply RoPE to Q and K (lines 126–127):**
   - `q = apply_rope(q, cos, sin)`
   - `k = apply_rope(k, cos, sin)`
   - *(Notice `v` is NOT rotated — only `q` and `k` determine attention weights!)*
3. **Update compact KV cache (lines 130–134):**
   - We cache `(k, v)` **BEFORE** expanding heads!
   - Cached shape is `(B, 2, T_total, 64)` instead of `(B, 8, T_total, 64)` → **4× smaller KV cache!**
   - For our `50m` model (`10 layers, 1024 ctx, fp16`): **5.0 MB** vs. GPT-2's **20.0 MB**!
4. **Expand KV heads for attention (lines 137–141):**
   ```python
   if self.n_rep > 1:
       k_attn = k.repeat_interleave(self.n_rep, dim=1)  # (1, 2, T, 64) -> (1, 8, T, 64)
       v_attn = v.repeat_interleave(self.n_rep, dim=1)
   ```
   - Q heads `0,1,2,3` attend to KV head `0`; Q heads `4,5,6,7` attend to KV head `1`.
5. **Scaled Dot-Product Attention + Output Projection (lines 144–157):**
   - Exact same FlashAttention (`F.scaled_dot_product_attention`) and head-merging `self.wo(y)` as GPT-2!

---

## 5. Upgrade #4: `SwiGLU` MLP (lines 160–172)

```python
class SwiGLU(nn.Module):
    def __init__(self, cfg: LLaMAConfig):
        super().__init__()
        hidden = cfg.get_hidden_dim()  # 1408 for C=512
        self.w_gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up   = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))
```

$$\text{SwiGLU}(x) = W_{\text{down}}\Big(\text{SiLU}(W_{\text{gate}} x) \odot (W_{\text{up}} x)\Big)$$

- **How gating works intuitively:**
  - `w_up(x)` extracts 1,408 candidate features from the token.
  - `F.silu(w_gate(x))` acts as a learned **dimmer switch / gate** for each of those 1,408 features (SiLU = $x \cdot \sigma(x)$, also called Swish).
  - Multiplying them elementwise (`*`) lets the network dynamically turn features on, off, or amplify them depending on the token's context!
  - `w_down(...)` projects the gated 1,408 features back to `512`.
- Empirically, SwiGLU consistently achieves **lower validation loss at the exact same parameter count** than a standard GELU MLP.

---

## 6. Block, Full Model, Init, Loss & Generation (lines 175–315)

- **`LLaMABlock` (lines 175–194):** Pre-RMSNorm residual block (`x = x + attn(norm1(x))`, `x = x + mlp(norm2(x))`).
- **Residual Scaling Init (lines 208–211):** Scales both `wo.weight` (attention output) and `w_down.weight` (MLP output) by $1/\sqrt{2 \cdot n_{\text{layer}}}$, matching GPT-2's residual variance stabilization.
- **Weight Tying (lines 200–201):** Shares `tok_emb.weight` with `lm_head.weight` (saves 8.4M params on our 50m model).
- **Exact same `forward`, `ignore_index=-100` SFT loss masking, `configure_optimizers`, and KV-cached `generate()`** as GPT-2 — so both models plug into the exact same training loop and CLI!

---

## Side-by-Side Cheat Sheet (Our `50m` Preset, `vocab=16,384`)

| Metric / Component | GPT-2 (`gpt2.py`) | LLaMA (`llama.py`) |
| :--- | :--- | :--- |
| **Total Parameters** | 40,437,760 | 36,579,840 |
| **KV Cache (1024 ctx, fp16)** | **20.0 MB** | **5.0 MB (4× smaller!)** |
| **Position Encoding** | Learned `wpe` table | **RoPE** (2D rotation on Q & K) |
| **Normalization** | `LayerNorm` (mean + var + bias) | **RMSNorm** (RMS only, no bias) |
| **Attention** | MHA (8 Q, 8 K, 8 V) | **GQA** (8 Q, 2 K, 2 V) |
| **MLP** | `512 → 2048 → 512` (GELU) | **SwiGLU** `512 → 1408 → 512` (SiLU gate) |
| **Linear Biases** | `True` | `False` |
