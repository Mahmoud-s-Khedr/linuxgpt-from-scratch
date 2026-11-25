# GPT-2 Inspired Model — Complete Line-by-Line Study Guide

**Source file:** [`models/gpt2.py`](../models/gpt2.py)  
**Concrete running example used throughout this guide (our `50m` preset):**
- Batch size `B = 1`
- Input sequence: `"docker compose up -d"` → 4 token IDs → `T = 4`
- Vocabulary size `V = 16,384` (our custom Linux BPE tokenizer)
- Context window `block_size = 1024`
- Embedding dimension `C = n_embd = 512`
- Layers `n_layer = 10`
- Attention heads `H = n_head = 8`, so head dimension `D = C // H = 64`

---

## High-Level Data Flow

```text
Token IDs (B, T)
   │
   ├──▶ wte (Token Embedding)      (B, T, 512)
   ├──▶ wpe (Position Embedding)   (T, 512)
   │      ↓ add + dropout
   │    x: (B, T, 512)
   │
   ▼  ┌─────────────────────── Repeat 10x (Block) ───────────────────────┐
      │  x = x + CausalSelfAttention(LayerNorm(x), kv_cache)             │
      │  x = x + MLP(LayerNorm(x))                                       │
      └──────────────────────────────────────────────────────────────────┘
   │
   ▼  ln_f (Final LayerNorm)       (B, T, 512)
   ▼  lm_head (Linear 512 → 16384) (B, T, 16384)  [shares weights with wte!]
   ▼  Cross-Entropy Loss (training) or Next-Token Sampling (inference)
```

---

## 1. Configuration (`GPT2Config`, lines 26–34)

```python
@dataclass
class GPT2Config:
    vocab_size: int = 16384
    block_size: int = 1024
    n_layer: int = 10
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.0
    bias: bool = True
```

- `vocab_size`: Number of unique tokens in our custom Linux BPE tokenizer.
- `block_size`: Maximum sequence length the model can see at once. For GPT-2, this is a **hard ceiling** because `wpe` has exactly `block_size` learned rows.
- `n_layer`: Number of stacked Transformer blocks. Deeper = more sequential reasoning steps.
- `n_head`: Number of parallel attention "perspectives." Each head operates on `head_dim = n_embd // n_head = 64` features.
- `n_embd`: Width of the residual stream (`C`). Every token is represented as a 512-number vector throughout the entire network.
- `dropout`: Randomly zeroes activations during training to prevent overfitting (0.0 during pretraining on 150M tokens, 0.05 during SFT).
- `bias`: Classic GPT-2 includes additive bias vectors `b` in `Linear` and `LayerNorm` layers.

---

## 2. Multi-Head Causal Self-Attention (`CausalSelfAttention`, lines 37–97)

**What problem does Attention solve?**  
Right after embedding, each token vector only knows *itself* (`compose` doesn't know `docker` came right before it). Attention lets every token look **backward** at earlier tokens and pull in relevant context.

### `__init__` (lines 40–50)
```python
assert cfg.n_embd % cfg.n_head == 0
self.n_head = cfg.n_head          # 8
self.n_embd = cfg.n_embd          # 512
self.head_dim = cfg.n_embd // cfg.n_head  # 64
self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
self.resid_dropout = nn.Dropout(cfg.dropout)
```
- **`c_attn` (`512 → 1536`)**: Instead of 3 separate `nn.Linear(512, 512)` calls for Query (`Q`), Key (`K`), and Value (`V`), GPT-2 fuses them into **one big matrix multiply** (`512 → 3 * 512`). One big matmul is faster on GPUs than three smaller ones.
  - **Query (`Q`)**: *"What information am I looking for?"*
  - **Key (`K`)**: *"What information do I contain / advertise?"*
  - **Value (`V`)**: *"If you attend to me, here is the actual information I will pass you."*
- **`c_proj` (`512 → 512`)**: After all 8 heads finish gathering information independently, `c_proj` mixes the 8 heads' outputs together and projects back into the residual stream.

### `forward` (lines 52–97)

#### Step 1: Project and split into Q, K, V (line 60)
```python
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```
- `x`: `(1, 4, 512)`
- `self.c_attn(x)`: `(1, 4, 1536)`
- `.split(512, dim=2)`: slices into three `(1, 4, 512)` tensors: `q`, `k`, `v`.

#### Step 2: Reshape into 8 independent heads (lines 63–65)
```python
q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
```
- `.view(1, 4, 8, 64)`: splits the 512 channels into 8 heads of 64 dims each.
- `.transpose(1, 2)`: swaps `T` and `H` → **`(1, 8, 4, 64)`** (`(B, H, T, D)`).
- **Why transpose?** PyTorch's attention function treats `(B, H)` as batch dimensions, so all 8 heads compute their `(4, 64) @ (64, 4)` attention matrices in parallel in a single GPU kernel!

#### Step 3: KV Cache (lines 68–72)
```python
if kv_cache is not None:
    prev_k, prev_v = kv_cache
    k = torch.cat([prev_k, k], dim=2)  # (B, H, T_past + T, D)
    v = torch.cat([prev_v, v], dim=2)
new_kv_cache = (k, v)
```
- **Why does KV caching matter?**
  Suppose we're generating token #50 during autocomplete.
  - **Without KV cache:** You feed all 50 tokens through the entire model again, recomputing `K` and `V` for tokens 1–49 (wasteful $O(T^2)$ work).
  - **With KV cache:** Because causal attention is strictly backward-looking, the `K` and `V` vectors of tokens 1–49 **never change** when token #50 arrives! So we save `(prev_k, prev_v)` from previous steps, feed **only** token #50 (`T = 1`), concatenate its new `k, v` along the time axis (`dim=2`), and attend to all 50 keys/values!
  - Notice `q` is **not** cached—we only need `q` for the *new* token (`T = 1`).
- **GPT-2 KV Cache Memory:** GPT-2 uses **Multi-Head Attention (MHA)**, so every layer caches all `H = 8` heads of `K` and `V`. For `50m` (`10 layers, 1024 ctx, fp16`):
  `2 (K,V) × 10 layers × 512 dims × 1024 tokens × 2 bytes = 20,971,520 bytes (20 MB)`.

#### Step 4: Scaled Dot-Product Attention & Causal Masking (lines 77–92)
```python
is_causal = (kv_cache is None) and (T > 1)
...
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=is_causal,
    dropout_p=self.dropout if self.training else 0.0,
)
```
Under the hood, `F.scaled_dot_product_attention` computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{D}} + M_{\text{causal}}\right) V$$
1. **$Q K^T$**: shape `(1, 8, T_q, T_k)` — dot product measuring how relevant every past token `k` is to each current token `q`.
2. **Divide by $\sqrt{D} = \sqrt{64} = 8$**: prevents dot products from growing huge (which would make `softmax` saturate into near-0 gradients).
3. **Causal mask (`is_causal=True`)**: sets future positions ($j > i$) to $-\infty$ before `softmax`, so token #2 (`compose`) can **never** cheat by peeking at token #3 (`up`).
   - During cached 1-token decoding (`T == 1`, `kv_cache is not None`), `q` is at the very end of the sequence, so *every* token in `k` is in the past → `is_causal=False` (no mask needed!).
   - On CUDA GPUs, PyTorch automatically dispatches this call to **FlashAttention** (fused, memory-efficient $O(T)$ RAM instead of $O(T^2)$).
4. **Multiply by $V$**: weighted average of the Value vectors → output `y` of shape **`(1, 8, 4, 64)`**.

#### Step 5: Recombine heads & output projection (lines 95–96)
```python
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.resid_dropout(self.c_proj(y))
```
- `.transpose(1, 2)`: `(1, 8, 4, 64)` → `(1, 4, 8, 64)`.
- `.contiguous()`: rearranges memory layout after transpose so `.view()` can safely flatten `(8, 64)` back to `512`.
- `self.c_proj(y)`: `(1, 4, 512)` → `(1, 4, 512)`.

---

## 3. Feed-Forward Network (`MLP`, lines 100–112)

```python
class MLP(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)                   # (B, T, 512) -> (B, T, 2048)
        x = F.gelu(x, approximate="tanh")  # smooth non-linearity
        x = self.c_proj(x)                 # (B, T, 2048) -> (B, T, 512)
        return self.dropout(x)
```
- **Role:** If Attention is **communication** *between* tokens, the MLP is **computation** *within* each token independently. Research shows most factual knowledge (like *"what does `tar -xzf` do?"*) is stored inside these MLP weights!
- **4× expansion (`512 → 2048 → 512`):** Temporarily expands each token into a wider 2048-dim workspace, applies the smooth **GELU** non-linearity (unlike ReLU, GELU doesn't have a dead flat-zero cliff at 0), and compresses back to 512.
- **Parameter count per layer:** `2 × (512 × 2048) = 2,097,152` params.

---

## 4. Transformer Block (`Block`, lines 115–133)

```python
class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x, kv_cache=None):
        attn_out, new_kv = self.attn(self.ln_1(x), kv_cache=kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_kv
```
- **Pre-Norm (`LayerNorm` BEFORE attention/MLP):** Classic GPT-2 puts `LayerNorm` on the branch *entering* attention and MLP, leaving the main residual highway (`x = x + ...`) completely clean and unobstructed.
- **Why residual connections (`x = x + ...`) are essential:** During backpropagation, gradients flow straight through `+` without shrinking—so even a 10-layer or 100-layer network trains cleanly from step 1!
- **What `LayerNorm` does:** For each token's 512-dim vector, subtracts its mean $\mu$, divides by its standard deviation $\sqrt{\sigma^2 + \epsilon}$, multiplies by a learned scale $\gamma$, and adds a learned shift $\beta$.

---

## 5. Full Model (`GPT2`, lines 136–269)

### Initialization & Weight Tying (lines 140–163)
```python
self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # (16384, 512)
self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # (1024, 512)
...
self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
self.wte.weight = self.lm_head.weight  # Weight Tying!
```
- **Weight Tying:** `wte` maps token ID → 512-dim vector; `lm_head` maps 512-dim vector → 16,384 token scores. Sharing the exact same `(16384, 512)` matrix saves **8.4M parameters** (~20% of our 50m model!) and improves generalization because tokens with similar input embeddings get similar output probabilities.
- **Residual Projection Scaling (lines 153–155):**
  ```python
  if name.endswith("c_proj.weight"):
      nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))
  ```
  Each of the `n_layer` blocks adds 2 branches (`attn` + `mlp`) to the residual stream (`2 * n_layer = 20` additions total). Scaling the output projection weights by $1/\sqrt{2 \cdot n_{\text{layer}}}$ keeps the variance of `x` constant from layer 1 to layer 10 at initialization!

### `forward` & Loss Masking (lines 174–209)
```python
past_len = kv_cache[0][0].size(2) if kv_cache is not None else 0
pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
x = self.drop(self.wte(idx) + self.wpe(pos))
```
- **`past_len`:** If we already cached 10 tokens and are now decoding token #11 (`T=1`), `pos` becomes `tensor([10])` so `wpe` looks up position 10 instead of position 0.
- **Training vs. Inference `lm_head` optimization (lines 195–207):**
  - **Training (`targets is not None`):** Projects **all** `T` positions to `(B, T, 16384)` and computes `F.cross_entropy(..., ignore_index=-100)`.
    - In **Stage 1 Pretraining**, no targets are `-100` → every token is predicted.
    - In **Stage 2 SFT**, prompt targets are set to `-100` → PyTorch ignores prompt tokens and only computes loss/gradients on assistant answers and command completions!
  - **Inference (`targets is None`):** Only projects the **very last position** `x[:, [-1], :]` → `(B, 1, 16384)`, saving massive compute during prompt prefill!

### `configure_optimizers` (lines 211–225)
```python
decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]
```
- **Why split parameters into two groups?**
  - 2D weight matrices (`Linear`, `Embedding`) get **weight decay (`0.1`)** for regularization.
  - 1D tensors (`LayerNorm` scales and `bias` vectors) get **weight decay `0.0`** — decaying LayerNorm scales toward 0 harms training stability!
- **`fused=True` on CUDA:** Runs the entire AdamW update in one fused GPU kernel.

### `generate` (lines 227–269)
1. **Prefill:** Feeds the entire prompt `idx` once (`logits, _, kv_cache = self(idx)`), building the KV cache for all prompt tokens.
2. **Decode loop (1 token at a time):**
   - `temperature <= 0.0`: **Greedy decoding** (`argmax`) — deterministic, ideal for inline CLI autocomplete (`docker compose ` → `up -d`).
   - `temperature > 0.0`: Divides logits by `temperature` (lower = sharper/safer, higher = more creative), optionally filters with **top-k** (keep only top $k$ tokens) and **top-p / nucleus** (keep smallest set of tokens whose probabilities sum to `top_p`), then samples via `torch.multinomial`.
   - Stops immediately when `stop_token_id` (`<|end|>`) is generated.
   - Calls `self(next_id, kv_cache=kv_cache)` feeding **only the 1 new token**!

---

## Summary: GPT-2 Strengths & Limitations
- **Strengths:** Clean, battle-tested, easy to understand, strong baseline.
- **Limitations (fixed in [`LLAMA_LINE_BY_LINE.md`](LLAMA_LINE_BY_LINE.md)):**
  1. Learned `wpe` cannot generalize cleanly to relative distances.
  2. `LayerNorm` does extra mean/bias work.
  3. Full MHA caches all 8 KV heads (**4× larger KV cache** than LLaMA's GQA).
  4. Plain GELU MLP is slightly less parameter-efficient than SwiGLU.
