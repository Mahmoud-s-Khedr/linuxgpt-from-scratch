"""
Local CPU sanity check for both GPT-2 and LLaMA architectures:
  1. Compares parameter counts and KV-cache memory across presets.
  2. Verifies exact numerical parity between KV-cached decoding and full forward pass.
  3. Overfits a tiny batch on CPU with both standard LM loss and SFT masked loss (-100).
"""

import time
import torch
from models import PRESETS, build_model


def check_kv_cache_parity(arch: str):
    torch.manual_seed(42)
    model = build_model(arch, preset="tiny", vocab_size=512, dropout=0.0)
    model.eval()

    x = torch.randint(0, 512, (2, 16))

    # Full forward pass (no cache)
    with torch.no_grad():
        logits_full, _, _ = model(x, targets=None)
        # Note: forward with targets=None returns only the last position's logits (B, 1, V)
        # Let's also check via targets to get all positions if needed, or step token-by-token:
        # Prefill first 10 tokens, then feed tokens 10..15 one by one with KV cache
        logits_step, _, kv = model(x[:, :10])
        for t in range(10, 16):
            logits_step, _, kv = model(x[:, t : t + 1], kv_cache=kv)

    max_diff = (logits_full - logits_step).abs().max().item()
    assert max_diff < 1e-4, f"{arch} KV cache mismatch: max_diff={max_diff}"
    print(f"  [{arch:5s}] KV-cache parity PASS (max abs diff = {max_diff:.2e})")


def check_overfit_and_sft_mask(arch: str):
    torch.manual_seed(1337)
    model = build_model(arch, preset="tiny", vocab_size=512, dropout=0.0)
    opt = model.configure_optimizers(weight_decay=0.0, learning_rate=3e-3, device_type="cpu")

    # Fake batch: 4 sequences of length 32, first 16 tokens are masked prompt (-100)
    seq = torch.randint(0, 512, (4, 33))
    x = seq[:, :-1]
    y = seq[:, 1:].clone()
    y[:, :16] = -100  # SFT prompt masking!

    t0 = time.perf_counter()
    losses = []
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        _, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    elapsed = time.perf_counter() - t0

    assert losses[-1] < 0.25, f"{arch} failed to overfit tiny batch: final loss={losses[-1]:.4f}"
    print(
        f"  [{arch:5s}] SFT masked overfit PASS "
        f"(loss {losses[0]:.3f} -> {losses[-1]:.4f} in 40 steps, {elapsed:.2f}s CPU)"
    )


def main():
    print("=== 1. Architecture Size & KV-Cache Comparison (vocab=16,384, fp16 KV) ===")
    header = f"{'Preset':<8} {'Arch':<6} {'Params':>11} {'KV Cache (KB)':>15} {'Context':>9}"
    print(header)
    print("-" * len(header))
    for preset in PRESETS:
        for arch in ("gpt2", "llama"):
            m = build_model(arch, preset=preset, vocab_size=16384)
            kv_kb = m.kv_cache_bytes(dtype_bytes=2) / 1024.0
            print(
                f"{preset:<8} {arch:<6} {m.num_parameters():>11,} {kv_kb:>14.1f}K {m.cfg.block_size:>9}"
            )
        print()

    print("=== 2. KV-Cache Exact Numerical Parity ===")
    for arch in ("gpt2", "llama"):
        check_kv_cache_parity(arch)

    print("\n=== 3. SFT Prompt-Masked Overfit Test (CPU) ===")
    for arch in ("gpt2", "llama"):
        check_overfit_and_sft_mask(arch)

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
