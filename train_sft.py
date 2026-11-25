"""
Stage 2 Supervised Fine-Tuning (SFT) Script with Prompt Loss Masking (-100).

Loads a Stage 1 pretrained checkpoint (GPT-2 or LLaMA) and fine-tunes on
mixed <|qa|> and <|complete|> examples from artifacts/sft_v2/sft_train.jsonl.

Key mechanism:
  - Prompt tokens are masked with ignore_index = -100 in target labels y.
  - Direct loss is computed only on assistant answers/completions; contextual gradients still flow through the prompt.

Example:
  python train_sft.py \
      --arch llama --preset 15m \
      --init_ckpt runs/pretrain_llama_15m/best.pt \
      --max_steps 1000 --batch_size 16 --grad_accum 2
"""

from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from models import build_model
from tokenizer import LinuxTokenizer


class DualModeSFTDataset(Dataset):
    """Pre-tokenizes prompt/completion pairs and applies exact -100 prompt masking."""

    def __init__(self, jsonl_path: Path, tok: LinuxTokenizer, block_size: int):
        self.pad_id = tok.pad_id
        self.block_size = block_size
        self.samples: list[tuple[list[int], list[int]]] = []

        skipped = 0
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                prompt_ids = tok.encode(rec["prompt"])
                comp_ids = tok.encode(rec["completion"])

                # Truncate overly long examples while keeping at least 16 completion tokens
                if len(prompt_ids) + len(comp_ids) > block_size + 1:
                    max_prompt = max(32, block_size - 64)
                    prompt_ids = prompt_ids[:max_prompt]
                    comp_ids = comp_ids[: (block_size + 1 - len(prompt_ids))]

                if len(prompt_ids) < 2 or len(comp_ids) < 1:
                    skipped += 1
                    continue

                full_ids = prompt_ids + comp_ids
                x = full_ids[:-1]
                y = full_ids[1:]

                # Mask all prompt target positions with -100
                num_masked = len(prompt_ids) - 1
                y[:num_masked] = [-100] * num_masked

                self.samples.append((x, y))

        print(f"Loaded {len(self.samples):,} SFT examples from {jsonl_path.name} (skipped {skipped})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def make_collate_fn(pad_id: int):
    def collate(batch: list[tuple[list[int], list[int]]]):
        max_len = max(len(x) for x, _ in batch)
        # Round up to multiple of 8 for Tensor Core efficiency
        max_len = ((max_len + 7) // 8) * 8
        xs, ys = [], []
        for x, y in batch:
            pad_n = max_len - len(x)
            xs.append(x + [pad_id] * pad_n)
            ys.append(y + [-100] * pad_n)  # pad positions are ALSO masked with -100!
        return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)

    return collate


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def evaluate_sft(model, val_loader, device: str, amp_ctx, max_batches: int = 30) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx:
            _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def run_dual_mode_probes(model, tok: LinuxTokenizer, device: str, arch: str):
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    probes = [
        ("Q&A", "<|qa|><|user|>How do I find large files in Linux?<|assistant|>", 48, 0.3),
        ("COMP", "<|complete|><|ctx|>files=docker-compose.yml<|cmd|>docker compose <|pred|>", 16, 0.0),
        ("COMP", "<|complete|><|cmd|>tar -xzf <|pred|>", 16, 0.0),
    ]
    print(f"  --- [{arch}] Live Dual-Mode Probes ---", flush=True)
    for label, prompt, max_toks, temp in probes:
        ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
        out = raw_model.generate(
            ids,
            max_new_tokens=max_toks,
            temperature=temp,
            top_p=0.9 if temp > 0 else None,
            stop_token_id=tok.end_id,
        )[0].tolist()
        gen_only = tok.decode(out[ids.size(1) :], skip_special_tokens=True).replace("\n", "\\n")
        print(f"    [{label}] {prompt} ==> {gen_only[:120]}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--preset", choices=["tiny", "15m", "50m", "124m"], default="15m")
    parser.add_argument("--init_ckpt", type=Path, required=True)
    parser.add_argument("--sft_dir", type=Path, default=Path("artifacts/sft_v2"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1200)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(f"runs/sft_{args.arch}_{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
    else:
        from contextlib import nullcontext
        amp_dtype = torch.float32
        amp_ctx = nullcontext()
        scaler = torch.amp.GradScaler("cuda", enabled=False)

    tok = LinuxTokenizer(args.tokenizer)
    model = build_model(
        arch=args.arch,
        preset=args.preset,
        vocab_size=tok.vocab_size,
        dropout=args.dropout,
    ).to(device)

    ckpt = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded pretrained weights from {args.init_ckpt} (step={ckpt.get('step')})", flush=True)

    train_ds = DualModeSFTDataset(args.sft_dir / "sft_train.jsonl", tok, model.cfg.block_size)
    val_ds = DualModeSFTDataset(args.sft_dir / "sft_val.jsonl", tok, model.cfg.block_size)

    collate = make_collate_fn(tok.pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device_type=device_type,
    )

    print(
        f"\n=== Stage 2 Dual-Mode SFT: {args.arch.upper()} ({args.preset}) ===\n"
        f"  Effective batch: {args.batch_size * args.grad_accum}\n"
        f"  Max steps:       {args.max_steps}\n"
        f"  Output dir:      {out_dir}\n",
        flush=True,
    )

    metrics_path = out_dir / "metrics_sft.jsonl"
    best_val = float("inf")
    data_iter = iter(train_loader)
    t_log = time.perf_counter()

    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            with amp_ctx:
                _, loss, _ = model(x, targets=y)
                loss = loss / args.grad_accum
            accum_loss += loss.item()
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            dt = time.perf_counter() - t_log
            print(
                f"[{args.arch}-sft] step {step:4d}/{args.max_steps} | "
                f"loss {accum_loss:.4f} | lr {lr:.2e} | {dt:.1f}s",
                flush=True,
            )
            t_log = time.perf_counter()

        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            val_loss = evaluate_sft(model, val_loader, device, amp_ctx)
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({"step": step, "val_loss": round(val_loss, 4), "lr": lr}) + "\n")
            print(f"  >>> [{args.arch}-sft] EVAL step {step}: val_loss={val_loss:.4f}", flush=True)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "arch": args.arch,
                        "preset": args.preset,
                        "vocab_size": model.cfg.vocab_size,
                        "block_size": model.cfg.block_size,
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "best_val": best_val,
                    },
                    out_dir / "sft_best.pt",
                )

            run_dual_mode_probes(model, tok, device, args.arch)
            t_log = time.perf_counter()

    print(f"\nStage 2 SFT complete! Best val_loss={best_val:.4f} -> {out_dir / 'sft_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
