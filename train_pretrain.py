"""
Stage 1 Pretraining Script (supports both --arch gpt2 and --arch llama).

Features:
  - Memory-mapped uint16 train.bin / val.bin loading (instant startup, low RAM)
  - Automatic bf16 AMP on A100/L4/H100 (fp16 on T4, fp32 on CPU)
  - Gradient accumulation, cosine LR schedule with linear warmup, grad clipping
  - Optional torch.compile for ~1.5-2x throughput boost on GPU
  - Periodic val loss evaluation + sample generation + JSONL metrics logging
  - Automatic checkpoint saving (latest.pt, best.pt) and --resume support

Example (local tiny test):
  python train_pretrain.py --arch llama --preset tiny --max_steps 100 --batch_size 8 --grad_accum 1 --eval_interval 25

Example (Colab L4/A100 50m run):
  python train_pretrain.py --arch llama --preset 50m --max_steps 3000 --batch_size 32 --grad_accum 4 --compile
"""

from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from models import build_model
from tokenizer import LinuxTokenizer


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model,
    train_data: np.memmap,
    val_data: np.memmap,
    eval_iters: int,
    batch_size: int,
    block_size: int,
    device: str,
    amp_ctx,
) -> dict[str, float]:
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, block_size, device)
            with amp_ctx:
                _, loss, _ = model(x, targets=y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(path: Path, model, optimizer, step: int, best_val: float, args_dict: dict):
    # Unwrap torch.compile if present
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "arch": args_dict["arch"],
            "preset": args_dict["preset"],
            "vocab_size": raw_model.cfg.vocab_size,
            "block_size": raw_model.cfg.block_size,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val": best_val,
            "args": args_dict,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--preset", choices=["tiny", "15m", "50m", "124m"], default="15m")
    parser.add_argument("--data_dir", type=Path, default=Path("artifacts/pretrain"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(f"runs/pretrain_{args.arch}_{args.preset}")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        gpu_name = torch.cuda.get_device_name(0)
    else:
        amp_dtype = torch.float32
        from contextlib import nullcontext
        amp_ctx = nullcontext()
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        gpu_name = "CPU"

    tok = LinuxTokenizer(args.tokenizer)
    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")

    model = build_model(
        arch=args.arch,
        preset=args.preset,
        vocab_size=tok.vocab_size,
        dropout=args.dropout,
    ).to(device)
    block_size = model.cfg.block_size

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        device_type=device_type,
    )

    start_step = 0
    best_val = float("inf")
    latest_ckpt = out_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt["step"] + 1
        best_val = ckpt.get("best_val", float("inf"))
        print(f"Resumed {args.arch} from {latest_ckpt} at step {start_step} (best_val={best_val:.4f})")

    if args.compile and device_type == "cuda":
        print("Compiling model with torch.compile() ...")
        model = torch.compile(model)

    tokens_per_step = args.batch_size * args.grad_accum * block_size
    print(
        f"\n=== Stage 1 Pretraining: {args.arch.upper()} ({args.preset}) ===\n"
        f"  Device:          {gpu_name} (amp={amp_dtype})\n"
        f"  Params:          {model.num_parameters():,}\n"
        f"  KV Cache (fp16): {model.kv_cache_bytes(dtype_bytes=2) / 1024:.1f} KB\n"
        f"  Block size:      {block_size}\n"
        f"  Tokens/step:     {tokens_per_step:,} (total planned: {tokens_per_step * args.max_steps:,})\n"
        f"  Train/Val toks:  {len(train_data):,} / {len(val_data):,}\n"
        f"  Output dir:      {out_dir}\n",
        flush=True,
    )

    metrics_path = out_dir / "metrics_pretrain.jsonl"
    t_log = time.perf_counter()

    for step in range(start_step, args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_data, args.batch_size, block_size, device)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        if step % args.log_interval == 0 or step == args.max_steps - 1:
            dt = time.perf_counter() - t_log
            steps_done = args.log_interval if step > start_step else 1
            tok_per_s = (tokens_per_step * steps_done) / max(dt, 1e-6)
            print(
                f"[{args.arch}] step {step:5d}/{args.max_steps} | "
                f"loss {accum_loss:.4f} | lr {lr:.2e} | {tok_per_s:,.0f} tok/s",
                flush=True,
            )
            t_log = time.perf_counter()

        if (step > 0 and step % args.eval_interval == 0) or step == args.max_steps - 1:
            losses = estimate_loss(
                model, train_data, val_data, args.eval_iters,
                args.batch_size, block_size, device, amp_ctx,
            )
            rec = {
                "step": step,
                "tokens_seen": (step + 1) * tokens_per_step,
                "train_loss": round(losses["train"], 4),
                "val_loss": round(losses["val"], 4),
                "lr": lr,
            }
            with metrics_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(rec) + "\n")

            print(
                f"  >>> [{args.arch}] EVAL step {step}: "
                f"train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}",
                flush=True,
            )

            args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
            save_checkpoint(latest_ckpt, model, optimizer, step, min(best_val, losses["val"]), args_dict)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "best.pt", model, optimizer, step, best_val, args_dict)

            # Quick sample generation
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            prompt_ids = torch.tensor([tok.encode("find /var/log ")], dtype=torch.long, device=device)
            gen_ids = raw_model.generate(prompt_ids, max_new_tokens=32, temperature=0.7, top_p=0.9)[0].tolist()
            sample_txt = tok.decode(gen_ids, skip_special_tokens=True).replace("\n", "\\n")
            print(f"  >>> [{args.arch}] SAMPLE: {sample_txt[:140]}", flush=True)
            t_log = time.perf_counter()

    print(f"\nStage 1 Pretraining complete! Best val_loss={best_val:.4f} -> {out_dir / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
