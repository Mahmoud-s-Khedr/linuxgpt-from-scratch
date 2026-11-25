"""
LinuxGPT CLI — Dual-Mode Terminal Assistant (Q&A + Context-Aware Autocomplete).

Supports all 4 trained scratch models (llama_15m, gpt2_15m, llama_50m, gpt2_50m)
with KV-cached streaming generation on local CPU.

Usage:
  # 1. Question Answering (streaming):
  python cli.py "How do I find large files?"

  # 2. Context-Aware Inline Autocomplete (reads real cwd files for <|ctx|>):
  python cli.py -c "docker compose "

  # 3. Interactive Warm REPL (model stays in RAM -> fast responses):
  python cli.py --chat

  # 4. Side-by-Side Model Comparison (GPT-2 vs LLaMA, 15m vs 50m):
  python cli.py --compare -c "systemctl status "
  python cli.py --compare "How do I extract a tar.gz file?"
"""

from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from models import build_model
from tokenizer import LinuxTokenizer

BASE_DIR = Path(__file__).resolve().parent
TOKENIZER_PATH = BASE_DIR / "artifacts" / "tokenizer.json"
RUNS_DIR = BASE_DIR / "runs"

MODEL_PRIORITY = ["llama_50m", "gpt2_50m", "llama_15m", "gpt2_15m"]
KEY_CONTEXT_FILES = [
    "docker-compose.yml", "docker-compose.yaml", "compose.yml", "Dockerfile",
    "Makefile", "CMakeLists.txt", "Cargo.toml", "go.mod", "pyproject.toml",
    "package.json", "requirements.txt", "nginx.conf",
]


def discover_models(stage: str = "sft") -> dict[str, Path]:
    fname = "sft_bf16.pt" if stage == "sft" else "pretrain_bf16.pt"
    found: dict[str, Path] = {}
    for name in MODEL_PRIORITY:
        p = RUNS_DIR / f"colab_{name}" / fname
        if p.exists():
            found[name] = p
    return found


def detect_cwd_context() -> str:
    """Inspect real current directory for lightweight, high-signal CLI context."""
    parts: list[str] = []
    cwd = Path.cwd()
    if (cwd / ".git").exists():
        parts.append("git=true")
    present = [f for f in KEY_CONTEXT_FILES if (cwd / f).exists()]
    if present:
        parts.append("files=" + ",".join(present[:4]))
    return " ".join(parts)


def check_bash_syntax(cmd: str) -> bool | None:
    """Run `bash -n` (syntax-only, never executes) on a completed command."""
    try:
        res = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=2)
        return res.returncode == 0
    except Exception:
        return None


def load_model(ckpt_path: Path, device: str = "cpu"):
    t0 = time.perf_counter()
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["arch"], ck["preset"], vocab_size=ck["vocab_size"]).to(device)
    # Cast bf16 weights to fp32 on CPU for fast standard matmuls (or keep bf16 on CUDA)
    state = {
        k: (v.float() if (device == "cpu" and v.is_floating_point()) else v)
        for k, v in ck["model_state_dict"].items()
    }
    model.load_state_dict(state)
    model.eval()
    load_ms = (time.perf_counter() - t0) * 1000
    return model, ck, load_ms


@torch.no_grad()
def stream_generate(
    model,
    tok: LinuxTokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.3,
    top_p: float = 0.9,
    rep_penalty: float = 1.15,
    stream: bool = True,
    device: str = "cpu",
) -> tuple[str, dict]:
    ids = tok.encode(prompt)
    if len(ids) > model.cfg.block_size - 1:
        ids = ids[-(model.cfg.block_size - 1) :]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    t_prefill0 = time.perf_counter()
    logits, _, kv_cache = model(idx)
    prefill_ms = (time.perf_counter() - t_prefill0) * 1000

    gen_ids: list[int] = []
    printed_len = 0
    t_dec0 = time.perf_counter()

    for _ in range(max_new_tokens):
        if len(ids) + len(gen_ids) >= model.cfg.block_size:
            break
        next_logits = logits[:, -1, :].clone()

        # Repetition penalty (Keskar et al. / CTRL): penalize recently generated tokens
        if rep_penalty > 1.0 and gen_ids:
            recent = set(gen_ids[-48:])
            for tid in recent:
                score = next_logits[0, tid]
                next_logits[0, tid] = score / rep_penalty if score > 0 else score * rep_penalty

        if temperature <= 0.0:
            next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cum_probs > top_p
                remove_mask[:, 1:] = remove_mask[:, :-1].clone()
                remove_mask[:, 0] = False
                to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
                next_logits[to_remove] = float("-inf")
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        tid = int(next_id.item())
        if tid in (tok.end_id, tok.eos_id):
            break
        gen_ids.append(tid)

        if stream:
            full_text = tok.decode(gen_ids, skip_special_tokens=True)
            delta = full_text[printed_len:]
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
                printed_len = len(full_text)

        logits, _, kv_cache = model(next_id, kv_cache=kv_cache)

    decode_s = max(time.perf_counter() - t_dec0, 1e-6)
    final_text = tok.decode(gen_ids, skip_special_tokens=True)
    if stream:
        sys.stdout.write("\n")
        sys.stdout.flush()

    stats = {
        "prefill_ms": prefill_ms,
        "decode_ms": decode_s * 1000,
        "total_ms": prefill_ms + decode_s * 1000,
        "tokens": len(gen_ids),
        "tok_per_s": len(gen_ids) / decode_s if gen_ids else 0.0,
        "kv_kb": model.kv_cache_bytes(dtype_bytes=2) / 1024.0,
    }
    return final_text, stats


def format_prompt(text: str, complete: bool, ctx: str | None) -> str:
    if complete:
        if ctx:
            return f"<|complete|><|ctx|>{ctx}<|cmd|>{text}<|pred|>"
        return f"<|complete|><|cmd|>{text}<|pred|>"
    return f"<|qa|><|user|>{text}<|assistant|>"


def run_chat_repl(model, tok: LinuxTokenizer, model_name: str, device: str):
    print(f"LinuxGPT Interactive Mode ({model_name} loaded warm in RAM)")
    print("  • Type a Linux question for Q&A")
    print("  • Prefix with '/c ' for inline autocomplete (e.g. /c tar -xzf )")
    print("  • Type 'exit' or Ctrl-D to quit\n")

    while True:
        try:
            line = input("linuxgpt> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            continue
        if line.strip().lower() in ("exit", "quit", "/q"):
            break

        if line.startswith("/c "):
            prefix = line[3:]
            ctx = detect_cwd_context()
            prompt = format_prompt(prefix, complete=True, ctx=ctx)
            sys.stdout.write(prefix)
            sys.stdout.flush()
            out, stats = stream_generate(
                model, tok, prompt, max_new_tokens=24, temperature=0.0, stream=True, device=device
            )
            valid = check_bash_syntax(prefix + out)
            badge = "bash -n: OK" if valid else ("bash -n: WARN" if valid is False else "")
            print(f"  [{stats['total_ms']:.0f} ms | {stats['tok_per_s']:.1f} tok/s | {badge}]\n")
        else:
            prompt = format_prompt(line.strip(), complete=False, ctx=None)
            _, stats = stream_generate(
                model, tok, prompt, max_new_tokens=160, temperature=0.3, top_p=0.9, stream=True, device=device
            )
            print(f"  [{stats['total_ms']:.0f} ms | {stats['tokens']} toks | {stats['tok_per_s']:.1f} tok/s]\n")


def main():
    parser = argparse.ArgumentParser(description="LinuxGPT CLI (Q&A + Context-Aware Autocomplete)")
    parser.add_argument("query", nargs="?", default=None, help="Question or command prefix")
    parser.add_argument("-c", "--complete", action="store_true", help="Inline autocomplete mode")
    parser.add_argument("--ctx", default=None, help="Override <|ctx|> string (default: auto-detect cwd)")
    parser.add_argument("--no-ctx", action="store_true", help="Disable auto cwd context detection")
    parser.add_argument("-m", "--model", default=None, help="Model name: llama_50m, gpt2_50m, llama_15m, gpt2_15m")
    parser.add_argument("--checkpoint", type=Path, help="Load a specific checkpoint instead of discovering Colab downloads")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH, help="Tokenizer JSON used to train the checkpoint")
    parser.add_argument("--stage", choices=["sft", "pretrain"], default="sft")
    parser.add_argument("--compare", action="store_true", help="Compare all downloaded models side-by-side")
    parser.add_argument("--chat", action="store_true", help="Interactive warm REPL")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temp", type=float, default=None)
    parser.add_argument("--rep-penalty", type=float, default=None, help="Repetition penalty (default: 1.15 for Q&A, 1.05 for complete)")
    parser.add_argument("--stats", action="store_true", help="Print latency and throughput stats")
    parser.add_argument("--list", action="store_true", help="List downloaded models")
    args = parser.parse_args()

    available = {args.model or "custom": args.checkpoint} if args.checkpoint else discover_models(stage=args.stage)
    if args.list:
        print("Available downloaded models:")
        for name, p in available.items():
            print(f"  - {name:<10s} ({p.stat().st_size / 1e6:.1f} MB)  {p}")
        return

    if not available:
        print(f"No checkpoints found in {RUNS_DIR} for stage={args.stage}.", file=sys.stderr)
        sys.exit(1)

    tok = LinuxTokenizer(args.tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.chat:
        name = args.model or next(iter(available))
        if name not in available:
            print(f"Model '{name}' not found. Available: {list(available.keys())}", file=sys.stderr)
            sys.exit(1)
        model, _, load_ms = load_model(available[name], device=device)
        print(f"Loaded {name} in {load_ms:.0f} ms.")
        run_chat_repl(model, tok, name, device)
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    ctx = "" if args.no_ctx else (args.ctx if args.ctx is not None else (detect_cwd_context() if args.complete else ""))
    prompt = format_prompt(args.query, complete=args.complete, ctx=ctx)
    max_toks = args.max_tokens or (24 if args.complete else 128)
    temp = args.temp if args.temp is not None else (0.0 if args.complete else 0.3)
    rep_pen = args.rep_penalty if args.rep_penalty is not None else (1.05 if args.complete else 1.18)

    if args.compare:
        mode_lbl = "AUTOCOMPLETE" if args.complete else "Q&A"
        print(f"=== Comparing {len(available)} models [{mode_lbl}] ===")
        print(f"Prompt: {prompt}\n")
        for name, ckpt_path in available.items():
            model, _, _ = load_model(ckpt_path, device=device)
            out, stats = stream_generate(
                model, tok, prompt, max_new_tokens=max_toks, temperature=temp, rep_penalty=rep_pen, stream=False, device=device
            )
            display = (args.query + out) if args.complete else out
            print(
                f"── {name:<10s} | {stats['total_ms']:6.0f} ms | "
                f"{stats['tok_per_s']:5.1f} tok/s | KV={stats['kv_kb']:.0f} KB ──"
            )
            print(display)
            print()
        return

    name = args.model or next(iter(available))
    if name not in available:
        print(f"Model '{name}' not found. Available: {list(available.keys())}", file=sys.stderr)
        sys.exit(1)

    model, _, load_ms = load_model(available[name], device=device)
    if args.complete:
        sys.stdout.write(args.query)
        sys.stdout.flush()
    out, stats = stream_generate(
        model, tok, prompt, max_new_tokens=max_toks, temperature=temp, rep_penalty=rep_pen, stream=True, device=device
    )
    if args.stats:
        extra = ""
        if args.complete:
            ok = check_bash_syntax(args.query + out)
            extra = f" | bash -n={'OK' if ok else 'WARN'}"
        print(
            f"\n[{name} | load={load_ms:.0f}ms | prefill={stats['prefill_ms']:.0f}ms | "
            f"decode={stats['decode_ms']:.0f}ms ({stats['tok_per_s']:.1f} tok/s){extra}]"
        )


if __name__ == "__main__":
    main()
