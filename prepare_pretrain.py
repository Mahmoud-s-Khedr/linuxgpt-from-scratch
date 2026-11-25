"""
Stage 1 Pretraining Data Preparation.

Reads all 18 cleaned source files in data/sources/, performs:
  1. Document-level splitting on '\\n---\\n'
  2. Stratified 2% validation holdout from EVERY source (zero source bias in val)
  3. Source-weighted upsampling (3x tldr/scripts/cheatsheets/books, 2x docs/wiki, 1x man/QA)
  4. Global document shuffling + encoding with our 16k Linux BPE tokenizer
  5. Streaming into uint16 numpy memmaps: artifacts/pretrain/train.bin & val.bin

Usage:
  python prepare_pretrain.py \
      --data_dir data/sources \
      --tokenizer artifacts/tokenizer.json \
      --out_dir artifacts/pretrain
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

from tokenizer import LinuxTokenizer

# Upsample high-density CLI/docs/books so raw Q&A (75% of bytes) doesn't drown out syntax
SOURCE_WEIGHTS: dict[str, int] = {
    "05_tldr_pages.txt": 3,
    "16_shell_scripts_and_configs.txt": 3,
    "17_local_linux_books.txt": 3,
    "18_cli_cheatsheets.txt": 3,
    "04_linux_kernel_docs.txt": 2,
    "06_arch_wiki.txt": 2,
    "07_ubuntu_server_docs.txt": 2,
    "08_systemd_git_neovim_docs.txt": 2,
    "09_docker_k8s_ansible_docs.txt": 2,
    "10_linux_infra_networking_docs.txt": 2,
    "14_linux_books_and_guides.txt": 2,
    # Remaining sources default to 1x (man pages, gnu info, usr/share/doc, StackExchange Q&As)
}


def iter_documents(filepath: Path):
    """Stream documents separated by a line containing only '---'."""
    buf: list[str] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "---":
                doc = "".join(buf).strip()
                if len(doc) >= 40:
                    yield doc
                buf.clear()
            else:
                buf.append(line)
        doc = "".join(buf).strip()
        if len(doc) >= 40:
            yield doc


def write_bin(
    docs: list[str],
    tok: LinuxTokenizer,
    out_path: Path,
    max_tokens: int | None = None,
    desc: str = "Encoding",
) -> int:
    """Encode documents and stream directly to a uint16 binary file."""
    assert tok.vocab_size <= 65535, "uint16 requires vocab_size <= 65535"
    eos = tok.eos_id
    total_tokens = 0
    chunk: list[int] = []
    FLUSH_SIZE = 1_000_000

    with out_path.open("wb") as f:
        for doc in tqdm(docs, desc=desc, unit="doc"):
            ids = tok.encode(doc)
            ids.append(eos)
            chunk.extend(ids)
            total_tokens += len(ids)

            if len(chunk) >= FLUSH_SIZE:
                np.asarray(chunk, dtype=np.uint16).tofile(f)
                chunk.clear()

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        if chunk:
            np.asarray(chunk, dtype=np.uint16).tofile(f)

    return total_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/sources"))
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/pretrain"))
    parser.add_argument("--val_ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_tokens",
        type=int,
        default=None,
        help="Optional cap on train tokens (e.g. 150000000 for ~150M token budget).",
    )
    parser.add_argument(
        "--max_val_tokens",
        type=int,
        default=4_000_000,
        help="Cap on val tokens so val evaluation stays fast.",
    )
    args = parser.parse_args()

    tok = LinuxTokenizer(args.tokenizer)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_docs: list[str] = []
    val_docs: list[str] = []
    source_stats: dict[str, dict[str, int]] = {}

    files = sorted(args.data_dir.glob("*.txt"))
    print(f"Scanning {len(files)} source files in {args.data_dir} ...")

    for fp in files:
        docs = list(iter_documents(fp))
        rng.shuffle(docs)
        n_val = max(1, int(len(docs) * args.val_ratio))
        v_docs = docs[:n_val]
        t_docs = docs[n_val:]

        weight = SOURCE_WEIGHTS.get(fp.name, 1)
        val_docs.extend(v_docs)
        for _ in range(weight):
            train_docs.extend(t_docs)

        source_stats[fp.name] = {
            "total_docs": len(docs),
            "val_docs": len(v_docs),
            "train_docs_weighted": len(t_docs) * weight,
            "weight": weight,
        }
        print(
            f"  {fp.name:<38s} docs={len(docs):>7,}  weight={weight}x  "
            f"train={len(t_docs) * weight:>7,}  val={len(v_docs):>5,}"
        )

    print("\nShuffling train and validation documents globally ...")
    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    train_bin = args.out_dir / "train.bin"
    val_bin = args.out_dir / "val.bin"

    n_val_toks = write_bin(
        val_docs, tok, val_bin, max_tokens=args.max_val_tokens, desc="Encoding val.bin"
    )
    n_train_toks = write_bin(
        train_docs, tok, train_bin, max_tokens=args.max_train_tokens, desc="Encoding train.bin"
    )

    meta = {
        "vocab_size": tok.vocab_size,
        "tokenizer_path": str(args.tokenizer),
        "train_tokens": n_train_toks,
        "val_tokens": n_val_toks,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "sources": source_stats,
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nDone!")
    print(f"  train.bin: {n_train_toks:>12,} tokens ({train_bin.stat().st_size / 1e6:.1f} MB)")
    print(f"  val.bin:   {n_val_toks:>12,} tokens ({val_bin.stat().st_size / 1e6:.1f} MB)")
    print(f"  meta.json: {meta_path}")


if __name__ == "__main__":
    main()
