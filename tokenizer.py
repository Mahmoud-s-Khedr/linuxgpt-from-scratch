"""
Custom 16,384-vocab ByteLevel BPE Tokenizer for LinuxGPT.

Includes dedicated special control tokens for our dual-mode CLI protocol:
  Mode 1 (Q&A):
    <|qa|><|user|>How do I find large files?<|assistant|>...<|end|>
  Mode 2 (Inline Autocomplete):
    <|complete|><|ctx|>files=docker-compose.yml<|cmd|>docker compose <|pred|>up -d<|end|>

Usage:
  python tokenizer.py --data_dir data/sources --out artifacts/tokenizer.json
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS = [
    "<|pad|>",
    "<|eos|>",
    "<|qa|>",
    "<|complete|>",
    "<|ctx|>",
    "<|cmd|>",
    "<|pred|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
]


class LinuxTokenizer:
    """Thin, fast wrapper around HuggingFace's Rust BPE tokenizer."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._tok = Tokenizer.from_file(self.path)
        self.vocab_size = self._tok.get_vocab_size()
        self.pad_id = self.token_to_id("<|pad|>")
        self.eos_id = self.token_to_id("<|eos|>")
        self.end_id = self.token_to_id("<|end|>")

    def token_to_id(self, token: str) -> int:
        tid = self._tok.token_to_id(token)
        if tid is None:
            raise KeyError(f"Token {token!r} not in vocabulary")
        return tid

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)


def sample_corpus_iterator(data_dir: Path, max_bytes_per_source: int = 8_000_000):
    """
    Yields balanced chunks across all sources so CLI cheatsheets, tldr, scripts,
    man pages, and Q&As all shape the BPE vocabulary evenly.
    """
    files = sorted(data_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    for fp in files:
        read_bytes = 0
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            buf = []
            for line in f:
                buf.append(line)
                read_bytes += len(line.encode("utf-8"))
                if len(buf) >= 200:
                    yield "".join(buf)
                    buf.clear()
                if read_bytes >= max_bytes_per_source:
                    break
            if buf:
                yield "".join(buf)


def train_tokenizer(
    data_dir: Path,
    out_path: Path,
    vocab_size: int = 16384,
    max_bytes_per_source: int = 8_000_000,
) -> LinuxTokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training {vocab_size}-vocab Linux BPE tokenizer from {data_dir} ...")
    tok.train_from_iterator(
        sample_corpus_iterator(data_dir, max_bytes_per_source=max_bytes_per_source),
        trainer=trainer,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    print(f"Saved tokenizer to {out_path} (vocab_size={tok.get_vocab_size()})")
    return LinuxTokenizer(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/sources"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--vocab_size", type=int, default=16384)
    parser.add_argument("--max_mb_per_source", type=int, default=8)
    args = parser.parse_args()

    lt = train_tokenizer(
        data_dir=args.data_dir,
        out_path=args.out,
        vocab_size=args.vocab_size,
        max_bytes_per_source=args.max_mb_per_source * 1_000_000,
    )

    # Quick smoke test on both CLI modes
    samples = [
        "<|qa|><|user|>How do I find large files?<|assistant|>Use `find / -type f -size +100M`<|end|>",
        "<|complete|><|ctx|>files=docker-compose.yml<|cmd|>docker compose <|pred|>up -d --build<|end|>",
        "systemctl status nginx.service && journalctl -u nginx -n 50 --no-pager",
    ]
    print("\n=== Tokenizer Smoke Test ===")
    for s in samples:
        ids = lt.encode(s)
        rt = lt.decode(ids, skip_special_tokens=False)
        assert rt == s, f"Roundtrip mismatch:\n  orig: {s!r}\n  back: {rt!r}"
        print(f"  [{len(ids):2d} toks] {s}")


if __name__ == "__main__":
    main()
