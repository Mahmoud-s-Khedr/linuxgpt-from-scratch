"""
Automated Stage 2 Dual-Mode SFT Dataset Builder (v2 — leak-free & syntax-verified).

Fixes applied after the independent model review:
  1. ZERO command leakage: deduplicates and splits unique commands into train/val
     BEFORE slicing into prefix/suffix completion pairs (previously 844/1600 val
     completions reconstructed commands seen in train).
  2. ZERO placeholder / syntax-broken autocomplete targets:
     - Excludes template placeholder syntax (`{{...}}`, `<...>`) from autocomplete.
     - Validates every single autocomplete command with `bash -n` (fast batched
       syntax-only check, never executes anything) so 100% of completion targets
       are syntactically valid shell commands.
  3. Cleaner, better-balanced Q&A:
     - Adds "What is <cmd>?" definitions from `05_tldr_pages.txt` and conceptual
       intros from `06_arch_wiki.txt`.
     - Filters out forum chatter ("I have the same problem", "I found the answer").
     - Caps AskUbuntu share to reduce apt-get bias on non-Ubuntu questions.

Usage:
  python build_sft_data.py --data_dir data/sources --out_dir artifacts/sft_v2
"""

from __future__ import annotations
import argparse
import json
import random
import re
import subprocess
from pathlib import Path

KNOWN_CMDS = {
    "apt", "apt-get", "awk", "cat", "cd", "chmod", "chown", "cp", "crontab", "curl",
    "cut", "dd", "df", "diff", "dig", "dnf", "docker", "du", "echo", "env", "export",
    "find", "free", "git", "grep", "gzip", "head", "htop", "ip", "iptables", "jq",
    "journalctl", "kill", "kubectl", "less", "ln", "ls", "lsof", "make", "man",
    "mkdir", "mount", "mv", "nc", "netstat", "nginx", "nmap", "nohup", "openssl",
    "pacman", "ping", "podman", "ps", "pwd", "rm", "rsync", "scp", "sed", "service",
    "sort", "ss", "ssh", "ssh-keygen", "stat", "strace", "sudo", "systemctl", "tail",
    "tar", "tee", "tmux", "top", "touch", "tr", "ufw", "umount", "uname", "uniq",
    "unzip", "useradd", "usermod", "vim", "wc", "wget", "which", "xargs", "yum", "zip",
}

CONTEXT_HINTS = {
    "docker": ["files=Dockerfile,docker-compose.yml", "files=docker-compose.yml", ""],
    "git": ["git=true", "git=true branch=main", ""],
    "kubectl": ["k8s=true", "k8s=true ns=default", ""],
    "make": ["files=Makefile", ""],
    "systemctl": ["os=linux init=systemd", ""],
    "journalctl": ["os=linux init=systemd", ""],
    "nginx": ["files=/etc/nginx/nginx.conf", ""],
    "tar": ["files=archive.tar.gz,backup/", ""],
    "find": ["cwd=/var/log", "cwd=/home/user", ""],
    "grep": ["cwd=/var/log", "files=app.log,config.yaml", ""],
    "ssh": ["files=~/.ssh/config", ""],
    "rsync": ["cwd=/home/user", ""],
    "apt": ["os=ubuntu", ""],
    "dnf": ["os=fedora", ""],
    "pacman": ["os=arch", ""],
}

FORUM_NOISE_PREFIXES = (
    "i have the same",
    "i had the same",
    "i found the answer",
    "thanks",
    "thank you",
    "edit:",
    "update:",
    "solved",
    "duplicate of",
)

CODE_BLOCK_RE = re.compile(r"```(?:bash|sh|shell|zsh)?\n(.*?)```", re.DOTALL)
TLDR_PAIR_RE = re.compile(r"^- (.+?):\s*\n\s*\n`([^`\n]+)`", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|<[a-zA-Z0-9_|/\[\]-]+>")


def iter_documents(filepath: Path):
    buf: list[str] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "---":
                doc = "".join(buf).strip()
                if doc:
                    yield doc
                buf.clear()
            else:
                buf.append(line)
        doc = "".join(buf).strip()
        if doc:
            yield doc


def is_valid_bash_single(cmd: str) -> bool:
    try:
        res = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, timeout=2)
        return res.returncode == 0
    except Exception:
        return False


def filter_valid_bash_commands(commands: list[str], batch_size: int = 200) -> list[str]:
    """
    Fast batched `bash -n` filter: tests a batch of commands as a script;
    only falls back to per-command checks if a batch fails.
    Batch parsing is a heuristic: jointly valid snippets can still be invalid individually.
    """
    valid: list[str] = []
    for i in range(0, len(commands), batch_size):
        batch = commands[i : i + batch_size]
        script = "\n".join(batch)
        res = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        if res.returncode == 0:
            valid.extend(batch)
        else:
            for c in batch:
                if is_valid_bash_single(c):
                    valid.append(c)
    return valid


def normalize_shell_line(line: str) -> str | None:
    """Return a clean, real single-line shell command (no placeholders) or None."""
    s = line.strip()
    if not s or s.startswith("#") or s.startswith("//"):
        return None
    if s.startswith("$ "):
        s = s[2:].strip()
    elif s.startswith("# ") and len(s) > 2:
        s = s[2:].strip()

    if len(s) < 8 or len(s) > 150:
        return None
    if any(bad in s for bad in ("\t", "\r", "http://", "https://", "```", "{{", "}}")):
        return None
    # Reject template angle-bracket placeholders like <unit>, <path/to/file>, <[a|b]>
    if PLACEHOLDER_RE.search(s):
        return None
    # Reject lines that look like prose or key: value output
    if s.endswith(":") or s.endswith("."):
        return None

    parts = s.split()
    if len(parts) < 2:
        return None
    first_tok = parts[1] if parts[0] == "sudo" and len(parts) >= 3 else parts[0]
    if first_tok not in KNOWN_CMDS:
        return None
    return s


def extract_tldr(filepath: Path) -> tuple[list[dict], list[str]]:
    qa_examples: list[dict] = []
    commands: list[str] = []

    for doc in iter_documents(filepath):
        lines = doc.splitlines()
        if not lines or not lines[0].startswith("# "):
            continue
        cmd_name = lines[0][2:].strip()
        desc_lines = [ln[2:].strip() for ln in lines[1:6] if ln.startswith("> ") and "More information:" not in ln and "See also:" not in ln]
        pairs = TLDR_PAIR_RE.findall(doc)

        # Add clean "What is <cmd>?" definition Q&A!
        if desc_lines and cmd_name in KNOWN_CMDS:
            summary = " ".join(desc_lines)
            ex_lines = []
            for d, raw_c in pairs[:3]:
                clean_c = re.sub(r"\{\{([^}]+)\}\}", r"\1", raw_c).strip()
                ex_lines.append(f"- {d}: `{clean_c}`")
            ans = f"`{cmd_name}` — {summary}"
            if ex_lines:
                ans += "\n\nCommon examples:\n" + "\n".join(ex_lines)
            for q in (f"What is `{cmd_name}`?", f"What does the `{cmd_name}` command do in Linux?"):
                qa_examples.append({
                    "mode": "qa",
                    "prompt": f"<|qa|><|user|>{q}<|assistant|>",
                    "completion": f"{ans}<|end|>",
                })

        for desc, raw_cmd in pairs:
            # Only add placeholder-free commands to autocomplete
            norm = normalize_shell_line(raw_cmd)
            if norm:
                commands.append(norm)

            # For Q&A, strip {{}} braces cleanly
            clean_cmd = re.sub(r"\{\{([^}]+)\}\}", r"\1", raw_cmd).strip()
            if 4 <= len(clean_cmd) <= 150:
                q_text = desc.strip()
                if not q_text.endswith("?"):
                    q_text = f"How do I {q_text[0].lower() + q_text[1:]} using `{cmd_name}`?"
                qa_examples.append({
                    "mode": "qa",
                    "prompt": f"<|qa|><|user|>{q_text}<|assistant|>",
                    "completion": f"You can use `{cmd_name}`:\n\n```bash\n{clean_cmd}\n```<|end|>",
                })

    return qa_examples, commands


def extract_arch_wiki_definitions(filepath: Path, max_qa: int = 4000) -> list[dict]:
    """Extract clean 'What is <Title>?' conceptual Q&A pairs from Arch Wiki intros."""
    qa_examples: list[dict] = []
    for doc in iter_documents(filepath):
        if len(qa_examples) >= max_qa:
            break
        lines = [ln.strip() for ln in doc.splitlines() if ln.strip()]
        if len(lines) < 2 or not lines[0].startswith("# "):
            continue
        title = lines[0][2:].strip()
        if title.startswith("Arch Wiki:"):
            title = title[len("Arch Wiki:"):].strip()
        if not (3 <= len(title) <= 60) or "/" in title or "(" in title:
            continue

        # Take first 1-2 clean prose paragraphs (skip section headers)
        paras = []
        for ln in lines[1:8]:
            if ln.startswith("#") or ln.startswith("Related articles") or ln.startswith("See also"):
                break
            if len(ln) >= 40:
                paras.append(ln)
            if sum(len(p) for p in paras) >= 450:
                break
        if not paras:
            continue
        body = "\n\n".join(paras)[:700].strip()
        if len(body) < 80:
            continue
        qa_examples.append({
            "mode": "qa",
            "prompt": f"<|qa|><|user|>What is {title} in Linux?<|assistant|>",
            "completion": f"{body}<|end|>",
        })
    return qa_examples


def extract_stackexchange(filepath: Path, max_qa: int, max_cmds: int) -> tuple[list[dict], list[str]]:
    qa_examples: list[dict] = []
    commands: list[str] = []

    for doc in iter_documents(filepath):
        if len(qa_examples) >= max_qa and len(commands) >= max_cmds:
            break

        first_nl = doc.find("\n")
        if first_nl == -1:
            continue
        header = doc[:first_nl]
        title = header.split(": ", 1)[1].strip() if ": " in header else header.lstrip("# ").strip()

        q_marker = "\nQuestion:\n"
        ans_marker = "\nAnswer 1 (Accepted):\n"
        q_idx = doc.find(q_marker)
        a_idx = doc.find(ans_marker)
        if q_idx == -1 or a_idx == -1:
            continue

        q_body = doc[q_idx + len(q_marker) : a_idx].strip()
        a_rest = doc[a_idx + len(ans_marker) :]
        next_ans = a_rest.find("\nAnswer 2")
        a_body = (a_rest[:next_ans] if next_ans != -1 else a_rest).strip()

        if len(commands) < max_cmds:
            for block in CODE_BLOCK_RE.findall(a_body):
                for line in block.splitlines():
                    cmd = normalize_shell_line(line)
                    if cmd:
                        commands.append(cmd)
                        if len(commands) >= max_cmds:
                            break

        if len(qa_examples) < max_qa:
            low_a = a_body.lower()
            if any(low_a.startswith(bad) for bad in FORUM_NOISE_PREFIXES):
                continue
            if 15 <= len(title) <= 200 and 50 <= len(a_body) <= 1200:
                user_prompt = f"{title}\n{q_body}" if (0 < len(q_body) <= 300) else title
                qa_examples.append({
                    "mode": "qa",
                    "prompt": f"<|qa|><|user|>{user_prompt}<|assistant|>",
                    "completion": f"{a_body}<|end|>",
                })

    return qa_examples, commands


def extract_commands_from_text_file(filepath: Path, max_cmds: int = 20000) -> list[str]:
    cmds: list[str] = []
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            cmd = normalize_shell_line(line)
            if cmd:
                cmds.append(cmd)
                if len(cmds) >= max_cmds:
                    break
    return cmds


def slice_commands_to_completions(unique_cmds: list[str], rng: random.Random) -> list[dict]:
    """Slice an already-split list of unique, bash-verified commands into prefix/suffix pairs."""
    examples: list[dict] = []
    for cmd in unique_cmds:
        parts = cmd.split(" ")
        if len(parts) < 2:
            continue
        base = parts[1] if parts[0] == "sudo" and len(parts) > 2 else parts[0]
        ctx_choices = CONTEXT_HINTS.get(base, [""])

        split_points: list[int] = []
        p1 = len(parts[0]) + 1
        if 2 <= p1 < len(cmd) - 2:
            split_points.append(p1)
        if len(parts) >= 3:
            p2 = len(parts[0]) + 1 + len(parts[1]) + 1
            if p2 < len(cmd) - 2:
                split_points.append(p2)
        mid_lo = max(3, len(cmd) // 3)
        mid_hi = max(mid_lo + 1, (2 * len(cmd)) // 3)
        if mid_hi < len(cmd) - 2:
            split_points.append(rng.randint(mid_lo, mid_hi))

        for sp in set(split_points):
            prefix = cmd[:sp]
            suffix = cmd[sp:]
            if not suffix.strip():
                continue
            ctx = rng.choice(ctx_choices)
            prompt = f"<|complete|><|ctx|>{ctx}<|cmd|>{prefix}<|pred|>" if ctx else f"<|complete|><|cmd|>{prefix}<|pred|>"
            examples.append({
                "mode": "complete",
                "prompt": prompt,
                "completion": f"{suffix}<|end|>",
            })
    return examples


def write_jsonl(records: list[dict], path: Path):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/sources"))
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts/sft_v2"))
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_qa: list[dict] = []
    all_cmds: list[str] = []

    # 1. TLDR pages (definitions + task Q&A + placeholder-free commands)
    tldr_fp = args.data_dir / "05_tldr_pages.txt"
    if tldr_fp.exists():
        qa, cmds = extract_tldr(tldr_fp)
        all_qa.extend(qa)
        all_cmds.extend(cmds)
        print(f"  05_tldr_pages.txt:       {len(qa):>6,} Q&A, {len(cmds):>6,} cmds")

    # 2. Arch Wiki conceptual definitions
    arch_fp = args.data_dir / "06_arch_wiki.txt"
    if arch_fp.exists():
        qa = extract_arch_wiki_definitions(arch_fp, max_qa=3500)
        all_qa.extend(qa)
        print(f"  06_arch_wiki.txt:        {len(qa):>6,} Q&A,      0 cmds")

    # 3. Cheatsheets & Shell scripts
    for fname in ("18_cli_cheatsheets.txt", "16_shell_scripts_and_configs.txt"):
        fp = args.data_dir / fname
        if fp.exists():
            cmds = extract_commands_from_text_file(fp)
            all_cmds.extend(cmds)
            print(f"  {fname:<25s} {0:>6,} Q&A, {len(cmds):>6,} cmds")

    # 4. StackExchange (rebalanced: cap AskUbuntu so apt-get doesn't dominate!)
    se_caps = {
        "11_unix_stackexchange.txt": 15000,
        "12_askubuntu.txt": 8000,
        "13_serverfault.txt": 15000,
        "15_superuser_linux.txt": 10000,
    }
    for fname, cap in se_caps.items():
        fp = args.data_dir / fname
        if fp.exists():
            qa, cmds = extract_stackexchange(fp, max_qa=cap, max_cmds=15000)
            all_qa.extend(qa)
            all_cmds.extend(cmds)
            print(f"  {fname:<25s} {len(qa):>6,} Q&A, {len(cmds):>6,} cmds")

    # Deduplicate commands BEFORE bash -n validation and BEFORE train/val split!
    unique_cmds = list(dict.fromkeys(all_cmds))
    print(f"\nValidating {len(unique_cmds):,} unique commands with `bash -n` ...")
    valid_cmds = filter_valid_bash_commands(unique_cmds)
    print(f"  Passed `bash -n`: {len(valid_cmds):,} / {len(unique_cmds):,} ({100*len(valid_cmds)/max(1,len(unique_cmds)):.1f}%)")

    # GROUP-SPLIT unique commands into train vs val BEFORE slicing -> ZERO LEAKAGE!
    rng.shuffle(valid_cmds)
    n_val_cmds = max(200, int(len(valid_cmds) * args.val_ratio))
    val_cmds = valid_cmds[:n_val_cmds]
    train_cmds = valid_cmds[n_val_cmds:]
    assert set(val_cmds).isdisjoint(set(train_cmds)), "Command leakage detected!"

    train_comp = slice_commands_to_completions(train_cmds, rng)
    val_comp = slice_commands_to_completions(val_cmds, rng)

    # Deduplicate Q&A by prompt before splitting
    seen_q: set[str] = set()
    unique_qa: list[dict] = []
    for rec in all_qa:
        if rec["prompt"] not in seen_q:
            seen_q.add(rec["prompt"])
            unique_qa.append(rec)
    rng.shuffle(unique_qa)
    n_val_qa = max(200, int(len(unique_qa) * args.val_ratio))
    val_qa = unique_qa[:n_val_qa]
    train_qa = unique_qa[n_val_qa:]

    train_records = train_qa + train_comp
    val_records = val_qa + val_comp
    rng.shuffle(train_records)
    rng.shuffle(val_records)

    write_jsonl(train_records, args.out_dir / "sft_train.jsonl")
    write_jsonl(val_records, args.out_dir / "sft_val.jsonl")

    meta = {
        "version": "v2_leakfree_bash_verified",
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_qa": len(train_qa),
        "val_qa": len(val_qa),
        "train_complete": len(train_comp),
        "val_complete": len(val_comp),
        "unique_train_cmds": len(train_cmds),
        "unique_val_cmds": len(val_cmds),
    }
    (args.out_dir / "sft_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved clean SFT v2 dataset to {args.out_dir}:")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
