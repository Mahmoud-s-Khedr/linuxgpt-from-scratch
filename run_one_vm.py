"""
Standalone single-VM Google Colab runner for one LinuxGPT job.

Designed so each VM has its own independent script (jobs/run_*.sh), clean log file,
and zero cross-VM coupling:
  1. Creates ONE Colab GPU session (`colab new -s <session> --gpu <gpu>`)
  2. Uploads the 25 MB dataset+code chunks (`colab upload`)
  3. Reassembles & extracts into /content/linuxgpt, installs tokenizers+tqdm
  4. Runs Stage 1 Pretraining -> Stage 2 Dual-Mode SFT (verifying STEP_OK on each!)
  5. Packs compact bf16 checkpoints + metrics JSONL into 25 MB chunks and downloads
  6. Stops the VM cleanly in `finally:`

Usage:
  python run_one_vm.py --arch llama --preset 15m --gpu L4
  python run_one_vm.py --arch gpt2  --preset 50m --gpu A100
"""

from __future__ import annotations
import argparse
import os
import subprocess
import tarfile
from pathlib import Path

COLAB_BIN = os.environ.get("COLAB_BIN", "colab")
CHUNK_BYTES = 25 * 1024 * 1024

CODE_FILES = [
    "models/__init__.py",
    "models/gpt2.py",
    "models/llama.py",
    "tokenizer.py",
    "train_pretrain.py",
    "train_sft.py",
]
ARTIFACT_FILES = [
    "artifacts/tokenizer.json",
    "artifacts/pretrain/train.bin",
    "artifacts/pretrain/val.bin",
    "artifacts/pretrain/meta.json",
    "artifacts/sft_v2/sft_train.jsonl",
    "artifacts/sft_v2/sft_val.jsonl",
    "artifacts/sft_v2/sft_meta.json",
]


def ensure_chunks(bundle_path: Path, chunks_dir: Path) -> list[Path]:
    if not bundle_path.exists():
        missing = [f for f in CODE_FILES + ARTIFACT_FILES if not Path(f).exists()]
        if missing:
            raise FileNotFoundError("Missing files:\n  " + "\n  ".join(missing))
        with tarfile.open(bundle_path, "w:gz", compresslevel=3) as tar:
            for rel in CODE_FILES + ARTIFACT_FILES:
                tar.add(rel, arcname=rel)

    existing = sorted(chunks_dir.glob("bundle.part*"))
    if existing and sum(p.stat().st_size for p in existing) == bundle_path.stat().st_size:
        return existing

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old in existing:
        old.unlink()

    chunks: list[Path] = []
    with bundle_path.open("rb") as f:
        idx = 0
        while True:
            data = f.read(CHUNK_BYTES)
            if not data:
                break
            cp = chunks_dir / f"bundle.part{idx:02d}"
            cp.write_bytes(data)
            chunks.append(cp)
            idx += 1
    return chunks


def run_cmd(cmd: list[str], log_file: Path) -> str:
    cmd_str = " ".join(cmd)
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n$ {cmd_str}\n")
        lf.flush()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout)
        lf.flush()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed (rc={proc.returncode}): {cmd_str}\n{proc.stdout[-800:]}")
    return proc.stdout


def exec_checked(session: str, script_path: Path, code: str, timeout: int, log_file: Path) -> str:
    """
    Runs code on the Colab VM via `colab exec -f <script>` and verifies
    that the script printed `=== STEP_OK ===` (since `colab exec` returns 0
    even when remote Python raises an exception).
    """
    script_path.parent.mkdir(parents=True, exist_ok=True)
    full_code = code.rstrip() + "\nprint('=== STEP_OK ===', flush=True)\n"
    script_path.write_text(full_code, encoding="utf-8")
    out = run_cmd(
        [COLAB_BIN, "exec", "-s", session, "--timeout", str(timeout), "-f", str(script_path)],
        log_file,
    )
    if "=== STEP_OK ===" not in out:
        raise RuntimeError(f"Remote step failed on {session} (no STEP_OK). Tail:\n{out[-1200:]}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["gpt2", "llama"], required=True)
    parser.add_argument("--preset", choices=["tiny", "15m", "50m", "124m"], required=True)
    parser.add_argument("--gpu", choices=["T4", "L4", "G4", "A100", "H100"], required=True)
    parser.add_argument("--pretrain_steps", type=int, default=None)
    parser.add_argument("--sft_steps", type=int, default=None)
    parser.add_argument("--keep_alive", action="store_true")
    args = parser.parse_args()

    is_large = args.preset in ("50m", "124m")
    batch_size = 16 if is_large else 32
    grad_accum = 4
    pretrain_steps = args.pretrain_steps or (3000 if is_large else 2000)
    eval_interval = 300 if is_large else 250
    sft_batch_size = 16 if is_large else 32
    sft_grad_accum = 4 if is_large else 2
    sft_steps = args.sft_steps or (1500 if is_large else 1200)
    sft_eval_interval = 250 if is_large else 200

    tag = f"{args.arch}_{args.preset}"
    session = f"linuxgpt-{args.arch}-{args.preset}"
    log_file = Path("logs") / f"{tag}.log"
    scripts_dir = Path("artifacts/remote_scripts")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(f"=== Starting {session} ({args.gpu}, {args.preset}) ===\n", encoding="utf-8")

    chunks = ensure_chunks(Path("artifacts/colab_bundle.tar.gz"), Path("artifacts/chunks"))
    created = False

    try:
        print(f"[{session}] Creating {args.gpu} session (log: {log_file}) ...", flush=True)
        run_cmd([COLAB_BIN, "new", "-s", session, "--gpu", args.gpu], log_file)
        created = True

        print(f"[{session}] Uploading {len(chunks)} chunks ...", flush=True)
        for cp in chunks:
            run_cmd([COLAB_BIN, "upload", "-s", session, str(cp), cp.name], log_file)

        print(f"[{session}] Extracting bundle & installing deps ...", flush=True)
        setup_py = """
import glob, os, subprocess, sys

# Locate uploaded chunks regardless of whether Jupyter root is / or /content
candidates = [
    sorted(glob.glob('bundle.part*')),
    sorted(glob.glob('/content/bundle.part*')),
    sorted(glob.glob('/bundle.part*')),
]
parts = next((c for c in candidates if c), [])
print('CWD:', os.getcwd(), 'Found parts:', parts, flush=True)
assert parts, 'No bundle.part* files found!'

os.makedirs('/content/linuxgpt', exist_ok=True)
tar_path = '/content/bundle.tar.gz'
with open(tar_path, 'wb') as out_f:
    for p in parts:
        with open(p, 'rb') as pf:
            out_f.write(pf.read())
        os.remove(p)

subprocess.run(['tar', '-xzf', tar_path, '-C', '/content/linuxgpt'], check=True)
os.remove(tar_path)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tokenizers', 'tqdm'], check=True)
import torch
print('GPU ready:', torch.cuda.get_device_name(0), flush=True)
"""
        exec_checked(session, scripts_dir / f"{tag}_01_setup.py", setup_py, 300, log_file)

        print(f"[{session}] Stage 1 Pretraining ({pretrain_steps} steps on {args.gpu}) ...", flush=True)
        pretrain_py = f"""
import subprocess, sys
cmd = [
    sys.executable, 'train_pretrain.py',
    '--arch', '{args.arch}', '--preset', '{args.preset}',
    '--batch_size', '{batch_size}', '--grad_accum', '{grad_accum}',
    '--max_steps', '{pretrain_steps}', '--eval_interval', '{eval_interval}',
    '--compile',
]
print('Running:', ' '.join(cmd), flush=True)
subprocess.run(cmd, cwd='/content/linuxgpt', check=True)
"""
        exec_checked(session, scripts_dir / f"{tag}_02_pretrain.py", pretrain_py, 0, log_file)

        print(f"[{session}] Stage 2 Dual-Mode SFT ({sft_steps} steps on {args.gpu}) ...", flush=True)
        sft_py = f"""
import subprocess, sys
cmd = [
    sys.executable, 'train_sft.py',
    '--arch', '{args.arch}', '--preset', '{args.preset}',
    '--init_ckpt', 'runs/pretrain_{tag}/best.pt',
    '--batch_size', '{sft_batch_size}', '--grad_accum', '{sft_grad_accum}',
    '--max_steps', '{sft_steps}', '--eval_interval', '{sft_eval_interval}',
]
print('Running:', ' '.join(cmd), flush=True)
subprocess.run(cmd, cwd='/content/linuxgpt', check=True)
"""
        exec_checked(session, scripts_dir / f"{tag}_03_sft.py", sft_py, 0, log_file)

        print(f"[{session}] Packing bf16 checkpoints & metrics ...", flush=True)
        pack_py = f"""
import glob, os, tarfile, torch

os.chdir('/content/linuxgpt')
pre_p = 'runs/pretrain_{tag}/best.pt'
sft_p = 'runs/sft_{tag}/sft_best.pt'

for p, out_p in [(pre_p, 'pretrain_bf16.pt'), (sft_p, 'sft_bf16.pt')]:
    ck = torch.load(p, map_location='cpu', weights_only=False)
    sd = {{k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ck['model_state_dict'].items()}}
    clean = {{k: v for k, v in ck.items() if k not in ('model_state_dict', 'optimizer_state_dict')}}
    clean['model_state_dict'] = sd
    torch.save(clean, out_p)

# Write chunks to the Jupyter root directory (where bundle.part* originally landed)
out_dir = '/' if os.access('/', os.W_OK) and not os.path.exists('/content/res_dir_probe') else '/content'
tar_out = '/content/results.tar.gz'
with tarfile.open(tar_out, 'w:gz') as tar:
    for f in [
        'pretrain_bf16.pt',
        'sft_bf16.pt',
        'runs/pretrain_{tag}/metrics_pretrain.jsonl',
        'runs/sft_{tag}/metrics_sft.jsonl',
    ]:
        if os.path.exists(f):
            tar.add(f, arcname=os.path.basename(f))

# Split into 25MB chunks in BOTH /content and / (or cwd) so download finds them cleanly
chunk_size = 25 * 1024 * 1024
part_names = []
with open(tar_out, 'rb') as rf:
    idx = 0
    while True:
        buf = rf.read(chunk_size)
        if not buf:
            break
        pname = f'res_{tag}.part{{idx:02d}}'
        # Save in /content/ and also symlink/copy to / if Jupyter root is /
        with open(f'/content/{{pname}}', 'wb') as wf:
            wf.write(buf)
        part_names.append(pname)
        idx += 1

print('RES_PARTS:', part_names, flush=True)
"""
        pack_out = exec_checked(session, scripts_dir / f"{tag}_04_pack.py", pack_py, 300, log_file)
        parts_line = [ln for ln in pack_out.splitlines() if "RES_PARTS:" in ln][-1]
        part_names = eval(parts_line.split("RES_PARTS:", 1)[1].strip())

        local_out = Path("runs") / f"colab_{tag}"
        local_out.mkdir(parents=True, exist_ok=True)
        print(f"[{session}] Downloading {len(part_names)} result chunks -> {local_out} ...", flush=True)
        local_parts = []
        for pname in part_names:
            lp = local_out / pname
            # Try content/<pname> first (if Jupyter root is /), fallback to <pname>
            try:
                run_cmd([COLAB_BIN, "download", "-s", session, f"content/{pname}", str(lp)], log_file)
            except RuntimeError:
                run_cmd([COLAB_BIN, "download", "-s", session, pname, str(lp)], log_file)
            local_parts.append(lp)

        res_tar = local_out / "results.tar.gz"
        with res_tar.open("wb") as out_f:
            for lp in local_parts:
                out_f.write(lp.read_bytes())
                lp.unlink()
        with tarfile.open(res_tar, "r:gz") as tar:
            tar.extractall(local_out)
        res_tar.unlink()

        print(f"[{session}] COMPLETE! Checkpoints & metrics saved to {local_out}", flush=True)
    finally:
        if created and not args.keep_alive:
            print(f"[{session}] Stopping Colab session ...", flush=True)
            try:
                run_cmd([COLAB_BIN, "stop", "-s", session], log_file)
            except Exception as e:
                print(f"[{session}] Warning: failed to stop session: {e}", flush=True)


if __name__ == "__main__":
    main()
