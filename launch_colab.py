"""
Parallel Google Colab Launcher for LinuxGPT (GPT-2 vs. LLaMA, 15M & 50M).

Runs Colab GPU VMs in parallel (default 2 concurrent, with staggered session
creation + retry if Colab temporarily refuses an allocation) so all 4 jobs
complete hands-free:
  - linuxgpt-gpt2-15m   (L4)
  - linuxgpt-llama-15m  (L4)
  - linuxgpt-gpt2-50m   (A100)
  - linuxgpt-llama-50m  (A100)

Each VM runs Stage 1 Pretraining -> Stage 2 Dual-Mode SFT -> downloads compact
bf16 checkpoints & metrics JSONL -> auto-stops the VM.
"""

from __future__ import annotations
import argparse
import os
import concurrent.futures
from dataclasses import dataclass
import subprocess
import tarfile
import threading
import time
from pathlib import Path

COLAB_BIN = os.environ.get("COLAB_BIN", "colab")
CHUNK_BYTES = 25 * 1024 * 1024  # 25 MB chunks (safe for Jupyter Contents API)
ALLOC_LOCK = threading.Lock()

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


@dataclass
class JobSpec:
    arch: str
    preset: str
    gpu: str
    batch_size: int
    grad_accum: int
    pretrain_steps: int
    eval_interval: int
    sft_batch_size: int
    sft_grad_accum: int
    sft_steps: int
    sft_eval_interval: int
    compile: bool = True

    @property
    def session(self) -> str:
        return f"linuxgpt-{self.arch}-{self.preset}"

    @property
    def tag(self) -> str:
        return f"{self.arch}_{self.preset}"


def build_bundle_chunks(bundle_path: Path, chunks_dir: Path) -> list[Path]:
    """Pack code + artifacts into tar.gz (reusing if already built) and split into 25 MB chunks."""
    if not bundle_path.exists():
        print(f"Packing code + prepared artifacts into {bundle_path} ...", flush=True)
        missing = [f for f in CODE_FILES + ARTIFACT_FILES if not Path(f).exists()]
        if missing:
            raise FileNotFoundError(
                "Missing required files before Colab launch:\n  " + "\n  ".join(missing)
            )
        with tarfile.open(bundle_path, "w:gz", compresslevel=3) as tar:
            for rel in CODE_FILES + ARTIFACT_FILES:
                tar.add(rel, arcname=rel)
    print(f"Bundle ready: {bundle_path} ({bundle_path.stat().st_size / 1e6:.1f} MB)", flush=True)

    existing = sorted(chunks_dir.glob("bundle.part*"))
    if existing and sum(p.stat().st_size for p in existing) == bundle_path.stat().st_size:
        print(f"Reusing {len(existing)} existing chunks in {chunks_dir}", flush=True)
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
    print(f"Split into {len(chunks)} chunks of <= {CHUNK_BYTES // (1024*1024)} MB in {chunks_dir}", flush=True)
    return chunks


def run_cmd(cmd: list[str], log_file: Path | None = None, dry_run: bool = False):
    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"  [DRY-RUN] {cmd_str}")
        return
    if log_file is not None:
        with log_file.open("a", encoding="utf-8") as lf:
            lf.write(f"\n$ {cmd_str}\n")
            lf.flush()
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=True)
            return
    subprocess.run(cmd, check=True)


def exec_script(session: str, script_path: Path, code: str, timeout: int, log_file: Path, dry_run: bool):
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code, encoding="utf-8")
    run_cmd([COLAB_BIN, "exec", "-s", session, "--timeout", str(timeout), "-f", str(script_path)], log_file, dry_run)


def create_session_with_retry(session: str, gpu: str, log_file: Path, dry_run: bool, max_attempts: int = 8):
    """Stagger `colab new` calls and retry if Colab temporarily refuses a concurrent slot."""
    for attempt in range(1, max_attempts + 1):
        with ALLOC_LOCK:
            try:
                run_cmd([COLAB_BIN, "new", "-s", session, "--gpu", gpu], log_file, dry_run)
                if not dry_run:
                    time.sleep(5)  # brief stagger before next worker calls `colab new`
                return
            except subprocess.CalledProcessError:
                if attempt == max_attempts:
                    raise
        wait_s = 60
        print(f"[{session}] Allocation busy (attempt {attempt}/{max_attempts}), retrying in {wait_s}s ...", flush=True)
        time.sleep(wait_s)


def run_worker(
    job: JobSpec,
    chunks: list[Path],
    keep_alive: bool = False,
    dry_run: bool = False,
) -> str:
    session = job.session
    log_file = Path("logs") / f"{job.tag}.log"
    scripts_dir = Path("artifacts/remote_scripts")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        log_file.write_text(f"=== Starting {session} ({job.gpu}, {job.preset}) ===\n")

    print(f"[{session}] Creating {job.gpu} session (logs -> {log_file}) ...", flush=True)
    created = False
    try:
        create_session_with_retry(session, job.gpu, log_file, dry_run)
        created = True

        print(f"[{session}] Uploading {len(chunks)} chunks ...", flush=True)
        for cp in chunks:
            run_cmd([COLAB_BIN, "upload", "-s", session, str(cp), cp.name], log_file, dry_run)

        print(f"[{session}] Reassembling bundle & installing deps ...", flush=True)
        setup_py = (
            "import subprocess, os\n"
            "os.makedirs('/content/linuxgpt', exist_ok=True)\n"
            "subprocess.run('cat /content/bundle.part* > /content/bundle.tar.gz && rm -f /content/bundle.part*', shell=True, check=True)\n"
            "subprocess.run(['tar', '-xzf', '/content/bundle.tar.gz', '-C', '/content/linuxgpt'], check=True)\n"
            "os.remove('/content/bundle.tar.gz')\n"
            "subprocess.run(['pip', 'install', '-q', 'tokenizers', 'tqdm'], check=True)\n"
            "import torch; print('GPU ready:', torch.cuda.get_device_name(0), flush=True)\n"
        )
        exec_script(session, scripts_dir / f"{job.tag}_01_setup.py", setup_py, 300, log_file, dry_run)

        # Stage 1: Pretraining
        compile_flag = "--compile" if job.compile else ""
        pretrain_py = (
            "import subprocess\n"
            f"cmd = 'cd /content/linuxgpt && python train_pretrain.py "
            f"--arch {job.arch} --preset {job.preset} "
            f"--batch_size {job.batch_size} --grad_accum {job.grad_accum} "
            f"--max_steps {job.pretrain_steps} --eval_interval {job.eval_interval} {compile_flag}'\n"
            "print('Running:', cmd, flush=True)\n"
            "subprocess.run(cmd, shell=True, check=True)\n"
        )
        print(f"[{session}] Stage 1 Pretraining ({job.pretrain_steps} steps on {job.gpu}) ...", flush=True)
        exec_script(session, scripts_dir / f"{job.tag}_02_pretrain.py", pretrain_py, 0, log_file, dry_run)

        # Stage 2: SFT
        sft_py = (
            "import subprocess\n"
            f"cmd = 'cd /content/linuxgpt && python train_sft.py "
            f"--arch {job.arch} --preset {job.preset} "
            f"--init_ckpt runs/pretrain_{job.arch}_{job.preset}/best.pt "
            f"--batch_size {job.sft_batch_size} --grad_accum {job.sft_grad_accum} "
            f"--max_steps {job.sft_steps} --eval_interval {job.sft_eval_interval}'\n"
            "print('Running:', cmd, flush=True)\n"
            "subprocess.run(cmd, shell=True, check=True)\n"
        )
        print(f"[{session}] Stage 2 Dual-Mode SFT ({job.sft_steps} steps on {job.gpu}) ...", flush=True)
        exec_script(session, scripts_dir / f"{job.tag}_03_sft.py", sft_py, 0, log_file, dry_run)

        # Pack compact bf16 checkpoints + metrics into 25 MB chunks for reliable download
        pack_py = (
            "import torch, tarfile, os, glob\n"
            "os.chdir('/content/linuxgpt')\n"
            f"pre_p = 'runs/pretrain_{job.arch}_{job.preset}/best.pt'\n"
            f"sft_p = 'runs/sft_{job.arch}_{job.preset}/sft_best.pt'\n"
            "for p, out_p in [(pre_p, 'pretrain_bf16.pt'), (sft_p, 'sft_bf16.pt')]:\n"
            "    ck = torch.load(p, map_location='cpu', weights_only=False)\n"
            "    sd = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v) for k, v in ck['model_state_dict'].items()}\n"
            "    clean = {k: v for k, v in ck.items() if k not in ('model_state_dict', 'optimizer_state_dict')}\n"
            "    clean['model_state_dict'] = sd\n"
            "    torch.save(clean, out_p)\n"
            "with tarfile.open('/content/results.tar.gz', 'w:gz') as tar:\n"
            f"    for f in ['pretrain_bf16.pt', 'sft_bf16.pt', 'runs/pretrain_{job.arch}_{job.preset}/metrics_pretrain.jsonl', 'runs/sft_{job.arch}_{job.preset}/metrics_sft.jsonl']:\n"
            "        if os.path.exists(f): tar.add(f, arcname=os.path.basename(f))\n"
            "sz = os.path.getsize('/content/results.tar.gz') / 1e6\n"
            "print(f'Packed results.tar.gz: {sz:.1f} MB', flush=True)\n"
            "os.system('split -b 25m -d /content/results.tar.gz /content/res.part')\n"
            "print('RES_PARTS:', sorted(os.path.basename(x) for x in glob.glob('/content/res.part*')), flush=True)\n"
        )
        print(f"[{session}] Packing results for download ...", flush=True)
        exec_script(session, scripts_dir / f"{job.tag}_04_pack.py", pack_py, 300, log_file, dry_run)

        local_out = Path("runs") / f"colab_{job.tag}"
        local_out.mkdir(parents=True, exist_ok=True)
        if not dry_run:
            log_txt = log_file.read_text(encoding="utf-8", errors="ignore")
            parts_line = [ln for ln in log_txt.splitlines() if "RES_PARTS:" in ln][-1]
            part_names = eval(parts_line.split("RES_PARTS:", 1)[1].strip())
            local_parts = []
            for pname in part_names:
                lp = local_out / pname
                run_cmd([COLAB_BIN, "download", "-s", session, pname, str(lp)], log_file, dry_run)
                local_parts.append(lp)
            res_tar = local_out / "results.tar.gz"
            with res_tar.open("wb") as out_f:
                for lp in local_parts:
                    out_f.write(lp.read_bytes())
                    lp.unlink()
            with tarfile.open(res_tar, "r:gz") as tar:
                tar.extractall(local_out)
            res_tar.unlink()

        print(f"[{session}] COMPLETE! Results saved in {local_out}", flush=True)
        return f"{session}: OK -> {local_out}"
    except Exception as e:
        print(f"[{session}] FAILED: {e} (see {log_file})", flush=True)
        return f"{session}: FAILED ({e})"
    finally:
        if created and not keep_alive:
            print(f"[{session}] Stopping Colab session to save compute units ...", flush=True)
            try:
                run_cmd([COLAB_BIN, "stop", "-s", session], log_file, dry_run)
            except Exception as stop_err:
                print(f"[{session}] Warning: failed to stop session: {stop_err}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all4", action="store_true", help="Run GPT-2 & LLaMA at 15m/L4 and 50m/A100")
    parser.add_argument("--gpu", default="L4", choices=["T4", "L4", "A100", "H100"])
    parser.add_argument("--preset", default="15m", choices=["tiny", "15m", "50m", "124m"])
    parser.add_argument("--max_parallel", type=int, default=2, help="Max concurrent Colab GPU sessions")
    parser.add_argument("--keep_alive", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.all4:
        jobs = [
            JobSpec("gpt2",  "15m", "L4",   batch_size=32, grad_accum=4, pretrain_steps=2000, eval_interval=250, sft_batch_size=32, sft_grad_accum=2, sft_steps=1200, sft_eval_interval=200),
            JobSpec("llama", "15m", "L4",   batch_size=32, grad_accum=4, pretrain_steps=2000, eval_interval=250, sft_batch_size=32, sft_grad_accum=2, sft_steps=1200, sft_eval_interval=200),
            JobSpec("gpt2",  "50m", "A100", batch_size=16, grad_accum=4, pretrain_steps=3000, eval_interval=300, sft_batch_size=16, sft_grad_accum=4, sft_steps=1500, sft_eval_interval=250),
            JobSpec("llama", "50m", "A100", batch_size=16, grad_accum=4, pretrain_steps=3000, eval_interval=300, sft_batch_size=16, sft_grad_accum=4, sft_steps=1500, sft_eval_interval=250),
        ]
    else:
        is_50m = args.preset in ("50m", "124m")
        jobs = [
            JobSpec(
                arch, args.preset, args.gpu,
                batch_size=16 if is_50m else 32,
                grad_accum=4,
                pretrain_steps=3000 if is_50m else 2000,
                eval_interval=300 if is_50m else 250,
                sft_batch_size=16 if is_50m else 32,
                sft_grad_accum=4 if is_50m else 2,
                sft_steps=1500 if is_50m else 1200,
                sft_eval_interval=250 if is_50m else 200,
            )
            for arch in ("gpt2", "llama")
        ]

    bundle_path = Path("artifacts/colab_bundle.tar.gz")
    chunks_dir = Path("artifacts/chunks")
    chunks = [] if args.dry_run else build_bundle_chunks(bundle_path, chunks_dir)

    workers = min(args.max_parallel, len(jobs))
    print(f"\nLaunching {len(jobs)} Colab GPU jobs ({workers} concurrent) ...\n", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run_worker, job, chunks, args.keep_alive, args.dry_run) for job in jobs]
        results = [f.result() for f in concurrent.futures.as_completed(futs)]

    print("\n=== Summary ===")
    for r in sorted(results):
        print(" ", r)


if __name__ == "__main__":
    main()
