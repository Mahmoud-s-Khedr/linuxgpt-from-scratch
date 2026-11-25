# LinuxGPT from scratch

A learning project implementing GPT-2-style and LLaMA-style decoder models in PyTorch, then pretraining and fine-tuning them for Linux questions and shell autocomplete. This is Experiment A of LinuxGPT; [Experiment B](https://github.com/Mahmoud-s-Khedr/linuxgpt-finetuning) starts from pretrained models instead.

The two tasks are Linux Q&A and prediction of the exact missing suffix after a shell command prefix. A plausible-looking command is insufficient: spaces, quoting, filenames and cursor position matter.

## Included

- GPT-2-style learned positions, LayerNorm and GELU; LLaMA-style RoPE, RMSNorm, SwiGLU and grouped-query attention.
- Shared model presets, custom byte-level BPE tokenizer and KV-cache generation.
- Source-weighted pretraining, dual-mode SFT with prompt loss masking, and command-group splitting before completion slicing.
- A local CLI, optional Colab launchers, [architecture study notes](docs/GPT2_LINE_BY_LINE.md), and [historical evaluation outputs](evaluation/README.md).

Datasets, tokenizer artifacts, checkpoints and runtime logs are generated locally and excluded from Git. There is no pretrained LinuxGPT model bundled with this repository.

## Setup

Python 3.10+ and Bash are required. From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python sanity_check.py
```

For GPU training, install a CUDA-compatible PyTorch build for your machine. The archived CPU evaluation used Python 3.12, torch 2.15.0+cpu, numpy 2.5.5 and tokenizers 0.23.3; exact recorded versions are in the evaluation metadata.

## Prepare data

Place your own cleaned text files in `data/sources/`. Documents are separated by a line containing `---`. Tokenizer training and pretraining read `*.txt`; named pretraining sources receive configured weights, and SFT extraction expects the recognized sources described in [the data format](docs/DATA.md).

```bash
python tokenizer.py --data_dir data/sources --out artifacts/tokenizer.json
python prepare_pretrain.py --data_dir data/sources --tokenizer artifacts/tokenizer.json --out_dir artifacts/pretrain
python build_sft_data.py --data_dir data/sources --out_dir artifacts/sft_v2
```

Inspect `artifacts/sft_v2/sft_meta.json` before training. The current corpus-scale splitter reserves at least 200 unique commands and 200 Q&A prompts for validation, so small inputs can leave an empty training split. Bash syntax filtering is a heuristic; it does not verify executable names, flags, intent or safety. Batched parsing also does not guarantee every extracted command parses independently.

## Train

Start with `--preset tiny` and a short run to check the pipeline. A representative 15M-preset run is:

```bash
python train_pretrain.py --arch llama --preset 15m --max_steps 2000
python train_sft.py --arch llama --preset 15m --init_ckpt runs/pretrain_llama_15m/best.pt --sft_dir artifacts/sft_v2 --max_steps 1200
```

Use `--arch gpt2` for the other architecture. Preset names are approximate size labels: the evaluated GPT-2/LLaMA 15M models contain 17.1M/15.7M parameters, and the 50M presets contain 40.4M/36.6M.

## Ask Linux questions from the CLI (Q&A)

Run these commands from the repository root with your virtual environment activated. First train a model as above, or copy your trained checkpoint and its matching tokenizer into this checkout. The repository contains source and saved evaluation responses; it does not include model weights.

### Ask one question

For the LLaMA 15M checkpoint produced by the training commands above:

```bash
python cli.py --checkpoint runs/sft_llama_15m/sft_best.pt --tokenizer artifacts/tokenizer.json "How do I find large files?"
```

Q&A is the default mode. The question must be quoted as one shell argument. The CLI prints the model's answer; generated commands are not executed. `--tokenizer` defaults to `artifacts/tokenizer.json` and must match the checkpoint's training tokenizer.

For a GPT-2 checkpoint trained with `--arch gpt2`, change the checkpoint path:

```bash
python cli.py --checkpoint runs/sft_gpt2_15m/sft_best.pt "What is the difference between chmod and chown?"
```

To control answer length and use greedy decoding:

```bash
python cli.py --checkpoint runs/sft_llama_15m/sft_best.pt --max-tokens 256 --temp 0 "How do I check free disk space?"
```

### Interactive Q&A

Load the model once, then enter questions at the prompt:

```bash
python cli.py --checkpoint runs/sft_llama_15m/sft_best.pt --chat
```

Type `quit` or `exit` to stop. Each question is answered independently; previous turns are not included as conversation history.

### Use a downloaded Colab checkpoint

The Colab runner stores checkpoints under `runs/colab_<arch>_<preset>/`. List them and select one by name:

```bash
python cli.py --list
python cli.py --model llama_50m "How do I find large files?"
python cli.py --model gpt2_15m --chat
```

These names work after you download the corresponding trained checkpoint. The default stage is SFT; `--stage pretrain` selects a pretraining checkpoint for comparison.

## Autocomplete

```bash
python cli.py --checkpoint runs/sft_llama_15m/sft_best.pt --complete --ctx 'files=compose.yml' 'docker compose '
```

The CLI displays suggestions; it does not execute generated commands. This is a CLI prototype, not an installed Bash/Zsh inline widget.

## Optional Colab automation

The launchers use the separate `google-colab-cli` tool (0.8.0 in the original environment), authenticated in your own environment. They are not standalone Colab notebooks. Set `COLAB_BIN` if its executable is outside `PATH`; job wrappers also accept `PYTHON_BIN`.

```bash
python run_one_vm.py --arch llama --preset 15m --gpu L4
# Or: bash jobs/run_llama_15m.sh
```

These commands allocate GPU sessions and train models. They require the prepared artifacts above and Colab credits. The launcher stops its created session in `finally` unless `--keep_alive` is requested. Cloud allocation and completion are not guaranteed; verify your sessions after failures.

## Measured results

Historical SFT checkpoints, greedy decoding, warm CPU inference, four threads:

| Model | Q&A rubric hits /54 | Strict suffix matches /74 | Median completion ms |
|---|---:|---:|---:|
| GPT-2 15M | 2 | 1 | 65 |
| LLaMA 15M | 2 | 1 | 67 |
| GPT-2 50M | 2 | 0 | 138 |
| LLaMA 50M | 6 | 0 | 147 |

These models demonstrate the training pipeline but are not reliable Linux assistants. Loss reductions and a few plausible samples did not translate into useful task performance. The published source includes later generation-mode/data-builder fixes and defaults to SFT v2; these historical checkpoints were not retrained for this publication. The table is not a measurement of a new v2 training run.

Recompute the frozen rubric scores without downloading weights:

```bash
python evaluation/score_saved.py
```

The suite has 138 cases: 54 ordinary scored Q&A, 74 constrained completions, eight open-ended completions and two long-input diagnostics. Rubric hits are not verified accuracy; valid alternatives and incorrect extra claims require manual review. See [evaluation details](evaluation/README.md) and the [LLaMA study notes](docs/LLAMA_LINE_BY_LINE.md).
