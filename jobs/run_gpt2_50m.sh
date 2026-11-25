#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" run_one_vm.py --arch gpt2 --preset 50m --gpu A100 "$@"
