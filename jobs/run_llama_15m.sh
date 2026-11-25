#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON_BIN:-python}" run_one_vm.py --arch llama --preset 15m --gpu L4 "$@"
