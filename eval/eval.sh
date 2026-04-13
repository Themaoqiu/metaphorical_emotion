#!/bin/bash
set -Eeuo pipefail

MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: bash eval.sh /path/to/checkpoint"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STVG_PROJECT_DIR="${STVG_PROJECT_DIR:-/root/autodl-tmp/DORO-STVG}"
STVG_ENV_PATH="${STVG_ENV_PATH:-/root/.virtualenvs/stvg/dev}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

cleanup() {
    local exit_code=$?
    echo "============================================"
    if [[ $exit_code -eq 0 ]]; then
        echo "Current checkpoint evaluation finished."
        if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
            shutdown
        fi
    else
        echo "Current checkpoint evaluation failed (exit code: $exit_code)."
    fi
    echo "============================================"
}
trap cleanup EXIT

if [ ! -d "$STVG_PROJECT_DIR" ]; then
    echo "[FATAL] STVG_PROJECT_DIR does not exist: $STVG_PROJECT_DIR"
    exit 1
fi

if [ ! -d "$STVG_ENV_PATH" ]; then
    echo "[FATAL] STVG_ENV_PATH does not exist: $STVG_ENV_PATH"
    exit 1
fi

cd "$STVG_PROJECT_DIR"
source scripts/activate_env.sh "$STVG_ENV_PATH"
cd "$SCRIPT_DIR"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
python eval.py --model_path "$MODEL_PATH"
