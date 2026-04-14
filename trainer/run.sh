#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-/root/autodl-tmp/OneMLLM/LLaMA-Factory}"
VENV_DIR="${VENV_DIR:-$LLAMA_FACTORY_DIR/.venv}"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_CLI="$VENV_DIR/bin/llamafactory-cli"
BASE_CONFIG="$SCRIPT_DIR/qwen3_vl_metaphor_emotion.yaml"
GENERATED_DIR="$SCRIPT_DIR/generated"
GENERATED_CONFIG="$GENERATED_DIR/qwen3_vl_metaphor_emotion.rendered.yaml"

TRAIN_MODEL_PATH="${TRAIN_MODEL_PATH:-/root/autodl-tmp/Qwen3-VL-4B-Instruct}"
TRAIN_DATASET_DIR="${TRAIN_DATASET_DIR:-/root/autodl-tmp/OneMLLM/LLaMA-Factory/data}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-/root/autodl-tmp/OneMLLM/LLaMA-Factory/saves/qwen3-vl-4b/full/metaphor_emotion}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"

mkdir -p "$LOG_DIR" "$GENERATED_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [ ! -d "$LLAMA_FACTORY_DIR" ]; then
    echo "[FATAL] LLAMA_FACTORY_DIR does not exist: $LLAMA_FACTORY_DIR"
    echo "[FATAL] run.sh depends on a local LLaMA-Factory checkout."
    exit 1
fi

if [ ! -f "$BASE_CONFIG" ]; then
    echo "[FATAL] Missing base config: $BASE_CONFIG"
    exit 1
fi

if [ ! -x "$VENV_PYTHON" ] || [ ! -x "$VENV_CLI" ]; then
    echo "[FATAL] Missing virtualenv runtime: $VENV_DIR"
    echo "[FATAL] run.sh must be executed with the LLaMA-Factory virtualenv available."
    exit 1
fi

sed \
    -e "s#__MODEL_PATH__#$TRAIN_MODEL_PATH#g" \
    -e "s#__DATASET_DIR__#$TRAIN_DATASET_DIR#g" \
    -e "s#__OUTPUT_DIR__#$TRAIN_OUTPUT_DIR#g" \
    "$BASE_CONFIG" > "$GENERATED_CONFIG"

echo "Logging to $LOG_FILE"
echo "Using config $GENERATED_CONFIG" | tee -a "$LOG_FILE"

cd "$LLAMA_FACTORY_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Environment Check ===" | tee -a "$LOG_FILE"
which python | tee -a "$LOG_FILE"
python -V 2>&1 | tee -a "$LOG_FILE"
which llamafactory-cli | tee -a "$LOG_FILE"
if ! python -m pip show deepspeed 2>&1 | tee -a "$LOG_FILE"; then
    echo "[FATAL] DeepSpeed is not available in current environment. Aborting before torchrun." | tee -a "$LOG_FILE"
    exit 1
fi

"$VENV_CLI" train "$GENERATED_CONFIG" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "SUCCESS" | tee -a "$LOG_FILE"
    if [ "$AUTO_SHUTDOWN" = "1" ]; then
        shutdown
    fi
else
    echo "FAILED" | tee -a "$LOG_FILE"
fi

sync
