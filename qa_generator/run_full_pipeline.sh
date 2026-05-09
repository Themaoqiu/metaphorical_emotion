#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

set -a
source "$REPO_ROOT/.env"
set +a

CAPTION_MODEL="qwen3.5-plus"
META_MODEL="qwen3.5-plus"
COT_MODEL="deepseek-v3.2"
MAX_CONCURRENT=50
MAX_RETRIES=5
LIMIT=1

run_caption() {
  local dataset="$1"
  local input_path="$2"
  local output_path="$3"
  local image_root="$4"

  echo "[pipeline] caption dataset=$dataset output=$output_path"
  python3 -m qa_generator.caption_generator run \
    --input "$input_path" \
    --output "$output_path" \
    --dataset "$dataset" \
    --api_model_name "$CAPTION_MODEL" \
    --image_root "$image_root" \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_retries "$MAX_RETRIES" \
    --generate_limit "$LIMIT"
}

run_meta() {
  local dataset="$1"
  local input_path="$2"
  local output_path="$3"
  local image_root="$4"

  echo "[pipeline] meta dataset=$dataset output=$output_path"
  python3 -m qa_generator.meta_generator run \
    --input "$input_path" \
    --output "$output_path" \
    --model "$META_MODEL" \
    --dataset "$dataset" \
    --image_root "$image_root" \
    --max_concurrent "$MAX_CONCURRENT" \
    --max_retries "$MAX_RETRIES" \
    --limit "$LIMIT"
}

run_cot() {
  local dataset="$1"
  local input_path="$2"
  local output_path="$3"
  local image_root=""
  if [[ $# -ge 4 ]]; then
    image_root="$4"
  fi

  echo "[pipeline] cot dataset=$dataset output=$output_path"
  if [[ -n "$image_root" ]]; then
    python3 -m qa_generator.cot_generator \
      "$dataset" \
      "$input_path" \
      "$output_path" \
      "$COT_MODEL" \
      --image_root "$image_root" \
      --max_concurrent "$MAX_CONCURRENT" \
      --max_retries "$MAX_RETRIES" \
      --limit "$LIMIT"
  else
    python3 -m qa_generator.cot_generator \
      "$dataset" \
      "$input_path" \
      "$output_path" \
      "$COT_MODEL" \
      --max_concurrent "$MAX_CONCURRENT" \
      --max_retries "$MAX_RETRIES" \
      --limit "$LIMIT"
  fi
}


CIIBENCH_INPUT="/home/wangxingjian/data/metaphor/CII-Bench/ciibench.jsonl"
CIIBENCH_IMAGE_ROOT="/home/wangxingjian/data/metaphor/CII-Bench/images"
CIIBENCH_WORK_FILE="/home/wangxingjian/MetaThinker/output/ciibench_full.jsonl"

IMAGEMET_INPUT="/home/wangxingjian/data/metaphor/imagemet/train.jsonl"
IMAGEMET_IMAGE_ROOT="/home/wangxingjian/data/metaphor/imagemet/images"
IMAGEMET_WORK_FILE="/home/wangxingjian/MetaThinker/output/imagemet_full.jsonl"

MEMECAP_INPUT="/home/wangxingjian/data/metaphor/memecap/memes-trainval.json"
MEMECAP_IMAGE_ROOT="/home/wangxingjian/data/metaphor/memecap/memes"
MEMECAP_WORK_FILE="/home/wangxingjian/MetaThinker/output/memecap_full.jsonl"

METMEME_INPUT="/home/wangxingjian/data/metaphor/metmeme/data/test-00000-of-00001.jsonl"
METMEME_IMAGE_ROOT="/home/wangxingjian/data/metaphor/metmeme/image"
METMEME_WORK_FILE="/home/wangxingjian/MetaThinker/output/metmeme_full.jsonl"

VFLUTE_INPUT="/home/wangxingjian/data/metaphor/V-FLUTE/vflute_full.jsonl"
VFLUTE_IMAGE_ROOT="/home/wangxingjian/data/metaphor/V-FLUTE/images"
VFLUTE_WORK_FILE="/home/wangxingjian/MetaThinker/output/vflute_full.jsonl"

# run_caption "ciibench" "$CIIBENCH_INPUT" "$CIIBENCH_WORK_FILE" "$CIIBENCH_IMAGE_ROOT"
# run_cot "ciibench" "$CIIBENCH_WORK_FILE" "$CIIBENCH_WORK_FILE"

# run_caption "imagemet" "$IMAGEMET_INPUT" "$IMAGEMET_WORK_FILE" "$IMAGEMET_IMAGE_ROOT"
# run_meta "imagemet" "$IMAGEMET_WORK_FILE" "$IMAGEMET_WORK_FILE" "$IMAGEMET_IMAGE_ROOT"
# run_cot "imagemet" "$IMAGEMET_WORK_FILE" "$IMAGEMET_WORK_FILE"

# run_caption "memecap" "$MEMECAP_INPUT" "$MEMECAP_WORK_FILE" "$MEMECAP_IMAGE_ROOT"
# run_meta "memecap" "$MEMECAP_WORK_FILE" "$MEMECAP_WORK_FILE" "$MEMECAP_IMAGE_ROOT"
# run_cot "memecap" "$MEMECAP_WORK_FILE" "$MEMECAP_WORK_FILE"

run_caption "metmeme" "$METMEME_INPUT" "$METMEME_WORK_FILE" "$METMEME_IMAGE_ROOT"
run_meta "metmeme" "$METMEME_WORK_FILE" "$METMEME_WORK_FILE" "$METMEME_IMAGE_ROOT"
run_cot "metmeme" "$METMEME_WORK_FILE" "$METMEME_WORK_FILE" "$METMEME_IMAGE_ROOT"

# run_caption "vflute" "$VFLUTE_INPUT" "$VFLUTE_WORK_FILE" "$VFLUTE_IMAGE_ROOT"
# run_meta "vflute" "$VFLUTE_WORK_FILE" "$VFLUTE_WORK_FILE" "$VFLUTE_IMAGE_ROOT"
# run_cot "vflute" "$VFLUTE_WORK_FILE" "$VFLUTE_WORK_FILE"

echo "[pipeline] all datasets completed"
