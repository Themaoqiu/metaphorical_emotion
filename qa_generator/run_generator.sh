#!/usr/bin/env bash

set -a
source /home/wangxingjian/metaphorical_emotion/.env
set +a


# 2) Paste absolute paths here.
INPUT_PATH="/home/wangxingjian/data/metaphor/metmeme/test_english.jsonl"
OUTPUT_PATH="/home/wangxingjian/metaphorical_emotion/output/test_english_output.jsonl"
IMAGE_ROOT="/home/wangxingjian/data/metaphor/metmeme/image/English"

# 3) Model and runtime params.
MODEL_NAME="gemini-3-flash-preview"
MAX_CONCURRENT=50
MAX_RETRIES=5
LIMIT=5

cd /home/wangxingjian/metaphorical_emotion

python3 -m qa_generator.meta_generator run \
  --input="${INPUT_PATH}" \
  --output="${OUTPUT_PATH}" \
  --model="${MODEL_NAME}" \
  --image_root="${IMAGE_ROOT}" \
  --max_concurrent="${MAX_CONCURRENT}" \
  --max_retries="${MAX_RETRIES}" \
  --limit="${LIMIT}"
