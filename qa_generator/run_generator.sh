#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

set -a
source "${REPO_ROOT}/.env"
set +a

python3 -m qa_generator.cot_generator \
  --dataset metmeme \
  --input /home/wangxingjian/MetaThinker/output/vflutetocot.jsonl \
  --output /home/wangxingjian/MetaThinker/data/vflute.jsonl \
  --model deepseek-v3.2 \
  --max_concurrent 50 \
  --max_retries 5 \
  --start 1 \
  --end 3

# python3 -m qa_generator.meta_generator run \
#   --input /home/wangxingjian/MetaThinker/output/vflute.jsonl \
#   --output /home/wangxingjian/metaphorical_emotion/output/vflutetocot.jsonl \
#   --model qwen3.5-plus \
#   --dataset vflute \
#   --image_root /home/wangxingjian/data/metaphor/V-FLUTE/images \
#   --max_concurrent 50 \
#   --max_retries 5 \
#   --start 1 \
#   --end 400 \
#   --limit 10


# python3 -m qa_generator.caption_generator run \
#   --input /home/wangxingjian/data/metaphor/V-FLUTE/vflute_full.jsonl \
#   --output /home/wangxingjian/MetaThinker/output/vflute.jsonl \
#   --dataset vflute \
#   --api_model_name qwen3.5-plus \
#   --image_root /home/wangxingjian/data/metaphor/V-FLUTE/images \
#   --max_concurrent 50 \
#   --max_retries 5 \
#   --generate_limit 11
