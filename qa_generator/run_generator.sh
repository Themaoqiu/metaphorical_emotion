#!/usr/bin/env bash

set -a
source /home/wangxingjian/metaphorical_emotion/.env
set +a

cd /home/wangxingjian/metaphorical_emotion

python3 -m qa_generator.meta_generator run \
  --input="/home/wangxingjian/data/metaphor/YesBut/yesbut_benchmark.json" \
  --output="/home/wangxingjian/metaphorical_emotion/output/yesbut_output.jsonl" \
  --model="gemini-3-flash-preview" \
  --dataset="yesbut" \
  --image_root="/home/wangxingjian/data/metaphor/YesBut" \
  --max_concurrent=50 \
  --max_retries=5 \
  --limit=5
