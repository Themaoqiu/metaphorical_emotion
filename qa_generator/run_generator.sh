#!/usr/bin/env bash

set -a
source /home/wangxingjian/metaphorical_emotion/.env
set +a

cd /home/wangxingjian/metaphorical_emotion

python3 -m qa_generator.meta_generator run \
  --input /home/wangxingjian/data/metaphor/metmeme/test_english.jsonl \
  --output /home/wangxingjian/metaphorical_emotion/output/metmeme_output.jsonl \
  --model gemini-3-flash-preview \
  --dataset metmeme \
  --image_root /home/wangxingjian/data/metaphor/metmeme/image/English \
  --max_concurrent 50 \
  --max_retries 5 \
  --limit 50
