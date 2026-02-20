#!/usr/bin/env bash

set -a
source /home/wangxingjian/metaphorical_emotion/.env
set +a

cd /home/wangxingjian/metaphorical_emotion

python3 -m qa_generator.meta_generator run \
  --input /home/wangxingjian/metaphorical_emotion/output/yesbut_output.jsonl \
  --output /home/wangxingjian/metaphorical_emotion/output/yesbut_output.jsonl \
  --model qwen3-vl-flash \
  --dataset yesbut \
  --image_root /home/wangxingjian/data/metaphor/YesBut/images \
  --max_concurrent 50 \
  --max_retries 5 \
  --start 1 \
  --end 400 \
  --limit 400

# python3 -m qa_generator.caption_generator run \
#   --input /home/wangxingjian/metaphorical_emotion/output/hummus_parallel_output.jsonl \
#   --output /home/wangxingjian/metaphorical_emotion/output/hummus_parallel_with_caption.jsonl \
#   --model_path Qwen/Qwen3-VL-8B-Instruct \
#   --image_root /home/wangxingjian/data/metaphor/hummus

# python3 -m qa_generator.caption_generator run \
#   --input /home/wangxingjian/metaphorical_emotion/output/metmeme_output.jsonl \
#   --output /home/wangxingjian/metaphorical_emotion/output/metmeme_output.jsonl \
#   --provider api \
#   --api_model_name qwen3-vl-flash \
#   --image_root /home/wangxingjian/data/metaphor/metmeme/image \
#   --max_concurrent 50 \
#   --max_retries 5 \
#   --generate_limit 400
