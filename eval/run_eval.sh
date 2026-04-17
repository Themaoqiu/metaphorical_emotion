#!/bin/bash
set -Eeuo pipefail


export CUDA_VISIBLE_DEVICES=0

python main.py run \
    --model_name qwen3.5 \
    --model_path /home/wangxingjian/model/qwen3.5-9b \
    --data_name multimm \
    --annotation_path /home/wangxingjian/data/metaphor/MultiMM/data \
    --image_dir /home/wangxingjian/data/metaphor/MultiMM/data \
    --output_dir /home/wangxingjian/MetaThinker/eval/results \
    --batch_size 64 \
    --max_tokens 2048 \
    --max_model_len 8192 \
    --temperature 0.0 \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.9 \
    --num_rounds 10 \
    --cn_sample_size 440 \
    --en_sample_size 407 \
    --random_seed 42
