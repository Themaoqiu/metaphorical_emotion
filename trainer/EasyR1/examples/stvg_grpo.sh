#!/bin/bash
set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_local_fs_capacity_threshold=0.999

# Prepare curriculum-sorted train/val splits (only runs once).
if [ ! -f data/stvg/train_curriculum.jsonl ]; then
    python3 examples/prepare_stvg.py \
        --input /home/wangxingjian/data/compstvg/compstvg_rl_2of7.sharegpt.with_difficulty.jsonl \
        --output data/stvg/train_curriculum.jsonl \
        --val_output data/stvg/val.jsonl \
        --val_ratio 0.05 \
        --curriculum
fi

# Launch GRPO training.
python3 -m verl.trainer.main \
    config=examples/stvg_grpo.yaml \
    data.train_files=data/stvg/train_curriculum.jsonl \
    data.val_files=data/stvg/train_curriculum.jsonl \
    data.video_dir=/home/wangxingjian/data/compstvg \
    trainer.experiment_name=stvg_grpo_curriculum
