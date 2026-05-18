#!/bin/bash
# GRPO training on YesBut metaphor-emotion dataset.
# Hyperparameters mirror Video-Thinker's GRPO recipe (lr=5e-6, kl_coef=0.04,
# n=8 rollouts, max_pixels=401408, max_response_length=1024, max_grad_norm=5,
# weight_decay=0.01, 2 epochs).

set -x

MODEL_PATH=Qwen/Qwen3-VL-7B-Instruct

# 1) Build train/val splits from the raw jsonl (only needs to run once).
#    Pass multiple files to --inputs to mix datasets (yesbut + hummus + metmeme, …).

if [ ! -f data/emotion_grpo/train.jsonl ]; then
    python3 examples/prepare_emotion.py \
        --inputs \
            imagemet.jsonl \
            memecap.jsonl \
            metmeme.jsonl \
            vflute.jsonl \
        --out_dir data/emotion_grpo \
        --val_ratio 0.05
fi

# 2) Launch GRPO.
python3 -m verl.trainer.main \
    config=examples/emotion_grpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen3_vl_yesbut_grpo \
    trainer.n_gpus_per_node=8
