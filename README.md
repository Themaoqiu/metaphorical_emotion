# Metaphorical Emotion
## Environment
```bash
cd metaphorical_emotion
uv sync
```

## Training
Training code is organized under `trainer/`.

```bash
cd trainer
bash run.sh
```

Useful environment variables:
- `LLAMA_FACTORY_DIR`: local LLaMA-Factory path
- `TRAIN_MODEL_PATH`: base model path
- `TRAIN_DATASET_DIR`: dataset directory
- `TRAIN_OUTPUT_DIR`: checkpoint output directory
- `AUTO_SHUTDOWN=1`: shut down after training succeeds

## Evaluation
Evaluation code is organized under `eval/`.

```bash
cd eval
bash eval.sh /path/to/checkpoint
```

Useful environment variables:
- `STVG_PROJECT_DIR`: local DORO-STVG path
- `STVG_ENV_PATH`: evaluation virtualenv path
- `EVAL_DATA_ROOT`: fixed test dataset root
- `EVAL_OUTPUT_DIR`: directory for metrics and csv outputs
