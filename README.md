# Metaphorical Emotion
## Environment
```bash
cd metaphorical_emotion
uv sync
```

This repository keeps the project-specific training and evaluation entrypoints, but it does not vendor the full upstream runtimes they depend on.

Environment requirements:
- `trainer/run.sh` must be executed inside a local [LLaMA-Factory](/root/metaphorical_emotion/trainer/run.sh) checkout. It expects a working LLaMA-Factory virtualenv and launches `llamafactory-cli train`.
- `eval/eval.sh` must be executed with the DORO-STVG runtime available locally. It activates the DORO-STVG environment first, then runs this repository's `eval/eval.py`.
- The corresponding environment snapshots are stored in [trainer/ENVIRONMENT.md](/root/metaphorical_emotion/trainer/ENVIRONMENT.md) and [eval/ENVIRONMENT.md](/root/metaphorical_emotion/eval/ENVIRONMENT.md), together with upstream and locked dependency lists.
- A source-only copy of LLaMA-Factory is vendored under [trainer/LLaMA-Factory](/root/metaphorical_emotion/trainer/LLaMA-Factory). Large local artifacts such as `.venv/`, `saves/`, and `.git/` are intentionally excluded.

## Training
Training code is organized under `trainer/`.

```bash
cd trainer
bash run.sh
```

Prerequisite:
- A local LLaMA-Factory checkout is required. By default the script looks for `/root/autodl-tmp/OneMLLM/LLaMA-Factory`.

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

Prerequisite:
- A local DORO-STVG checkout and its Python environment are required. By default the script looks for `/root/autodl-tmp/DORO-STVG` and `/root/.virtualenvs/stvg/dev`.

Useful environment variables:
- `STVG_PROJECT_DIR`: local DORO-STVG path
- `STVG_ENV_PATH`: evaluation virtualenv path
- `EVAL_DATA_ROOT`: fixed test dataset root
- `EVAL_OUTPUT_DIR`: directory for metrics and csv outputs
